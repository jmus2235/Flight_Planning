import arcpy
import os
import pandas as pd
import argparse
import sys
import tempfile

def calculate_dem_statistics(polygon_shapefile, dem_raster, output_csv, zone_field=None):
    """
    Calculate DEM statistics for each polygon feature in a shapefile.
    
    Requirements: Input shapefiles must be in WGS84 (not UTM)
    
    Parameters: Both the input DEM and cropped buffer areas polygon shapefiles should be in the same projection (UTM recommended)
    -----------
    polygon_shapefile : str
        Path to the input polygon shapefile
    dem_raster : str
        Path to the input DEM raster (GeoTIFF)
    output_csv : str
        Path to the output CSV file
    zone_field : str, optional
        Field to use as the zone field. If None, the first appropriate field will be used.
    """
    # Set workspace environment
    arcpy.env.overwriteOutput = True
    
    # Validate input files
    if not arcpy.Exists(polygon_shapefile):
        print(f"Error: Polygon shapefile does not exist at {polygon_shapefile}")
        return None
    
    if not arcpy.Exists(dem_raster):
        print(f"Error: DEM raster does not exist at {dem_raster}")
        return None
    
    # Check for Spatial Analyst extension
    if arcpy.CheckExtension("Spatial") == "Available":
        arcpy.CheckOutExtension("Spatial")
    else:
        print("Error: Spatial Analyst extension is required but not available.")
        return None
    
    try:
        print(f"Processing zonal statistics...")
        print(f"Polygon shapefile: {polygon_shapefile}")
        print(f"DEM raster: {dem_raster}")
        
        # Extract feature information from the shapefile for cross-referencing
        feature_info = {}
        field_names = [f.name for f in arcpy.ListFields(polygon_shapefile)]
        
        # # Check if FLNUM_1 exists
        # use_flnum = "FLNUM_1" in field_names
        # # If not, check if FLNUM exists as an alternative
        # if not use_flnum and "FLNUM" in field_names:
        #     use_flnum = True
        #     flnum_field = "FLNUM"
        # else:
        #     flnum_field = "FLNUM_1"
        
        # Check if Name_1 exists
        use_flnum = "Name_1" in field_names
        # If not, check if FLNUM exists as an alternative
        if not use_flnum and "Name" in field_names:
            use_flnum = True
            flnum_field = "Name"
        else:
            flnum_field = "Name_1"
        
        
        # Get field information
        fields_to_extract = ["FID", "OID@"]
        if use_flnum:
            fields_to_extract.append(flnum_field)
        
        # Extract the fields we need
        with arcpy.da.SearchCursor(polygon_shapefile, fields_to_extract) as cursor:
            for row in cursor:
                fid = row[0]  # FID
                oid = row[1]  # OBJECTID
                
                feature_info[fid] = {
                    "OBJECTID": oid
                }
                
                if use_flnum:
                    flnum_value = row[2]  # FLNUM_1 or FLNUM
                    feature_info[fid][flnum_field] = flnum_value
        
        print(f"Extracted feature information for {len(feature_info)} features")
        
        # Find an integer field to use for zonal statistics
        fields = arcpy.ListFields(polygon_shapefile)
        integer_fields = [field.name for field in fields if field.type in ["Integer", "SmallInteger", "OID"]]
        
        if not integer_fields:
            print("No integer fields found. Creating a temporary shapefile with an integer field.")
            
            # Create a temporary directory and shapefile
            temp_dir = tempfile.mkdtemp()
            temp_shapefile = os.path.join(temp_dir, "temp_poly.shp")
            
            # Copy the shapefile
            arcpy.CopyFeatures_management(polygon_shapefile, temp_shapefile)
            
            # Add an integer field
            arcpy.AddField_management(temp_shapefile, "ZONE_ID", "LONG")
            
            # Populate the field
            with arcpy.da.UpdateCursor(temp_shapefile, ["ZONE_ID", "FID"]) as cursor:
                for row in cursor:
                    row[0] = row[1] + 1  # Use FID+1 to avoid zero values
                    cursor.updateRow(row)
            
            # Use the new shapefile and field
            polygon_shapefile = temp_shapefile
            use_field = "ZONE_ID"
        else:
            # Use the specified zone field or the first integer field
            if zone_field and zone_field in integer_fields:
                use_field = zone_field
            else:
                use_field = integer_fields[0]
            
            print(f"Using integer field '{use_field}' for zonal statistics")
        
        # Create a temporary output table
        temp_table = "in_memory\\zonal_stats"
        
        # Run ZonalStatisticsAsTable
        print(f"Running ZonalStatisticsAsTable...")
        arcpy.sa.ZonalStatisticsAsTable(
            polygon_shapefile, 
            use_field,
            dem_raster,
            temp_table,
            "DATA",
            "ALL"  # Calculate all statistics
        )
        
        print(f"Zonal statistics calculated. Processing results...")
        
        # Get the output field that contains the zone value
        value_field = None
        for field in arcpy.ListFields(temp_table):
            if field.name.upper() in ["VALUE", use_field.upper()]:
                value_field = field.name
                break
        
        if not value_field:
            print("Warning: Could not find VALUE field in output table. Using first field.")
            value_field = arcpy.ListFields(temp_table)[0].name
        
        # Get all fields from the output table
        all_output_fields = [field.name for field in arcpy.ListFields(temp_table)]
        
        # Convert table to pandas DataFrame
        df = pd.DataFrame.from_records(
            data=[row for row in arcpy.da.SearchCursor(temp_table, all_output_fields)],
            columns=all_output_fields
        )
        
        # If ZONE_ID was used, we need to subtract 1 to get back to the original FID
        if use_field == "ZONE_ID":
            df[value_field] = df[value_field] - 1
        
        # Add the FLNUM_1 column based on the feature_info dictionary
        if use_flnum:
            df[flnum_field] = df[value_field].apply(
                lambda x: feature_info.get(x, {}).get(flnum_field, "UNKNOWN")
            )
        
        # Rename columns to match the desired output format
        column_mapping = {
            'OBJECTID': 'OBJECTID',
            value_field: 'FID',
            'COUNT': 'COUNT',
            'AREA': 'AREA',
            'MIN': 'MIN',
            'MAX': 'MAX',
            'RANGE': 'RANGE',
            'MEAN': 'MEAN',
            'STD': 'STD',
            'SUM': 'SUM',
            'MEDIAN': 'MEDIAN',
            'PCT90': 'PCT90'
        }
        
        # Map column names (case-insensitive)
        for old_col, new_col in column_mapping.items():
            for col in df.columns:
                if col.upper() == old_col.upper():
                    df = df.rename(columns={col: new_col})
                    break
        
        # Reorder columns to match the example output
        desired_order = ["OBJECTID"]
        
        if use_flnum:
            desired_order.append(flnum_field)
        
        desired_order.extend(["COUNT", "AREA", "MIN", "MAX", "RANGE", "MEAN", "STD", "SUM", "MEDIAN", "PCT90"])
        
        # Keep only columns that exist in the DataFrame
        final_order = [col for col in desired_order if col in df.columns]
        
        # Add any additional columns that weren't in the desired_order
        for col in df.columns:
            if col not in final_order and col != "FID":  # Skip FID since we've mapped it
                final_order.append(col)
        
        # Reorder the columns
        df = df[final_order]
        
        # Save to CSV
        df.to_csv(output_csv, index=False, float_format='%.6f')
        
        print(f"Processing complete. Results saved to {output_csv}")
        
        # Print a sample of the output and the column mapping
        print("\nOutput sample:")
        print(df.head().to_string())
        
        print("\nField mappings:")
        if use_flnum:
            print(f"{flnum_field} field is included for cross-referencing")
        
        return df  # Return the DataFrame for optional further processing
        
    except arcpy.ExecuteError:
        print(f"Error: {arcpy.GetMessages(2)}")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Clean up
        if arcpy.Exists("in_memory\\zonal_stats"):
            arcpy.Delete_management("in_memory\\zonal_stats")
        
        # Check in the Spatial Analyst extension
        arcpy.CheckInExtension("Spatial")
        
        # Clean up temporary shapefile if created
        if 'temp_shapefile' in locals() and temp_shapefile and os.path.exists(temp_shapefile):
            try:
                arcpy.Delete_management(temp_shapefile)
                os.rmdir(os.path.dirname(temp_shapefile))
            except:
                pass

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Calculate DEM statistics for polygon features')
    parser.add_argument('-p', '--polygon', required=True, help='Path to polygon shapefile')
    parser.add_argument('-d', '--dem', required=True, help='Path to DEM raster file')
    parser.add_argument('-o', '--output', required=True, help='Path to output CSV file')
    parser.add_argument('-z', '--zone', required=False, help='Field to use as zone field')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the main function
    calculate_dem_statistics(args.polygon, args.dem, args.output, args.zone)

if __name__ == "__main__":
    main()