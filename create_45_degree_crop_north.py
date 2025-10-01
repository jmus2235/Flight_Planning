import arcpy
import os
import math
import time

def create_45_degree_crop(input_shapefile, output_shapefile):
    """
    Crops north-facing polygons creating a triangular area with 45-degree angles from the center of the base
    The apex of the triangle is at the bottom, with the base at the top of the polygon
    
    Requirements: Input shapefiles must be in WGS84 (not UTM)
    
    Parameters:
    input_shapefile - Path to the input shapefile containing polygons
    output_shapefile - Path to save the output cropped polygons
    """
    arcpy.env.overwriteOutput = True
    
    # Create a timestamp for temporary files
    timestamp = int(time.time())
    
    # Create a temporary workspace
    workspace = os.path.dirname(output_shapefile)
    temp_dir = os.path.join(workspace, "temp_crop_files")
    
    # Create directory if it doesn't exist
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # Get fields from input to preserve in output
    fields = arcpy.ListFields(input_shapefile)
    field_info = [(field.name, field.type, field.length, field.precision, field.scale, field.isNullable) 
                for field in fields if not field.required]
    
    # Create output feature class
    desc = arcpy.Describe(input_shapefile)
    spatial_ref = desc.spatialReference
    
    arcpy.CreateFeatureclass_management(
        os.path.dirname(output_shapefile),
        os.path.basename(output_shapefile),
        "POLYGON",
        spatial_reference=spatial_ref
    )
    
    # Add fields to output
    for field_name, field_type, field_length, field_precision, field_scale, field_nullable in field_info:
        arcpy.AddField_management(
            output_shapefile,
            field_name,
            field_type,
            field_precision,
            field_scale,
            field_length,
            field_name,
            field_nullable
        )
    
    # Create list of field names for cursors
    input_fields = ["SHAPE@", "OID@"] + [field[0] for field in field_info]
    output_fields = ["SHAPE@"] + [field[0] for field in field_info]
    
    # Process each polygon
    with arcpy.da.SearchCursor(input_shapefile, input_fields) as cursor:
        for row in cursor:
            polygon = row[0]
            oid = row[1]
            attributes = row[2:]
            
            print(f"Processing polygon {oid}")
            
            try:
                # Get extent
                extent = polygon.extent
                
                # Create triangular areas to cut
                # We'll use a more direct approach focused on creating the trimming polygons
                
                # 1. Identify the base points (should be at the bottom for north-facing polygons)
                # We'll use a feature class to collect the vertices
                vertices_fc = os.path.join(temp_dir, f"vertices_{timestamp}_{oid}.shp")
                arcpy.CreateFeatureclass_management(
                    os.path.dirname(vertices_fc),
                    os.path.basename(vertices_fc),
                    "POINT",
                    spatial_reference=spatial_ref
                )
                
                # Add XY fields to help us sort
                arcpy.AddField_management(vertices_fc, "POINT_X", "DOUBLE")
                arcpy.AddField_management(vertices_fc, "POINT_Y", "DOUBLE")
                
                # Extract all vertices to the point feature class
                with arcpy.da.InsertCursor(vertices_fc, ["SHAPE@XY", "POINT_X", "POINT_Y"]) as insert_cursor:
                    for part in polygon:
                        for point in part:
                            if point:
                                insert_cursor.insertRow([(point.X, point.Y), point.X, point.Y])
                
                # Find the minimum Y value
                stats_table = os.path.join(temp_dir, f"stats_{timestamp}_{oid}.dbf")
                arcpy.Statistics_analysis(vertices_fc, stats_table, [["POINT_Y", "MIN"]])
                
                # Find the correct field name for minimum Y
                min_field = None
                for field in arcpy.ListFields(stats_table):
                    if 'MIN' in field.name.upper() and 'POINT_Y' in field.name.upper():
                        min_field = field.name
                        break
                
                # If we didn't find a field with both MIN and POINT_Y, look for just MIN
                if not min_field:
                    for field in arcpy.ListFields(stats_table):
                        if 'MIN' in field.name.upper():
                            min_field = field.name
                            break
                
                if not min_field:
                    raise ValueError(f"Could not find MIN field in statistics output for polygon {oid}")
                
                min_y = 0
                with arcpy.da.SearchCursor(stats_table, [min_field]) as stats_cursor:
                    for stats_row in stats_cursor:
                        min_y = stats_row[0]
                        break
                
                # Select points at or near the minimum Y value
                arcpy.MakeFeatureLayer_management(vertices_fc, "vertices_lyr")
                arcpy.SelectLayerByAttribute_management("vertices_lyr", "NEW_SELECTION", 
                                                     f"POINT_Y <= {min_y + 0.1}")
                
                # Get the leftmost and rightmost points from the selected set
                x_stats_table = os.path.join(temp_dir, f"x_stats_{timestamp}_{oid}.dbf")
                arcpy.Statistics_analysis("vertices_lyr", x_stats_table, 
                                        [["POINT_X", "MIN"], ["POINT_X", "MAX"]])
                
                # Find correct field names for min and max X
                min_x_field = None
                max_x_field = None
                
                for field in arcpy.ListFields(x_stats_table):
                    if 'MIN' in field.name.upper() and 'POINT_X' in field.name.upper():
                        min_x_field = field.name
                    elif 'MAX' in field.name.upper() and 'POINT_X' in field.name.upper():
                        max_x_field = field.name
                
                # If we didn't find specific fields, look for just MIN and MAX
                if not min_x_field:
                    for field in arcpy.ListFields(x_stats_table):
                        if 'MIN' in field.name.upper():
                            min_x_field = field.name
                            break
                
                if not max_x_field:
                    for field in arcpy.ListFields(x_stats_table):
                        if 'MAX' in field.name.upper():
                            max_x_field = field.name
                            break
                
                if not min_x_field or not max_x_field:
                    raise ValueError(f"Could not find MIN/MAX X fields in statistics output for polygon {oid}")
                
                min_x = 0
                max_x = 0
                with arcpy.da.SearchCursor(x_stats_table, [min_x_field, max_x_field]) as x_stats_cursor:
                    for x_stats_row in x_stats_cursor:
                        min_x = x_stats_row[0]
                        max_x = x_stats_row[1]
                        break
                
                # Get the maximum Y value for the polygon
                max_y = extent.YMax
                
                # Calculate the center point at the base
                center_x = (min_x + max_x) / 2
                
                # Calculate the horizontal offset for 45 degrees
                # For a 45-degree angle, the horizontal offset equals the height
                height = max_y - min_y
                horizontal_offset = height
                
                # Create a triangular clipping polygon
                clip_polygon_fc = os.path.join(temp_dir, f"clip_poly_{timestamp}_{oid}.shp")
                arcpy.CreateFeatureclass_management(
                    os.path.dirname(clip_polygon_fc),
                    os.path.basename(clip_polygon_fc),
                    "POLYGON",
                    spatial_reference=spatial_ref
                )
                
                # Create a triangular polygon that represents the area to KEEP
                with arcpy.da.InsertCursor(clip_polygon_fc, ["SHAPE@"]) as clip_cursor:
                    # Create a triangle with 45-degree angles to the center base
                    # The point (apex) of the triangle is at the bottom
                    clip_array = arcpy.Array([
                        # Bottom point (apex)
                        arcpy.Point(center_x, min_y),
                        # Top-left corner
                        arcpy.Point(center_x - horizontal_offset, max_y),
                        # Top-right corner
                        arcpy.Point(center_x + horizontal_offset, max_y),
                        # Close the polygon
                        arcpy.Point(center_x, min_y)
                    ])
                    clip_cursor.insertRow([arcpy.Polygon(clip_array, spatial_ref)])
                
                # Clip the original polygon with our triangular clipping polygon
                clip_fc = os.path.join(temp_dir, f"clip_{timestamp}_{oid}.shp")
                arcpy.Clip_analysis(polygon, clip_polygon_fc, clip_fc)
                
                # Add the clipped polygon to the output
                with arcpy.da.SearchCursor(clip_fc, ["SHAPE@"]) as clip_cursor:
                    for clip_row in clip_cursor:
                        with arcpy.da.InsertCursor(output_shapefile, output_fields) as output_cursor:
                            output_cursor.insertRow([clip_row[0]] + list(attributes))
                    
                    # # Clip the original polygon
                    # arcpy.Clip_analysis(polygon, clip_polygon_fc, clip_fc)
                    
                    # # Add to output
                    # with arcpy.da.SearchCursor(clip_fc, ["SHAPE@"]) as clip_cursor:
                    #     for clip_row in clip_cursor:
                    #         with arcpy.da.InsertCursor(output_shapefile, output_fields) as output_cursor:
                    #             output_cursor.insertRow([clip_row[0]] + list(attributes))
            
            except Exception as e:
                print(f"Error processing polygon {oid}: {str(e)}")
                continue
    
    # Clean up temp files
    try:
        for file in os.listdir(temp_dir):
            if str(timestamp) in file:
                try:
                    os.remove(os.path.join(temp_dir, file))
                except:
                    pass
    except:
        pass
    
    print(f"Processing complete. Output saved to {output_shapefile}")
    return output_shapefile

# Example usage
if __name__ == "__main__":
    create_45_degree_crop(
        r"C:\Users\jmusinsky\Documents\Data\NEON Sites\Flight_Boundaries_ArcGIS_Online\D17_SOAP\Shapes\D17_SOAP_R1_P1_P2_P3_v7_Var_Alt_fltlines_3nm_buff_cropped_north.shp",
        r"C:\Users\jmusinsky\Documents\Data\NEON Sites\Flight_Boundaries_ArcGIS_Online\D17_SOAP\Shapes\D17_SOAP_R1_P1_P2_P3_v7_Var_Alt_fltlines_3nm_buff_cropped_north_45_A.shp"
    )