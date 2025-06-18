import arcpy
import os
import math
import time

def create_45_degree_crop_south(input_shapefile, output_shapefile):
    """
    Crops south-facing polygons creating a triangular area with 45-degree angles from the center of the top
    The apex of the triangle is at the top, with the base at the bottom of the polygon
    
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
                min_y = extent.YMin
                
                # 1. Identify the top points (should be at the top for south-facing polygons)
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
                
                # Find the maximum Y value
                stats_table = os.path.join(temp_dir, f"stats_{timestamp}_{oid}.dbf")
                arcpy.Statistics_analysis(vertices_fc, stats_table, [["POINT_Y", "MAX"]])
                
                # Find the correct field name for maximum Y
                max_field = None
                for field in arcpy.ListFields(stats_table):
                    if 'MAX' in field.name.upper() and 'POINT_Y' in field.name.upper():
                        max_field = field.name
                        break
                
                # If we didn't find a field with both MAX and POINT_Y, look for just MAX
                if not max_field:
                    for field in arcpy.ListFields(stats_table):
                        if 'MAX' in field.name.upper():
                            max_field = field.name
                            break
                
                if not max_field:
                    raise ValueError(f"Could not find MAX field in statistics output for polygon {oid}")
                
                max_y = 0
                with arcpy.da.SearchCursor(stats_table, [max_field]) as stats_cursor:
                    for stats_row in stats_cursor:
                        max_y = stats_row[0]
                        break
                
                # Select points at or near the maximum Y value
                arcpy.MakeFeatureLayer_management(vertices_fc, "vertices_lyr")
                arcpy.SelectLayerByAttribute_management("vertices_lyr", "NEW_SELECTION", 
                                                     f"POINT_Y >= {max_y - 0.1}")
                
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
                
                # Calculate the center point at the top
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
                    # Create a triangle with 45-degree angles from center top
                    # The point (apex) of the triangle is at the top
                    clip_array = arcpy.Array([
                        # Top point (apex)
                        arcpy.Point(center_x, max_y),
                        # Bottom-left corner
                        arcpy.Point(center_x - horizontal_offset, min_y),
                        # Bottom-right corner
                        arcpy.Point(center_x + horizontal_offset, min_y),
                        # Close the polygon
                        arcpy.Point(center_x, max_y)
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
    create_45_degree_crop_south(
        r"C:\Users\jmusinsky\Documents\Data\NEON Sites\Flight_Boundaries_ArcGIS_Online\z_Assignable_Assets\D13_LBLB\Shapes\D13_UPTA_S2_P3_625m_max_v6_VQ780_fltlines_3nm_buff_cropped_south.shp",
        r"C:\Users\jmusinsky\Documents\Data\NEON Sites\Flight_Boundaries_ArcGIS_Online\z_Assignable_Assets\D13_LBLB\Shapes\D13_UPTA_3nm_buff_south_45_C.shp"
    )