import geopandas as gpd
import pandas as pd
import sys
import os

# Hard-coded paths
INPUT_PATH = r"C:\Users\jmusinsky\Documents\Data\NEON Sites\Flight_Boundaries_ArcGIS_Online\z_Assignable_Assets\D13_LBLB\Shapes"
INPUT_FILE = "flight_trajectory_CRBU_P1C1_L1_GPSIMU_sbet_2025061614_subset.shp"
OUTPUT_PATH = r"C:\Users\jmusinsky\Documents\Data\TopoFlight\Conversions\pointsToElev\data_out"

def extract_flight_line(line_number, start_gps_time, end_gps_time):
    """
    Extract flight line data from shapefile based on line number and GPS_TIME range.
    
    Parameters:
    -----------
    line_number : int
        Line number to assign to extracted records
    start_gps_time : float
        Start GPS_TIME for extraction
    end_gps_time : float
        End GPS_TIME for extraction
    """
    
    # Construct full input path
    input_shapefile = os.path.join(INPUT_PATH, INPUT_FILE)
    
    # Check if input file exists
    if not os.path.exists(input_shapefile):
        print(f"Error: Input shapefile not found: {input_shapefile}")
        return
    
    # Read shapefile
    print(f"Reading shapefile: {input_shapefile}")
    gdf = gpd.read_file(input_shapefile)
    
    # Filter by GPS_TIME range (exclusive boundaries)
    print(f"Filtering records where GPS_TIME > {start_gps_time} and < {end_gps_time}")
    filtered_gdf = gdf[(gdf['GPS_TIME'] > start_gps_time) & 
                       (gdf['GPS_TIME'] < end_gps_time)].copy()
    
    if len(filtered_gdf) == 0:
        print(f"Warning: No records found in GPS_TIME range {start_gps_time} to {end_gps_time}")
        return
    
    # Add line number as first column
    line_col_name = f"line_{line_number}"
    filtered_gdf.insert(0, 'LINE', line_col_name)
    
    # Drop geometry column for CSV export
    df = pd.DataFrame(filtered_gdf.drop(columns='geometry'))
    
    # Create output filename
    base_name = os.path.splitext(INPUT_FILE)[0]
    output_filename = f"{base_name}_line{line_number}.csv"
    output_filepath = os.path.join(OUTPUT_PATH, output_filename)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Export to CSV
    print(f"Exporting {len(df)} records to: {output_filepath}")
    df.to_csv(output_filepath, index=False)
    
    print(f"Successfully extracted line {line_number} with {len(df)} records")
    print(f"GPS_TIME range: {df['GPS_TIME'].min()} to {df['GPS_TIME'].max()}")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) != 4:
        print("Usage: python script.py <line_number> <start_gps_time> <end_gps_time>")
        print("Example: python script.py 31 236974.6735 236982.1753")
        sys.exit(1)
    
    try:
        line_num = int(sys.argv[1])
        start_time = float(sys.argv[2])
        end_time = float(sys.argv[3])
        
        # Validate inputs
        if start_time >= end_time:
            print("Error: start_gps_time must be less than end_gps_time")
            sys.exit(1)
        
        # Extract flight line
        extract_flight_line(line_num, start_time, end_time)
        
    except ValueError as e:
        print(f"Error: Invalid input values. Line number must be an integer, GPS times must be numeric.")
        print(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)