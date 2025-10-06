import geopandas as gpd
import pandas as pd
import sys
import os

# ============================================================================
# USER DEFINED PARAMETERS
# ============================================================================

# Path to the CSV file containing SBET GPS times information
FLIGHT_LINES_CSV = r"C:\Users\jmusinsky\Documents\Data\TopoFlight\Conversions\pointsToElev\BART_2025_GPS_start_end_times.csv"

# Input shapefile path
INPUT_PATH = r"C:\Users\jmusinsky\Documents\Data\NEON Sites\Flight_Boundaries_ArcGIS_Online\D01_BART\Shapes"

# Output path for CSV files
OUTPUT_PATH = r"C:\Users\jmusinsky\Documents\Data\TopoFlight\Conversions\pointsToElev\data_out"

# ============================================================================
# FUNCTIONS
# ============================================================================

def extract_flight_line(line_number, north_gps_time, south_gps_time, shapefile_name):
    """
    Extract flight line data from shapefile based on line number and GPS_TIME range.
    Automatically determines correct time ordering based on which time is earlier.
    
    Parameters:
    -----------
    line_number : int
        Line number to assign to extracted records
    north_gps_time : float
        GPS_TIME at the north end of the line
    south_gps_time : float
        GPS_TIME at the south end of the line
    shapefile_name : str
        Name of the shapefile (without .shp extension)
    """
    
    # Add .shp extension if not present
    if not shapefile_name.endswith('.shp'):
        shapefile_name = shapefile_name + '.shp'
    
    # Construct full input path
    input_shapefile = os.path.join(INPUT_PATH, shapefile_name)
    
    # Check if input file exists
    if not os.path.exists(input_shapefile):
        print(f"Error: Input shapefile not found: {input_shapefile}")
        return False
    
    # Determine flight direction and set start/end times accordingly
    if north_gps_time < south_gps_time:
        start_gps_time = north_gps_time
        end_gps_time = south_gps_time
        direction = "North to South"
    else:
        start_gps_time = south_gps_time
        end_gps_time = north_gps_time
        direction = "South to North"
    
    # Read shapefile
    print(f"  Reading shapefile: {shapefile_name}")
    print(f"  Flight direction: {direction}")
    gdf = gpd.read_file(input_shapefile)
    
    # Filter by GPS_TIME range (exclusive boundaries)
    print(f"  Filtering records where GPS_TIME > {start_gps_time} and < {end_gps_time}")
    filtered_gdf = gdf[(gdf['GPS_TIME'] > start_gps_time) & 
                       (gdf['GPS_TIME'] < end_gps_time)].copy()
    
    if len(filtered_gdf) == 0:
        print(f"  Warning: No records found in GPS_TIME range {start_gps_time} to {end_gps_time}")
        return False
    
    # Add line number as first column
    line_col_name = f"line_{line_number}"
    filtered_gdf.insert(0, 'LINE', line_col_name)
    
    # Drop geometry column for CSV export
    df = pd.DataFrame(filtered_gdf.drop(columns='geometry'))
    
    # Create output filename (remove .shp extension from shapefile_name for output)
    base_name = os.path.splitext(shapefile_name)[0]
    output_filename = f"{base_name}_line{line_number}.csv"
    output_filepath = os.path.join(OUTPUT_PATH, output_filename)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Export to CSV
    print(f"  Exporting {len(df)} records to: {output_filename}")
    df.to_csv(output_filepath, index=False)
    
    print(f"  Successfully extracted line {line_number} with {len(df)} records")
    print(f"  GPS_TIME range: {df['GPS_TIME'].min():.1f} to {df['GPS_TIME'].max():.1f}")
    
    return True

def process_flight_lines_from_csv(csv_filepath):
    """
    Read flight line information from CSV and process each line.
    
    Parameters:
    -----------
    csv_filepath : str
        Path to CSV file containing Line, North, South, and Shapefile columns
    """
    
    # Check if CSV file exists
    if not os.path.exists(csv_filepath):
        print(f"Error: Flight lines CSV not found: {csv_filepath}")
        return
    
    # Read the CSV file
    print(f"Reading flight lines from: {csv_filepath}")
    df = pd.read_csv(csv_filepath)
    
    # Validate required columns
    required_columns = ['Line', 'North', 'South', 'Shapefile']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns in CSV: {missing_columns}")
        return
    
    print(f"Found {len(df)} flight lines to process\n")
    
    # Process each flight line
    success_count = 0
    fail_count = 0
    
    for idx, row in df.iterrows():
        line_number = int(row['Line'])
        north_time = float(row['North'])
        south_time = float(row['South'])
        shapefile = str(row['Shapefile']).strip()
        
        print(f"Processing Line {line_number}:")
        
        # Extract the flight line (function handles direction automatically)
        success = extract_flight_line(line_number, north_time, south_time, shapefile)
        
        if success:
            success_count += 1
        else:
            fail_count += 1
        
        print()  # Blank line between processing outputs
    
    # Summary
    print("=" * 70)
    print("Processing Summary")
    print("=" * 70)
    print(f"Total lines processed: {len(df)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"\nOutput files saved to: {OUTPUT_PATH}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        # Process all flight lines from CSV
        process_flight_lines_from_csv(FLIGHT_LINES_CSV)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)