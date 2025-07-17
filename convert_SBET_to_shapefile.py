#!/usr/bin/env python3
"""
SBET to Shapefile Converter (ArcGIS Compatible)
Converts Smoothed Best Estimate of Trajectory (SBET) binary files to ArcGIS-compatible shapefiles.
Uses only basic libraries to avoid dependency conflicts in ArcGIS Pro environments.

Example of SBET path on GCS:
    neon-aop-daily/2024/Daily/2024080115_P3C1/L1/GPSIMU

"""

import numpy as np
import pandas as pd
import os
import struct
import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION - Edit these paths and filenames as needed
# =============================================================================

# Input file path and name
INPUT_FILE = r"C:\Users\jmusinsky\Documents\Data\NEON Sites\Flight_Boundaries_ArcGIS_Online\D10_RMNP\SBET\sbet_2024080714.out"

# Output directory and filename (without extension)
OUTPUT_DIR = r"C:\Users\jmusinsky\Documents\Data\NEON Sites\Flight_Boundaries_ArcGIS_Online\D10_RMNP\Shapes"
OUTPUT_FILENAME = "flight_trajectory_RMNP_2024080714_subset"

# Optional: Coordinate Reference System (WGS84 is default)
# Change this if you need a different coordinate system
COORDINATE_SYSTEM = "WGS84"

# Point Decimation Setting
# OPTION 1: Point-based decimation (simple, fast)
# Set to 1 to keep all points, or higher numbers to reduce point density
# For example: POINT_DECIMATION = 10 keeps every 10th point
POINT_DECIMATION = 50  # Keep every 50th point

# OPTION 2: Time-based decimation (more precise, based on GPS time)
# Set TIME_BASED_DECIMATION = True to use this instead of point-based
# This ensures consistent time intervals between points regardless of data rate variations
TIME_BASED_DECIMATION = False  # Set to True to use time-based instead of point-based

# =============================================================================
# FUNCTIONS
# =============================================================================

def readFlatBinaryFile(file_path, dtype=np.double, numCols=17):
    """
    Read SBET binary file and return structured numpy array.
    
    Parameters:
    -----------
    file_path : str
        Path to the SBET binary file
    dtype : numpy dtype
        Data type for reading (default: np.double)
    numCols : int
        Number of columns in the data (default: 17)
    
    Returns:
    --------
    numpy.ndarray
        Reshaped array with trajectory data
    """
    try:
        with open(file_path, 'rb') as file:
            # Read the entire file into a numpy array
            data = np.fromfile(file, dtype=dtype)
        
        # Check if data size is compatible with expected columns
        if data.size % numCols != 0:
            print(f"Warning: Data size ({data.size}) is not evenly divisible by {numCols} columns")
            # Trim data to fit complete rows
            data = data[:-(data.size % numCols)]
        
        # Determine the number of rows
        numRows = data.size // numCols
        print(f"Read {numRows} trajectory points from {file_path}")
        
        # Reshape the array to have the correct number of columns
        data = data.reshape((numRows, numCols))
        return data
        
    except FileNotFoundError:
        raise FileNotFoundError(f"SBET file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading SBET file: {str(e)}")

def sbet_to_dataframe(sbet_data):
    """
    Convert SBET numpy array to pandas DataFrame with proper column names.
    
    Parameters:
    -----------
    sbet_data : numpy.ndarray
        Raw SBET data array
    
    Returns:
    --------
    pandas.DataFrame
        Structured DataFrame with named columns
    """
    # Create DataFrame with descriptive column names
    df = pd.DataFrame(sbet_data, columns=[
        'gps_time',           # GPS time of week
        'latitude_rad',       # latitude in radians
        'longitude_rad',      # longitude in radians  
        'elevation',          # altitude
        'x_velocity',         # velocity in x direction
        'y_velocity',         # velocity in y direction
        'z_velocity',         # velocity in z direction
        'roll',              # roll angle
        'pitch',             # pitch angle
        'heading',           # heading angle
        'wander',            # wander
        'x_force',           # force in x direction
        'y_force',           # force in y direction
        'z_force',           # force in z direction
        'x_angular_rate',    # angular rate in x direction
        'y_angular_rate',    # angular rate in y direction
        'z_angular_rate'     # angular rate in z direction
    ])
    
    # Convert radians to degrees for latitude and longitude
    df['latitude'] = np.degrees(df['latitude_rad'])
    df['longitude'] = np.degrees(df['longitude_rad'])
    
    # Add additional useful columns
    df['speed_3d'] = np.sqrt(df['x_velocity']**2 + df['y_velocity']**2 + df['z_velocity']**2)
    df['speed_2d'] = np.sqrt(df['x_velocity']**2 + df['y_velocity']**2)
    
    # Convert angles from radians to degrees
    df['roll_deg'] = np.degrees(df['roll'])
    df['pitch_deg'] = np.degrees(df['pitch'])
    df['heading_deg'] = np.degrees(df['heading'])
    df['wander_deg'] = np.degrees(df['wander'])
    
    return df

def decimate_by_time(df, interval_seconds=1.0):
    """
    Decimate points based on time interval to get approximately one point per specified interval.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Trajectory data with gps_time column
    interval_seconds : float
        Desired time interval between points in seconds
    
    Returns:
    --------
    pandas.DataFrame
        Decimated DataFrame
    """
    if 'gps_time' not in df.columns:
        print("Warning: GPS time not available, falling back to point-based decimation")
        return df.iloc[::10].copy()  # Default to every 10th point
    
    # Sort by GPS time to ensure proper ordering
    df_sorted = df.sort_values('gps_time').copy()
    
    # Find points that are at least interval_seconds apart
    selected_indices = [0]  # Always keep first point
    last_time = df_sorted.iloc[0]['gps_time']
    
    for i in range(1, len(df_sorted)):
        current_time = df_sorted.iloc[i]['gps_time']
        if current_time - last_time >= interval_seconds:
            selected_indices.append(i)
            last_time = current_time
    
    return df_sorted.iloc[selected_indices].copy()

def write_shapefile_components(df, output_path):
    """
    Create shapefile components (.shp, .shx, .dbf, .prj) manually.
    This avoids dependency issues with geopandas/fiona.
    """
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    num_points = len(df)
    
    # Calculate bounding box
    min_x, max_x = df['longitude'].min(), df['longitude'].max()
    min_y, max_y = df['latitude'].min(), df['latitude'].max()
    
    # Write .shp file (main shapefile)
    with open(f"{output_path}.shp", 'wb') as shp_file:
        # Write header
        shp_file.write(struct.pack('>I', 9994))  # File code
        shp_file.write(b'\x00' * 20)  # Unused bytes
        file_length = 50 + num_points * 14  # Header + records
        shp_file.write(struct.pack('>I', file_length))  # File length
        shp_file.write(struct.pack('<I', 1000))  # Version
        shp_file.write(struct.pack('<I', 1))  # Shape type (point)
        shp_file.write(struct.pack('<d', min_x))  # Bounding box
        shp_file.write(struct.pack('<d', min_y))
        shp_file.write(struct.pack('<d', max_x))
        shp_file.write(struct.pack('<d', max_y))
        shp_file.write(struct.pack('<d', 0.0))  # Z range
        shp_file.write(struct.pack('<d', 0.0))
        shp_file.write(struct.pack('<d', 0.0))  # M range
        shp_file.write(struct.pack('<d', 0.0))
        
        # Write point records
        for i, (_, row) in enumerate(df.iterrows()):
            record_length = 10  # Point record length
            shp_file.write(struct.pack('>I', i + 1))  # Record number
            shp_file.write(struct.pack('>I', record_length))  # Content length
            shp_file.write(struct.pack('<I', 1))  # Shape type (point)
            shp_file.write(struct.pack('<d', row['longitude']))  # X
            shp_file.write(struct.pack('<d', row['latitude']))   # Y
    
    # Write .shx file (index)
    with open(f"{output_path}.shx", 'wb') as shx_file:
        # Write header (same as .shp)
        shx_file.write(struct.pack('>I', 9994))
        shx_file.write(b'\x00' * 20)
        index_length = 50 + num_points * 4
        shx_file.write(struct.pack('>I', index_length))
        shx_file.write(struct.pack('<I', 1000))
        shx_file.write(struct.pack('<I', 1))
        shx_file.write(struct.pack('<d', min_x))
        shx_file.write(struct.pack('<d', min_y))
        shx_file.write(struct.pack('<d', max_x))
        shx_file.write(struct.pack('<d', max_y))
        shx_file.write(struct.pack('<d', 0.0))
        shx_file.write(struct.pack('<d', 0.0))
        shx_file.write(struct.pack('<d', 0.0))
        shx_file.write(struct.pack('<d', 0.0))
        
        # Write index records
        offset = 50
        for i in range(num_points):
            shx_file.write(struct.pack('>I', offset))  # Offset
            shx_file.write(struct.pack('>I', 14))      # Content length
            offset += 14
    
    # Write .prj file (projection)
    with open(f"{output_path}.prj", 'w') as prj_file:
        # WGS84 Geographic Coordinate System
        wgs84_wkt = ('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
                    'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
                    'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]')
        prj_file.write(wgs84_wkt)

def write_dbf_file(df, output_path):
    """
    Write DBF file with attribute data.
    """
    # Select columns for output (limit field names to 10 chars for DBF compatibility)
    output_data = df[['gps_time', 'latitude', 'longitude', 'elevation', 
                     'speed_3d', 'speed_2d', 'roll_deg', 'pitch_deg', 
                     'heading_deg', 'x_velocity', 'y_velocity', 'z_velocity']].copy()
    
    # Rename columns to be DBF-friendly (max 10 characters)
    output_data.columns = ['GPS_TIME', 'LATITUDE', 'LONGITUDE', 'ELEVATION',
                          'SPEED_3D', 'SPEED_2D', 'ROLL_DEG', 'PITCH_DEG',
                          'HEAD_DEG', 'X_VEL', 'Y_VEL', 'Z_VEL']
    
    num_records = len(output_data)
    num_fields = len(output_data.columns)
    
    # Calculate record length (1 byte for deletion flag + field lengths)
    record_length = 1 + num_fields * 19  # Using 19 bytes per numeric field
    
    with open(f"{output_path}.dbf", 'wb') as dbf_file:
        # Write header
        dbf_file.write(struct.pack('B', 0x03))  # DBF version
        
        # Current date
        now = datetime.datetime.now()
        dbf_file.write(struct.pack('BBB', now.year - 1900, now.month, now.day))
        
        dbf_file.write(struct.pack('<L', num_records))  # Number of records
        header_length = 32 + num_fields * 32 + 1
        dbf_file.write(struct.pack('<H', header_length))  # Header length
        dbf_file.write(struct.pack('<H', record_length))  # Record length
        dbf_file.write(b'\x00' * 20)  # Reserved bytes
        
        # Write field descriptors
        for col_name in output_data.columns:
            field_name = col_name.ljust(11, '\x00')[:11].encode('ascii')
            dbf_file.write(field_name)  # Field name
            dbf_file.write(b'N')  # Field type (Numeric)
            dbf_file.write(b'\x00' * 4)  # Reserved
            dbf_file.write(struct.pack('B', 19))  # Field length
            dbf_file.write(struct.pack('B', 6))   # Decimal places
            dbf_file.write(b'\x00' * 14)  # Reserved
        
        # Header terminator
        dbf_file.write(b'\x0D')
        
        # Write records
        for _, row in output_data.iterrows():
            dbf_file.write(b' ')  # Deletion flag (space = not deleted)
            for value in row:
                # Format as 19-character string with 6 decimal places
                value_str = f"{float(value):19.6f}".encode('ascii')
                dbf_file.write(value_str)
        
        # End of file marker
        dbf_file.write(b'\x1A')

def create_shapefile(df, output_path):
    """
    Create complete shapefile from trajectory DataFrame.
    """
    try:
        # Create all shapefile components
        write_shapefile_components(df, output_path)
        write_dbf_file(df, output_path)
        
        print(f"Shapefile created successfully: {output_path}.shp")
        print(f"Number of points: {len(df)}")
        
        # Print basic statistics
        print("\nTrajectory Statistics:")
        print(f"Latitude range: {df['latitude'].min():.6f} to {df['latitude'].max():.6f}")
        print(f"Longitude range: {df['longitude'].min():.6f} to {df['longitude'].max():.6f}")
        print(f"Elevation range: {df['elevation'].min():.2f} to {df['elevation'].max():.2f}")
        print(f"Speed range: {df['speed_3d'].min():.2f} to {df['speed_3d'].max():.2f}")
        
        # List created files
        extensions = ['.shp', '.shx', '.dbf', '.prj']
        print(f"\nCreated files:")
        for ext in extensions:
            filepath = f"{output_path}{ext}"
            if os.path.exists(filepath):
                size_kb = os.path.getsize(filepath) / 1024
                print(f"  {filepath} ({size_kb:.1f} KB)")
        
    except Exception as e:
        print(f"Error creating shapefile: {str(e)}")
        raise

def main():
    """
    Main execution function.
    """
    print("SBET to Shapefile Converter (ArcGIS Compatible)")
    print("=" * 50)
    
    # Validate input file
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file does not exist: {INPUT_FILE}")
        return
    
    # Create output path
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    try:
        # Step 1: Read SBET binary file
        print(f"Reading SBET file: {INPUT_FILE}")
        sbet_data = readFlatBinaryFile(INPUT_FILE)
        
        # Step 2: Convert to DataFrame
        print("Converting to structured format...")
        df = sbet_to_dataframe(sbet_data)
        
        # Step 3: Apply point decimation if specified
        if TIME_BASED_DECIMATION:
            original_count = len(df)
            df = decimate_by_time(df, interval_seconds=1.0)  # Change interval_seconds here if needed
            decimated_count = len(df)
            print(f"Applied time-based decimation (1 point per second):")
            print(f"  Original points: {original_count:,}")
            print(f"  Decimated points: {decimated_count:,}")
            print(f"  Reduction: {((original_count - decimated_count) / original_count * 100):.1f}%")
        elif POINT_DECIMATION > 1:
            original_count = len(df)
            df = df.iloc[::POINT_DECIMATION].copy()
            decimated_count = len(df)
            print(f"Applied point decimation (keeping every {POINT_DECIMATION} points):")
            print(f"  Original points: {original_count:,}")
            print(f"  Decimated points: {decimated_count:,}")
            print(f"  Reduction: {((original_count - decimated_count) / original_count * 100):.1f}%")
        
        # Step 4: Create shapefile
        print(f"Creating shapefile: {output_path}.shp")
        create_shapefile(df, output_path)
        
        print("\nConversion completed successfully!")
        print("The shapefile is ready to import into ArcGIS.")
        
        # Optional: Save CSV for inspection
        csv_path = f"{output_path}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nCSV file also created for inspection: {csv_path}")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    main()