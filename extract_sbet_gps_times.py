import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import os
from pathlib import Path

# =====================================================================
# USER-DEFINED PARAMETERS
# =====================================================================

# Site name
SITE = "UPTA"

# Input files
SBET_SHAPEFILE = r"C:\Users\jmusinsky\Documents\Data\NEON Sites\Flight_Boundaries_ArcGIS_Online\z_Assignable_Assets\D13_LBLB\Shapes\flight_trajectory_UPTA_Merge.shp"
FLIGHTLINE_SHAPEFILE = r"C:\Users\jmusinsky\Documents\Data\NEON Sites\Flight_Boundaries_ArcGIS_Online\z_Assignable_Assets\D13_LBLB\Shapes\D13_UPTA_S2_P3_625m_max_v7_VQ780_fltlines.shp"

# Output file
OUTPUT_CSV = r"C:\Users\jmusinsky\Documents\Data\TopoFlight\Conversions\pointsToElev\UPTA_GPS_start_end_times.csv"

# Coordinate system (EPSG code)
CRS = "EPSG:32613"  # UTM Zone 13N

# Analysis parameters
BUFFER_DISTANCE = 150  # meters
HEADING_TOLERANCE = 15  # degrees from north (0°) or south (180°)
TIME_GAP_THRESHOLD = 5  # seconds - gap indicating break in data collection
PROCESS_SPECIFIC_LINE = None  # Set to a line number (int) to process only that line, or None for all lines

# =====================================================================
# FUNCTIONS
# =====================================================================

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate bearing between two points in degrees (0-360)
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    
    bearing = np.arctan2(x, y)
    bearing = np.degrees(bearing)
    bearing = (bearing + 360) % 360
    
    return bearing

def is_north_south_bearing(bearing, tolerance):
    """
    Check if bearing is within tolerance of north (0°) or south (180°)
    """
    # Normalize bearing to 0-360
    bearing = bearing % 360
    
    # Check if close to north (0° or 360°)
    close_to_north = (bearing <= tolerance) or (bearing >= (360 - tolerance))
    
    # Check if close to south (180°)
    close_to_south = abs(bearing - 180) <= tolerance
    
    return close_to_north or close_to_south

def find_temporal_segments(gdf, time_gap_threshold):
    """
    Group points into temporally contiguous segments based on GPS time gaps
    Returns a list of GeoDataFrame segments
    """
    if len(gdf) == 0:
        return []
    
    # Sort by GPS time
    gdf = gdf.sort_values('GPS_TIME').reset_index(drop=True)
    
    # Calculate time differences between consecutive points
    time_diffs = gdf['GPS_TIME'].diff()
    
    # Identify breaks (gaps larger than threshold)
    breaks = time_diffs > time_gap_threshold
    
    # Create segment IDs
    segment_ids = breaks.cumsum()
    gdf['segment_id'] = segment_ids
    
    # Split into separate segments
    segments = []
    for seg_id in gdf['segment_id'].unique():
        segment = gdf[gdf['segment_id'] == seg_id].copy()
        segments.append(segment)
    
    return segments

def filter_north_south_segments(segment, heading_tolerance):
    """
    Filter segment to only include points moving in north-south direction
    Returns filtered GeoDataFrame
    """
    if len(segment) < 2:
        return segment
    
    # Calculate bearings between consecutive points
    bearings = []
    valid_indices = [0]  # Always include first point
    
    for i in range(len(segment) - 1):
        lat1 = segment.iloc[i]['LATITUDE']
        lon1 = segment.iloc[i]['LONGITUDE']
        lat2 = segment.iloc[i + 1]['LATITUDE']
        lon2 = segment.iloc[i + 1]['LONGITUDE']
        
        bearing = calculate_bearing(lat1, lon1, lat2, lon2)
        bearings.append(bearing)
        
        if is_north_south_bearing(bearing, heading_tolerance):
            valid_indices.append(i + 1)
    
    # Return filtered segment
    filtered = segment.iloc[valid_indices].copy()
    
    return filtered

def process_flight_line(line_num, flight_line_geom, sbet_gdf, buffer_dist, heading_tol, time_gap_thresh, sbet_filename):
    """
    Process a single flight line to extract GPS times
    Returns dict with results or None if no valid data found
    """
    print(f"  Processing line {line_num}...")
    
    # Buffer the flight line
    buffered_line = flight_line_geom.buffer(buffer_dist)
    
    # Find SBET points within buffer
    candidates = sbet_gdf[sbet_gdf.intersects(buffered_line)].copy()
    
    if len(candidates) == 0:
        print(f"    No SBET points found within {buffer_dist}m buffer")
        return None
    
    print(f"    Found {len(candidates)} candidate points within buffer")
    
    # Split into temporal segments
    segments = find_temporal_segments(candidates, time_gap_thresh)
    print(f"    Identified {len(segments)} temporal segment(s)")
    
    # Filter each segment for north-south movement
    filtered_segments = []
    for i, seg in enumerate(segments):
        filtered = filter_north_south_segments(seg, heading_tol)
        if len(filtered) >= 2:  # Need at least 2 points
            filtered_segments.append(filtered)
            print(f"      Segment {i+1}: {len(filtered)} points after N-S filtering")
    
    if len(filtered_segments) == 0:
        print(f"    No valid north-south segments found")
        return None
    
    # Select the longest segment
    longest_segment = max(filtered_segments, key=len)
    print(f"    Selected longest segment with {len(longest_segment)} points")
    
    # Find northernmost and southernmost points
    north_point = longest_segment.loc[longest_segment['LATITUDE'].idxmax()]
    south_point = longest_segment.loc[longest_segment['LATITUDE'].idxmin()]
    
    north_gps = int(north_point['GPS_TIME'])
    south_gps = int(south_point['GPS_TIME'])
    
    print(f"    North GPS time: {north_gps}, South GPS time: {south_gps}")
    
    return {
        'Line': line_num,
        'North': north_gps,
        'South': south_gps,
        'Shapefile': sbet_filename
    }

# =====================================================================
# MAIN PROCESSING
# =====================================================================

def main():
    print("=" * 70)
    print("SBET Flight Line GPS Time Extractor")
    print("=" * 70)
    print(f"\nSite: {SITE}")
    print(f"Buffer distance: {BUFFER_DISTANCE}m")
    print(f"Heading tolerance: ±{HEADING_TOLERANCE}°")
    print(f"Time gap threshold: {TIME_GAP_THRESHOLD}s")
    
    # Load shapefiles
    print("\nLoading data...")
    print(f"  SBET shapefile: {SBET_SHAPEFILE}")
    sbet_gdf = gpd.read_file(SBET_SHAPEFILE)
    
    print(f"  Flight line shapefile: {FLIGHTLINE_SHAPEFILE}")
    flightlines_gdf = gpd.read_file(FLIGHTLINE_SHAPEFILE)
    
    # Ensure correct CRS
    if sbet_gdf.crs != CRS:
        print(f"  Reprojecting SBET to {CRS}")
        sbet_gdf = sbet_gdf.to_crs(CRS)
    
    if flightlines_gdf.crs != CRS:
        print(f"  Reprojecting flight lines to {CRS}")
        flightlines_gdf = flightlines_gdf.to_crs(CRS)
    
    print(f"\nLoaded {len(sbet_gdf)} SBET points")
    print(f"Loaded {len(flightlines_gdf)} flight lines")
    
    # Get SBET filename (without path and extension)
    sbet_filename = Path(SBET_SHAPEFILE).stem
    
    # Process flight lines
    results = []
    duplicates = []
    
    # Filter to specific line if requested
    if PROCESS_SPECIFIC_LINE is not None:
        flightlines_to_process = flightlines_gdf[flightlines_gdf['FLNUM'] == PROCESS_SPECIFIC_LINE]
        print(f"\nProcessing only line {PROCESS_SPECIFIC_LINE}")
    else:
        flightlines_to_process = flightlines_gdf
        print(f"\nProcessing all flight lines...")
    
    for idx, row in flightlines_to_process.iterrows():
        line_num = row['FLNUM']
        line_geom = row.geometry
        
        result = process_flight_line(
            line_num, 
            line_geom, 
            sbet_gdf, 
            BUFFER_DISTANCE, 
            HEADING_TOLERANCE, 
            TIME_GAP_THRESHOLD,
            sbet_filename
        )
        
        if result is not None:
            # Check for duplicates
            existing = [r for r in results if r['Line'] == line_num]
            if existing:
                duplicates.append(line_num)
                print(f"    WARNING: Duplicate data found for line {line_num}")
                print(f"             Keeping first occurrence")
            else:
                results.append(result)
    
    # Create output DataFrame
    if len(results) > 0:
        output_df = pd.DataFrame(results)
        output_df = output_df.sort_values('Line').reset_index(drop=True)
        
        # Save to CSV
        output_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n{'=' * 70}")
        print(f"SUCCESS! Processed {len(results)} flight lines")
        print(f"Output saved to: {OUTPUT_CSV}")
        
        if duplicates:
            print(f"\nWARNING: Duplicate data found for lines: {duplicates}")
        
        # Display summary
        print(f"\nSummary:")
        print(output_df.to_string(index=False))
    else:
        print(f"\n{'=' * 70}")
        print("No valid data found for any flight lines")
    
    print("=" * 70)

if __name__ == "__main__":
    main()