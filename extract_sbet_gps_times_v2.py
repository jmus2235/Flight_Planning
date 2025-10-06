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
SITE = "BART"

# Input files
SBET_SHAPEFILE = r"C:\Users\jmusinsky\Documents\Data\NEON Sites\Flight_Boundaries_ArcGIS_Online\D01_BART\Shapes\flight_trajectory_BART_2025_Daily_2025082113_P2C1_L1_GPSIMU_sbet_2025082113_subset.shp"
FLIGHTLINE_SHAPEFILE = r"C:\Users\jmusinsky\Documents\Data\NEON Sites\Flight_Boundaries_ArcGIS_Online\D01_BART\Shapes\D1_BART_R1_P1_v1_fltlines_utm19n.shp"

# Output file
OUTPUT_CSV = r"C:\Users\jmusinsky\Documents\Data\TopoFlight\Conversions\pointsToElev\BART_2025_GPS_start_end_times.csv"

# UTM zone
UTM_ZONE = "19N"

# Coordinate system (EPSG code)
# CRS = "EPSG:32619"  # UTM Zone 19N
CRS = f"326{UTM_ZONE[:-1]:0>2}" if UTM_ZONE[-1] == 'N' else f"327{UTM_ZONE[:-1]:0>2}"

# Analysis parameters
BUFFER_DISTANCE = 200  # meters
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
    
    # Get the start and end points of the flight line
    # For LineString, first coord is start, last coord is end
    line_coords = list(flight_line_geom.coords)
    line_start = Point(line_coords[0])
    line_end = Point(line_coords[-1])
    
    # Determine which is north and which is south based on latitude
    if line_start.y > line_end.y:
        north_endpoint = line_start
        south_endpoint = line_end
    else:
        north_endpoint = line_end
        south_endpoint = line_start
    
    print(f"    North endpoint: ({north_endpoint.x:.2f}, {north_endpoint.y:.2f})")
    print(f"    South endpoint: ({south_endpoint.x:.2f}, {south_endpoint.y:.2f})")
    
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
    
    # Find closest points to the north and south endpoints
    # Calculate distances from each SBET point to the endpoints
    longest_segment['dist_to_north'] = longest_segment.geometry.distance(north_endpoint)
    longest_segment['dist_to_south'] = longest_segment.geometry.distance(south_endpoint)
    
    # Find the closest points
    north_point = longest_segment.loc[longest_segment['dist_to_north'].idxmin()]
    south_point = longest_segment.loc[longest_segment['dist_to_south'].idxmin()]
    
    north_gps = int(north_point['GPS_TIME'])
    south_gps = int(south_point['GPS_TIME'])
    
    north_dist = north_point['dist_to_north']
    south_dist = south_point['dist_to_south']
    
    print(f"    North GPS time: {north_gps} (distance to endpoint: {north_dist:.2f}m)")
    print(f"    South GPS time: {south_gps} (distance to endpoint: {south_dist:.2f}m)")
    
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