# -*- coding: utf-8 -*-
"""
Created on Wed May  7 17:51:17 2025

@author: jmusinsky
"""
# Function to calculate aircraft altitude at each point along# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variable Altitude Flight Planner for U-shaped Valleys
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import math
from pathlib import Path

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_distances_along_line(df):
    """Calculate cumulative distances along the flight line"""
    distances = [0]
    for i in range(1, len(df)):
        dist = calculate_distance(
            df.iloc[i-1]['x'], df.iloc[i-1]['y'],
            df.iloc[i]['x'], df.iloc[i]['y']
        )
        distances.append(distances[-1] + dist)
    return distances

def meters_to_feet(meters):
    """Convert meters to feet"""
    return meters * 3.28084

def feet_to_meters(feet):
    """Convert feet to meters"""
    return feet / 3.28084

def calculate_flight_altitude(ground_elevation, agl_height):
    """Calculate flight altitude in feet, rounded up to nearest 100 ft"""
    altitude_meters = ground_elevation + agl_height
    altitude_feet = meters_to_feet(altitude_meters)
    return math.ceil(altitude_feet / 100) * 100

def find_nearest_index(array, value):
    """Find index of nearest value in array"""
    return (np.abs(np.array(array) - value)).argmin()

def process_flight_line(line_id, flight_lines_df, turning_altitudes_df, 
                        initial_dist, min_agl, buffer_height, nom_flight_height, 
                        climb_rate, aircraft_speed, output_dir):
    """
    Process a single flight line to create variable altitude flight plan
    
    Parameters:
    -----------
    line_id : int or str
        ID of the flight line to process
    flight_lines_df : pandas.DataFrame
        DataFrame containing flight line elevation profiles
    turning_altitudes_df : pandas.DataFrame
        DataFrame containing turning altitudes for each flight line
    initial_dist : float
        Distance from start of flight line to first inflection point (m)
    min_agl : float
        Minimum height above ground level for eye safety (m)
    buffer_height : float
        Additional buffer height (m)
    nom_flight_height : float
        Nominal flight height AGL (m)
    climb_rate : float
        Climb/descent rate in ft/min
    aircraft_speed : float
        Aircraft speed in knots
    output_dir : str
        Directory to save output files
    
    Returns:
    --------
    tuple
        (DataFrame containing inflection points, DataFrame with flight track data)
    """
    # Filter data for the specific flight line
    line_data = flight_lines_df[flight_lines_df['line_id'] == line_id].copy()
    
    if len(line_data) == 0:
        print(f"No data found for flight line {line_id}")
        return None, None
    
    # Extract turning altitudes for this flight line
    turning_data = turning_altitudes_df[turning_altitudes_df['line_id'] == line_id]
    if len(turning_data) == 0:
        print(f"No turning altitude data found for flight line {line_id}")
        return None, None
    
    # Get start and end altitudes (in feet)
    start_altitude_ft = turning_data.iloc[0]['start_altitude_ft']
    end_altitude_ft = turning_data.iloc[0]['end_altitude_ft']
    
    # Sort data from north to south (assuming y decreases from north to south)
    line_data = line_data.sort_values(by='y', ascending=False).reset_index(drop=True)
    
    # Determine flight direction
    is_north_to_south = line_data.iloc[0]['y'] > line_data.iloc[-1]['y']
    flight_direction = "N to S" if is_north_to_south else "S to N"
    
    # Calculate distances along the flight line
    distances = calculate_distances_along_line(line_data)
    total_distance = distances[-1]
    line_data['distance'] = distances
    
    # Calculate distances from north and south
    line_data['distance_from_north'] = line_data['distance']
    line_data['distance_from_south'] = total_distance - line_data['distance']
    
    # Find the index of the first inflection point
    first_inflection_idx = find_nearest_index(distances, initial_dist)
    
    # Safety height (min_agl + buffer)
    safety_height = min_agl + buffer_height
    
    # Convert climb rate from ft/min to m/s for proper time calculations
    aircraft_speed_mps = aircraft_speed * 0.514444  # knots to m/s
    climb_rate_mps = (climb_rate / 60) * 0.3048  # ft/min to m/s
    
    # Function to calculate aircraft altitude at each point along the flight line
    def calculate_flight_path(second_inflection_idx):
        # Initialize the flight path array
        flight_path = np.zeros(len(line_data))
        
        # Start altitude (convert from feet to meters)
        start_alt_m = feet_to_meters(start_altitude_ft)
        end_alt_m = feet_to_meters(end_altitude_ft)
        
        # Calculate required distance for descent based on physics
        altitude_diff_ft = start_altitude_ft - end_altitude_ft
        time_required_sec = (altitude_diff_ft / climb_rate) * 60  # time in seconds
        required_distance_m = aircraft_speed_mps * time_required_sec  # distance in meters
        
        # Fixed altitude segments
        # From start to first inflection
        for i in range(first_inflection_idx + 1):
            flight_path[i] = start_alt_m
        
        # From second inflection to end
        for i in range(second_inflection_idx, len(flight_path)):
            flight_path[i] = end_alt_m
        
        # Calculate descent from first inflection to second inflection
        segment_distance = distances[second_inflection_idx] - distances[first_inflection_idx]
        
        # If the segment distance is less than required, log a warning
        if segment_distance < required_distance_m and altitude_diff_ft > 0:
            print(f"WARNING: Line {line_id} - Calculated segment distance ({segment_distance:.1f}m) " +
                  f"is less than required ({required_distance_m:.1f}m) for descent at {climb_rate} ft/min")
            actual_descent_rate = (altitude_diff_ft / (segment_distance / aircraft_speed_mps) * 60)
            print(f"  This results in a descent rate of {actual_descent_rate:.1f} ft/min instead of {climb_rate} ft/min")
        
        # Perform the linear descent regardless
        for i in range(first_inflection_idx + 1, second_inflection_idx):
            progress = (distances[i] - distances[first_inflection_idx]) / segment_distance
            flight_path[i] = start_alt_m - progress * (start_alt_m - end_alt_m)
            
        # Calculate AGL at each point
        agl = flight_path - line_data['elevation'].values
        
        return flight_path, agl
    
    # Calculate required distance for the descent
    altitude_diff_ft = start_altitude_ft - end_altitude_ft
    time_required_sec = (altitude_diff_ft / climb_rate) * 60  # time in seconds
    required_distance_m = aircraft_speed_mps * time_required_sec  # distance in meters
    
    print(f"Flight line {line_id}: Required distance for {altitude_diff_ft} ft descent at {climb_rate} ft/min: {required_distance_m:.1f} meters")
    
    # Find the ideal second inflection point based on required descent distance first
    estimated_second_inflection_dist = distances[first_inflection_idx] + required_distance_m
    
    # Find the closest point that's at least the required distance away
    target_distances = np.array(distances)
    valid_points = np.where(target_distances >= estimated_second_inflection_dist)[0]
    
    if len(valid_points) > 0:
        initial_second_inflection_idx = valid_points[0]
    else:
        # If we can't find a point that's far enough away, use the last point
        initial_second_inflection_idx = len(line_data) - 1
        print(f"WARNING: Line {line_id} - Could not find a point far enough for proper descent rate. " +
              f"Need {required_distance_m:.1f}m but total distance is only {distances[-1]:.1f}m")
    
    # Calculate the vertical descent rate in meters per 50m horizontal distance
    # 50m is the approximate distance between elevation points
    vertical_descent_per_50m = (climb_rate / 60) * 0.3048 * (50 / aircraft_speed_mps)
    print(f"Flight line {line_id}: Vertical descent per 50m horizontal distance: {vertical_descent_per_50m:.2f}m")
    
    # Now, verify this point maintains min AGL requirement with buffer
    # Calculate what the flight path would be with the initial second inflection point
    flight_path, agl = calculate_flight_path(initial_second_inflection_idx)
    
    # Check if any point violates min AGL + buffer
    safety_height = min_agl + buffer_height
    min_agl_found = np.min(agl)
    valid_path_found = True  # Initialize to True, set to False if we can't find a solution
    
    if min_agl_found < safety_height:
        # Find the largest violations
        agl_violation = safety_height - min_agl_found
        print(f"Flight line {line_id}: Minimum AGL found ({min_agl_found:.1f}m) is below safety height ({safety_height:.1f}m)")
        print(f"Flight line {line_id}: AGL violation of {agl_violation:.1f}m detected")
        
        # Calculate how many 50m indices we need to move back to resolve the violation
        indices_to_move_back = math.ceil(agl_violation / vertical_descent_per_50m)
        print(f"Flight line {line_id}: Need to move second inflection point back by {indices_to_move_back} indices (approx. {indices_to_move_back * 50}m)")
        
        # Adjust the second inflection point
        second_inflection_idx = max(first_inflection_idx + 1, initial_second_inflection_idx - indices_to_move_back)
        
        # Recalculate the flight path with the adjusted second inflection point
        flight_path, agl = calculate_flight_path(second_inflection_idx)
        
        # Calculate new minimum AGL to verify we've fixed the issue
        new_min_agl = np.min(agl)
        print(f"Flight line {line_id}: After adjustment, minimum AGL is now {new_min_agl:.1f}m")
        
        # If we still have a violation, we need to adjust the end altitude
        if new_min_agl < safety_height:
            additional_altitude_needed_m = safety_height - new_min_agl
            additional_altitude_needed_ft = additional_altitude_needed_m * 3.28084
            
            # Calculate a new end altitude that would meet the safety requirement
            # Round up to nearest 100 ft
            new_end_altitude_ft = math.ceil((end_altitude_ft + additional_altitude_needed_ft) / 100) * 100
            
            print(f"Flight line {line_id}: Still have AGL violation. Adjusting end altitude from {end_altitude_ft} to {new_end_altitude_ft} ft")
            
            # Use the new end altitude
            end_altitude_ft = new_end_altitude_ft
            
            # Recalculate the flight path with the adjusted end altitude
            flight_path, agl = calculate_flight_path(second_inflection_idx)
            
            # Verify the fix
            final_min_agl = np.min(agl)
            print(f"Flight line {line_id}: Final minimum AGL is {final_min_agl:.1f}m")
            
            if final_min_agl < safety_height:
                print(f"WARNING: Line {line_id} - Could not achieve minimum safety height even with altitude adjustment")
                valid_path_found = False
    else:
        # If no violations, use the initial second inflection point
        second_inflection_idx = initial_second_inflection_idx
        print(f"Flight line {line_id}: No AGL violations detected. Minimum AGL is {min_agl_found:.1f}m")
    
    if not valid_path_found:
        print(f"Could not find a valid flight path for flight line {line_id}")
        return None, None
    
    # Final calculation of flight path
    flight_path, agl = calculate_flight_path(second_inflection_idx)
    
    # Create flight track DataFrame with AGL values
    flight_track_df = line_data.copy()
    flight_track_df['flight_altitude_m'] = flight_path
    flight_track_df['agl_m'] = agl
    flight_track_df['flight_altitude_ft'] = flight_track_df.apply(
        lambda row: calculate_flight_altitude(row['elevation'], row['agl_m']), axis=1
    )
    
    # Create inflection points DataFrame
    inflection_points = []
    
    # First inflection point (start, level mode)
    inflection_points.append({
        'line_id': line_id,
        'point_id': 0,
        'x': line_data.iloc[0]['x'],
        'y': line_data.iloc[0]['y'],
        'elevation': line_data.iloc[0]['elevation'],
        'flight_altitude': calculate_flight_altitude(line_data.iloc[0]['elevation'], agl[0]),
        'mode': 'level',
        'distance_to_next': distances[first_inflection_idx] - distances[0],
        'agl': agl[0],
        'distance_from_south': line_data.iloc[0]['distance_from_south'],
        'distance_from_north': line_data.iloc[0]['distance_from_north']
    })
    
    # Second inflection point (first transition point, descending mode)
    inflection_points.append({
        'line_id': line_id,
        'point_id': 1,
        'x': line_data.iloc[first_inflection_idx]['x'],
        'y': line_data.iloc[first_inflection_idx]['y'],
        'elevation': line_data.iloc[first_inflection_idx]['elevation'],
        'flight_altitude': calculate_flight_altitude(line_data.iloc[first_inflection_idx]['elevation'], agl[first_inflection_idx]),
        'mode': 'descending',
        'distance_to_next': distances[second_inflection_idx] - distances[first_inflection_idx],
        'agl': agl[first_inflection_idx],
        'distance_from_south': line_data.iloc[first_inflection_idx]['distance_from_south'],
        'distance_from_north': line_data.iloc[first_inflection_idx]['distance_from_north']
    })
    
    # Third inflection point (second transition point, climbing mode)
    inflection_points.append({
        'line_id': line_id,
        'point_id': 2,
        'x': line_data.iloc[second_inflection_idx]['x'],
        'y': line_data.iloc[second_inflection_idx]['y'],
        'elevation': line_data.iloc[second_inflection_idx]['elevation'],
        'flight_altitude': calculate_flight_altitude(line_data.iloc[second_inflection_idx]['elevation'], agl[second_inflection_idx]),
        'mode': 'level',
        'distance_to_next': distances[-1] - distances[second_inflection_idx],
        'agl': agl[second_inflection_idx],
        'distance_from_south': line_data.iloc[second_inflection_idx]['distance_from_south'],
        'distance_from_north': line_data.iloc[second_inflection_idx]['distance_from_north']
    })
    
    # Fourth inflection point (end, level mode)
    inflection_points.append({
        'line_id': line_id,
        'point_id': 3,
        'x': line_data.iloc[-1]['x'],
        'y': line_data.iloc[-1]['y'],
        'elevation': line_data.iloc[-1]['elevation'],
        'flight_altitude': calculate_flight_altitude(line_data.iloc[-1]['elevation'], agl[-1]),
        'mode': 'level',
        'distance_to_next': 0,
        'agl': agl[-1],
        'distance_from_south': line_data.iloc[-1]['distance_from_south'],
        'distance_from_north': line_data.iloc[-1]['distance_from_north']
    })
    
    # Create DataFrame
    inflection_df = pd.DataFrame(inflection_points)
    
    # Create plot
    plot_flight_path(line_id, line_data, flight_path, agl, inflection_df, 
                     min_agl, buffer_height, nom_flight_height, output_dir, flight_direction)
    
    return inflection_df, flight_track_df

def plot_flight_path(line_id, line_data, flight_path, agl, inflection_df, 
                     min_agl, buffer_height, nom_flight_height, output_dir, flight_direction, aircraft_speed=97.0):
    """
    Create a plot showing the flight path, terrain, and safety limits
    
    Parameters:
    -----------
    line_id : int or str
        ID of the flight line
    line_data : pandas.DataFrame
        DataFrame containing the flight line data
    flight_path : numpy.ndarray
        Array containing the flight path altitudes
    agl : numpy.ndarray
        Array containing the heights above ground level
    inflection_df : pandas.DataFrame
        DataFrame containing the inflection points
    min_agl : float
        Minimum height above ground level for eye safety (m)
    buffer_height : float
        Additional buffer height (m)
    nom_flight_height : float
        Nominal flight height AGL (m)
    output_dir : str
        Directory to save the plot
    flight_direction : str
        Direction of the flight ("N to S" or "S to N")
    aircraft_speed : float, optional
        Aircraft speed in knots, defaults to 97.0
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot terrain
    terrain = line_data['elevation'].values
    ax.fill_between(line_data['distance'], 0, terrain, color='tan', alpha=0.7, label='Terrain')
    
    # Plot eye safety limit
    safety_line = terrain + min_agl
    ax.plot(line_data['distance'], safety_line, 'r--', label=f'Eye Safety Limit ({min_agl:.1f}m)')
    
    # Plot safety height
    safety_height = terrain + min_agl + buffer_height
    ax.plot(line_data['distance'], safety_height, 'r-', label=f'Safety Height ({min_agl + buffer_height:.1f}m)')
    
    # Plot nominal height
    nominal_height = terrain + nom_flight_height
    ax.plot(line_data['distance'], nominal_height, 'k:', label=f'Nominal Height ({nom_flight_height:.1f}m)')
    
    # Plot flight path
    ax.plot(line_data['distance'], flight_path, 'b-', linewidth=2, label='Flight Path')
    
    # Plot inflection points
    inflection_x = inflection_df['distance_from_north'].values
    inflection_y = [flight_path[find_nearest_index(line_data['distance'], x)] for x in inflection_x]
    ax.scatter(inflection_x, inflection_y, color='blue', s=80, zorder=5, label='Inflection Points')
    
    # Add annotations for inflection points
    for i, (x, y, agl_val, mode, alt_ft) in enumerate(zip(
            inflection_x, 
            inflection_y, 
            inflection_df['agl'], 
            inflection_df['mode'], 
            inflection_df['flight_altitude'])):
        y_offset = 0.05 * (plt.ylim()[1] - plt.ylim()[0])
        ax.annotate(f"Mode: {mode}\nAGL: {agl_val:.0f}m\nAlt: {alt_ft:.0f}ft", 
                    xy=(x, y), xytext=(x, y + y_offset),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                    ha='center', va='bottom')
    
    # Calculate and annotate actual descent rate
    if len(inflection_df) >= 4:  # Make sure we have enough inflection points
        # Get the first and second inflection points (where descent happens)
        p1 = inflection_df.iloc[1]  # Start of descent
        p2 = inflection_df.iloc[2]  # End of descent
        
        # Calculate altitude change in feet
        alt_change_ft = p1['flight_altitude'] - p2['flight_altitude']
        
        # Calculate distance in meters
        descent_distance_m = p1['distance_to_next']
        
        # Calculate time to cover distance at aircraft speed
        aircraft_speed_mps = aircraft_speed * 0.514444  # knots to m/s
        time_seconds = descent_distance_m / aircraft_speed_mps
        time_minutes = time_seconds / 60
        
        # Calculate actual descent rate
        actual_descent_rate = alt_change_ft / time_minutes if time_minutes > 0 else 0
        
        # Annotate descent information
        descent_info = (
            f"Descent distance: {descent_distance_m:.0f}m\n"
            f"Altitude change: {alt_change_ft:.0f}ft\n"
            f"Time: {time_minutes:.1f}min\n"
            f"Actual descent rate: {actual_descent_rate:.0f}ft/min"
        )
        
        # Position the annotation near the middle of the descent path
        mid_x = (inflection_x[1] + inflection_x[2]) / 2
        mid_y = (inflection_y[1] + inflection_y[2]) / 2
        
        # Add the annotation
        ax.annotate(descent_info,
                    xy=(mid_x, mid_y),
                    xytext=(mid_x, mid_y - 0.15 * (plt.ylim()[1] - plt.ylim()[0])),
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8),
                    ha='center', va='top')
    
    # Set labels and title
    ax.set_xlabel('Distance along flight line (m)')
    ax.set_ylabel('Altitude (m above sea level)')
    ax.set_title(f'Flight Line {line_id} - Variable Altitude Flight Plan')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(loc='lower left')
    
    # Ensure the y-axis starts at 0
    ax.set_ylim(bottom=0)
    
    # Add direction indicator in the bottom right
    # Determine which end of the plot (right side) is north or south
    if flight_direction == "N to S":
        direction_indicator = "S"  # Right side (end) is South
    else:
        direction_indicator = "N"  # Right side (end) is North
    
    # Get axis limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    # Place direction indicator in the bottom right corner
    ax.text(x_max - 0.02 * (x_max - x_min), 
            y_min + 0.05 * (y_max - y_min), 
            direction_indicator, 
            fontsize=14, 
            fontweight='bold', 
            ha='right', 
            va='bottom',
            bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="black"))
    
    # Save plot
    os.makedirs(os.path.join(output_dir, 'png'), exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'png', f'flight_line_{line_id}_v5.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to process flight lines"""
    parser = argparse.ArgumentParser(description='Variable Altitude Flight Planner')
    
    # Required arguments
    parser.add_argument('flight_lines_csv', type=str, help='Name of flight lines CSV file')
    parser.add_argument('turning_altitudes_csv', type=str, help='Name of turning altitudes CSV file')
    
    # Optional arguments with default values
    parser.add_argument('--line_id', type=str, help='Specific flight line ID to process (process all if not specified)')
    parser.add_argument('--initial_dist', type=float, default=5000.0, 
                        help='Distance from start of flight line to first inflection point (m)')
    parser.add_argument('--min_agl', type=float, default=490.0, 
                        help='Minimum height above ground level for eye safety (m)')
    parser.add_argument('--buffer_height', type=float, default=100.0, 
                        help='Additional buffer height (m)')
    parser.add_argument('--nom_flight_height', type=float, default=625.0, 
                        help='Nominal flight height AGL (m)')
    parser.add_argument('--climb_rate', type=float, default=400.0, 
                        help='Climb/descent rate in ft/min')
    parser.add_argument('--aircraft_speed', type=float, default=97.0, 
                        help='Aircraft speed in knots')
    parser.add_argument('--input_dir', type=str, 
                        default="C:\\Users\\jmusinsky\\Documents\\Data\\TopoFlight\\Conversions\\pointsToElev",
                        help='Directory containing input files')
    parser.add_argument('--output_dir', type=str, 
                        default="C:\\Users\\jmusinsky\\Documents\\Data\\TopoFlight\\Conversions\\pointsToElev",
                        help='Directory to save output files')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Construct full file paths
    flight_lines_path = os.path.join(args.input_dir, args.flight_lines_csv)
    turning_altitudes_path = os.path.join(args.input_dir, args.turning_altitudes_csv)
    
    # Check if input files exist
    if not os.path.exists(flight_lines_path):
        print(f"Error: Flight lines file not found at {flight_lines_path}")
        return
    
    if not os.path.exists(turning_altitudes_path):
        print(f"Error: Turning altitudes file not found at {turning_altitudes_path}")
        return
    
    # Load flight lines data
    print(f"Loading flight lines data from {flight_lines_path}")
    flight_lines_df = pd.read_csv(flight_lines_path)
    
    # Load turning altitudes data
    print(f"Loading turning altitudes data from {turning_altitudes_path}")
    turning_altitudes_df = pd.read_csv(turning_altitudes_path)
    
    # Get unique line IDs
    if args.line_id:
        line_ids = [int(args.line_id) if args.line_id.isdigit() else args.line_id]
    else:
        line_ids = flight_lines_df['line_id'].unique()
    
    # Process each flight line
    all_inflection_points = []
    all_flight_tracks = []
    
    for line_id in line_ids:
        print(f"Processing flight line {line_id}...")
        inflection_df, flight_track_df = process_flight_line(
            line_id, flight_lines_df, turning_altitudes_df,
            args.initial_dist, args.min_agl, args.buffer_height,
            args.nom_flight_height, args.climb_rate, args.aircraft_speed,
            args.output_dir
        )
        
        if inflection_df is not None and flight_track_df is not None:
            all_inflection_points.append(inflection_df)
            all_flight_tracks.append(flight_track_df)
    
    # Round up decimal values to the nearest meter for specified fields
    for df in all_inflection_points:
        # For integer columns, ensure they are integers
        df['x'] = np.ceil(df['x']).astype(int)
        df['y'] = np.ceil(df['y']).astype(int)
        df['distance_to_next'] = np.ceil(df['distance_to_next']).astype(int)
        df['agl'] = np.ceil(df['agl']).astype(int)
        df['distance_from_south'] = np.ceil(df['distance_from_south']).astype(int)
        df['distance_from_north'] = np.ceil(df['distance_from_north']).astype(int)
    
    # Combine all inflection points and save to CSV
    if all_inflection_points:
        combined_df = pd.concat(all_inflection_points, ignore_index=True)
        
        # Set the output CSV filename based on whether a specific line_id was provided
        if args.line_id:
            output_csv_filename = f'inflection_points_line_{args.line_id}.csv'
            output_track_filename = f'flight_track_line_{args.line_id}.csv'
        else:
            output_csv_filename = 'inflection_points_all.csv'
            output_track_filename = 'flight_track_all.csv'
            
        output_csv_path = os.path.join(args.output_dir, output_csv_filename)
        combined_df.to_csv(output_csv_path, index=False)
        print(f"Saved inflection points to {output_csv_path}")
        
        # Save flight track data (with AGL values) to CSV
        if all_flight_tracks:
            combined_track_df = pd.concat(all_flight_tracks, ignore_index=True)
            # Select only desired columns for output and apply rounding in a single step
            output_track_df = combined_track_df[['line_id', 'x', 'y', 'elevation']].copy()
            # Round AGL to nearest meter using proper pandas method to avoid SettingWithCopyWarning
            output_track_df.loc[:, 'agl_m'] = np.ceil(combined_track_df['agl_m']).astype(int)
            output_track_path = os.path.join(args.output_dir, output_track_filename)
            output_track_df.to_csv(output_track_path, index=False)
            print(f"Saved flight track with AGL values to {output_track_path}")

if __name__ == "__main__":
    main()