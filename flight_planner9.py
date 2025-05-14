# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 16:10:08 2025

@author: jmusinsky
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 15:26:36 2025

@author: jmusinsky
"""

    #!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Variable Altitude Flight Planner v2

A new implementation of the flight planner that follows a revised logic:
1. Identify which end of the flight line has the highest elevation peak
2. Set starting altitude based on either turning altitude or elevation + nom_flight_height
3. Maintain fixed altitude until the highest elevation point
4. Calculate descent path based on climb rate
5. Ensure minimum AGL requirements are met during descent

@author: Claude
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from math import sqrt, ceil
from typing import List, Dict, Tuple, Optional
import pathlib
import argparse

class FlightPlannerV2:
    """
    A class to calculate variable altitude flight plans with inflection points
    based on terrain elevation data, with a focus on highest elevation peak logic.
    """
    
    def __init__(
        self,
        min_agl: float = 490.0,            # Minimum height above ground level for eye safety (m)
        buffer_height: float = 100.0,      # Additional buffer height (m)
        nom_flight_height: float = 625.0,  # Nominal flight height AGL (m)
        climb_rate: float = 400.0,         # Climb/descent rate in ft/min
        aircraft_speed: float = 97.0,      # Aircraft speed in knots
        input_dir: str = "C:\\Users\\jmusinsky\\Documents\\Data\\TopoFlight\\Conversions\\pointsToElev",
        output_dir: str = "C:\\Users\\jmusinsky\\Documents\\Data\\TopoFlight\\Conversions\\pointsToElev"
    ):
        """
        Initialize the FlightPlanner with configurable parameters.
        """
        self.min_agl = min_agl
        self.buffer_height = buffer_height
        self.safety_height = min_agl + buffer_height  # Minimum safe height with buffer
        self.nom_flight_height = nom_flight_height
        
        # Convert climb rate from ft/min to m/min
        self.climb_rate = climb_rate * 0.3048  # m/min
        
        # Convert aircraft speed from knots to m/s
        self.aircraft_speed = aircraft_speed * 0.514444  # m/s
        
        # Convert climb rate from m/min to m per segment
        # This calculates how much altitude changes over 50m of horizontal distance
        # based on climb rate and aircraft speed
        self.climb_rate_per_segment = (self.climb_rate / 60) / self.aircraft_speed * 50
        
        # Ensure absolute paths for input and output directories
        if input_dir and not os.path.isabs(input_dir):
            self.input_dir = os.path.abspath(input_dir)
        else:
            self.input_dir = input_dir
            
        if output_dir and not os.path.isabs(output_dir):
            self.output_dir = os.path.abspath(output_dir)
        else:
            self.output_dir = output_dir
        
        # Dictionary to store turning area altitudes (ft MSL) for each flight line
        self.turning_altitudes = {}
        
        print(f"Initialized FlightPlanner with the following parameters:")
        print(f"  min_agl: {self.min_agl} m")
        print(f"  buffer_height: {self.buffer_height} m")
        print(f"  safety_height: {self.safety_height} m")
        print(f"  nom_flight_height: {self.nom_flight_height} m")
        print(f"  climb_rate: {climb_rate} ft/min ({self.climb_rate:.2f} m/min)")
        print(f"  aircraft_speed: {aircraft_speed} knots ({self.aircraft_speed:.2f} m/s)")
        print(f"  climb_rate_per_segment: {self.climb_rate_per_segment:.2f} m per 50m segment")
        print(f"  input_dir: {self.input_dir}")
        print(f"  output_dir: {self.output_dir}")
    
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load flight line data from CSV file with fixed path handling.
        
        Args:
            csv_path: Path to the CSV file
        
        Returns:
            DataFrame with flight line data
        """
        # Print the current working directory and input directory for debugging
        print(f"Current working directory: {os.getcwd()}")
        print(f"Input directory: {self.input_dir}")
        print(f"Input file path provided: {csv_path}")
        
        # First try the direct path in case it's absolute
        if os.path.exists(csv_path):
            actual_path = csv_path
            print(f"File found at direct path: {actual_path}")
        # Then try with the input_dir
        elif self.input_dir and os.path.exists(os.path.join(self.input_dir, csv_path)):
            actual_path = os.path.join(self.input_dir, csv_path)
            print(f"File found at input_dir path: {actual_path}")
        else:
            # Try some alternative paths as fallbacks
            alternatives = [
                csv_path,  # Original path
                os.path.join(os.getcwd(), csv_path),  # Current directory + filename
                os.path.join("C:\\Users\\jmusinsky\\Documents\\Data\\TopoFlight\\Conversions\\pointsToElev", csv_path),  # Hardcoded path
                r"C:\Users\jmusinsky\Documents\Data\TopoFlight\Conversions\pointsToElev\\" + csv_path  # Alternative hardcoded format
            ]
            
            found = False
            for alt_path in alternatives:
                if os.path.exists(alt_path):
                    actual_path = alt_path
                    found = True
                    print(f"File found at alternative path: {alt_path}")
                    break
            
            if not found:
                raise FileNotFoundError(f"Could not find input file '{csv_path}'. Make sure it exists in {self.input_dir}")
        
        # Read the CSV file
        print(f"Loading data from {actual_path}")
        df = pd.read_csv(actual_path)
        
        # Keep only the necessary columns
        essential_columns = ['line_id', 'elevation', 'x', 'y']
        for col in essential_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV file")
        
        # Create a clean dataframe with only the columns we need
        clean_df = df[essential_columns].copy()
        
        # Convert to numeric as needed
        for col in essential_columns:
            if clean_df[col].dtype == 'object':
                clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
        
        # Drop any rows with missing data
        clean_df = clean_df.dropna(subset=essential_columns)
        
        # Keep track of the original point IDs
        if 'Id' in df.columns:
            clean_df['point_id'] = df['Id']
        else:
            clean_df['point_id'] = np.arange(len(clean_df))
        
        # Report summary
        unique_lines = clean_df['line_id'].unique()
        print(f"Loaded {len(clean_df)} data points across {len(unique_lines)} flight lines")
        
        return clean_df
    
    def load_turning_altitudes(self, csv_path: str) -> Dict[int, Dict[str, float]]:
        """
        Load turning area altitudes from CSV file with fixed path handling.
        
        Args:
            csv_path: Path to the CSV file containing turning area altitudes
        
        Returns:
            Dictionary with flight line IDs as keys and their turning altitudes as values
        """
        # First try the direct path in case it's absolute
        if os.path.exists(csv_path):
            actual_path = csv_path
            print(f"Turning altitudes file found at direct path: {actual_path}")
        # Then try with the input_dir
        elif self.input_dir and os.path.exists(os.path.join(self.input_dir, csv_path)):
            actual_path = os.path.join(self.input_dir, csv_path)
            print(f"Turning altitudes file found at input_dir path: {actual_path}")
        else:
            # Try some alternative paths as fallbacks
            alternatives = [
                csv_path,  # Original path
                os.path.join(os.getcwd(), csv_path),  # Current directory + filename
                os.path.join("C:\\Users\\jmusinsky\\Documents\\Data\\TopoFlight\\Conversions\\pointsToElev", csv_path),  # Hardcoded path
                r"C:\Users\jmusinsky\Documents\Data\TopoFlight\Conversions\pointsToElev\\" + csv_path  # Alternative hardcoded format
            ]
            
            found = False
            for alt_path in alternatives:
                if os.path.exists(alt_path):
                    actual_path = alt_path
                    found = True
                    print(f"Turning altitudes file found at alternative path: {alt_path}")
                    break
            
            if not found:
                raise FileNotFoundError(f"Could not find turning altitudes file '{csv_path}'. Make sure it exists in {self.input_dir}")
        
        # Read the CSV file
        print(f"Loading turning area altitudes from {actual_path}")
        df = pd.read_csv(actual_path)
        
        # Check required columns are present
        required_columns = ['line_id', 'start_altitude_ft']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in turning altitudes CSV file")
        
        # Optional end_altitude_ft column
        has_end_altitude = 'end_altitude_ft' in df.columns
        
        # Convert to numeric as needed
        for col in required_columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if has_end_altitude and df['end_altitude_ft'].dtype == 'object':
            df['end_altitude_ft'] = pd.to_numeric(df['end_altitude_ft'], errors='coerce')
        
        # Create dictionary of turning altitudes
        turning_altitudes = {}
        for _, row in df.iterrows():
            line_id = int(row['line_id'])
            start_alt_ft = float(row['start_altitude_ft'])
            
            # Convert feet to meters
            start_alt_m = start_alt_ft / 3.28084
            
            # Dictionary to store altitude information
            altitude_data = {
                'start_ft': start_alt_ft,
                'start_m': start_alt_m,
                'has_end_altitude': False  # Default is no end altitude specified
            }
            
            # Only add end altitude if it's explicitly provided
            if has_end_altitude and not pd.isna(row['end_altitude_ft']):
                end_alt_ft = float(row['end_altitude_ft'])
                end_alt_m = end_alt_ft / 3.28084
                altitude_data['end_ft'] = end_alt_ft
                altitude_data['end_m'] = end_alt_m
                altitude_data['has_end_altitude'] = True
            
            turning_altitudes[line_id] = altitude_data
        
        # Store the turning altitudes in the class
        self.turning_altitudes = turning_altitudes
        
        # Print summary of loaded data
        end_alt_count = sum(1 for data in turning_altitudes.values() if data['has_end_altitude'])
        print(f"Loaded turning altitudes for {len(turning_altitudes)} flight lines")
        print(f"  - {len(turning_altitudes)} start altitudes")
        print(f"  - {end_alt_count} end altitudes")
        
        return turning_altitudes
    
    
    def calculate_descent_distance(self, altitude_diff_m: float) -> float:
        """
        Calculate the horizontal distance needed to descend a given altitude difference
        at the specified climb rate.
        
        Args:
            altitude_diff_m: Altitude difference to descend in meters
            
        Returns:
            Horizontal distance needed in meters
        """
        # Convert climb rate from ft/min to m/s
        climb_rate_m_s = (self.climb_rate / 60)  # m/s
        
        # Calculate descent slope - how many meters of altitude change per meter of horizontal distance
        # when descending at the specified climb rate and aircraft speed
        descent_slope = climb_rate_m_s / self.aircraft_speed  # m/m
        
        # Calculate the horizontal distance needed for the descent
        # We need to check if the altitude difference is positive (actual descent)
        if altitude_diff_m <= 0:
            return 0
        
        distance_needed = altitude_diff_m / descent_slope  # meters
        
        print(f"  Descent calculation details:")
        print(f"    Starting climb rate: {self.climb_rate:.2f} m/min ({self.climb_rate / 60:.4f} m/s)")
        print(f"    Aircraft speed: {self.aircraft_speed:.2f} m/s")
        print(f"    Descent slope: {descent_slope:.6f} m/m")
        print(f"    Altitude difference: {altitude_diff_m:.2f} m")
        print(f"    Horizontal distance needed: {distance_needed:.2f} m")
        
        return distance_needed
    
    
    def calculate_flight_plan(self, data: pd.DataFrame, specific_line_id: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate the flight plan with inflection points for all flight lines or a specific one.
        
        Args:
            data: DataFrame with flight line data
            specific_line_id: Optional specific line_id to process. If None, all lines are processed.
        
        Returns:
            DataFrame with inflection points
        """
        # Initialize results dataframe
        results = []
        
        # Get the flight lines to process
        if specific_line_id is not None:
            # Convert specific_line_id to int to ensure proper comparison
            specific_line_id = int(specific_line_id)
            # Check if the specific line_id exists in the data
            if specific_line_id not in data['line_id'].unique():
                print(f"Warning: Flight line {specific_line_id} not found in the dataset.")
                print(f"Available flight lines: {sorted(data['line_id'].unique())}")
                return pd.DataFrame(columns=[
                    'line_id', 'point_id', 'x', 'y', 'elevation', 
                    'cum_distance', 'flight_altitude_msl', 'agl', 
                    'mode', 'is_inflection', 'dist_from_prev'
                ])
            
            # Filter to get only the specified flight line
            line_ids_to_process = [specific_line_id]
            print(f"Processing only flight line {specific_line_id}")
        else:
            # Process all flight lines
            line_ids_to_process = sorted(data['line_id'].unique())
            print(f"Processing all {len(line_ids_to_process)} flight lines")
        
        # Process each flight line separately
        for line_id in line_ids_to_process:
            line_data = data[data['line_id'] == line_id].copy().reset_index(drop=True)
            print(f"Processing flight line {line_id} with {len(line_data)} points")
            
            # Check if we have enough data points for this flight line
            if len(line_data) < 10:
                print(f"  Warning: Flight line {line_id} has fewer than 10 points, skipping")
                continue
                
            line_results = self._process_flight_line(line_data)
            results.append(line_results)
        
        # Combine results from all flight lines
        if results:
            results_df = pd.concat(results, ignore_index=True)
            print(f"Generated flight plan with {len(results_df)} total points")
            
            # Count inflection points
            inflection_count = results_df['is_inflection'].sum()
            print(f"Total inflection points: {inflection_count}")
            
            return results_df
        else:
            print("Warning: No flight lines were processed successfully")
            # Return empty dataframe with expected columns
            return pd.DataFrame(columns=[
                'line_id', 'point_id', 'x', 'y', 'elevation', 
                'cum_distance', 'flight_altitude_msl', 'agl', 
                'mode', 'is_inflection', 'dist_from_prev'
            ])
    
    def _process_flight_line(self, line_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process a single flight line to determine inflection points based on peak and valley logic.
        Adapts to "bowl-like" terrain by identifying valleys and ensuring safe ascent/descent.
        """
        # Make a copy of the input data to avoid modifying the original
        line_data = line_data.copy().reset_index(drop=True)
        line_id = int(line_data['line_id'].iloc[0])

        print(f"\nProcessing flight line {line_id} with {len(line_data)} points")

        # Define the north and south ends of the flight line
        north_idx = line_data['y'].idxmax()
        south_idx = line_data['y'].idxmin()

        # Calculate cumulative distance along the flight line
        line_data['cum_distance'] = 0
        for i in range(1, len(line_data)):
            dx = line_data.loc[i, 'x'] - line_data.loc[i-1, 'x']
            dy = line_data.loc[i, 'y'] - line_data.loc[i-1, 'y']
            dist = sqrt(dx**2 + dy**2)
            line_data.loc[i, 'cum_distance'] = line_data.loc[i-1, 'cum_distance'] + dist

        # Find the highest peak
        highest_peak_idx = line_data['elevation'].idxmax()
        highest_peak_elev = line_data.loc[highest_peak_idx, 'elevation']
        highest_peak_dist = line_data.loc[highest_peak_idx, 'cum_distance']

        print(f"  Highest peak at index {highest_peak_idx}, elevation: {highest_peak_elev:.2f}m")

        # Find the lowest valley after the highest peak
        valley_data = line_data.iloc[highest_peak_idx + 1:]
        if not valley_data.empty:
            valley_idx = valley_data['elevation'].idxmin()
            valley_elev = line_data.loc[valley_idx, 'elevation']
            valley_dist = line_data.loc[valley_idx, 'cum_distance']
            print(f"  Valley found at index {valley_idx}, elevation: {valley_elev:.2f}m")
        else:
            valley_idx = None
            valley_elev = None
            valley_dist = None
            print("  No valley found after the highest peak.")

        # Add columns for flight altitude planning
        line_data['flight_altitude_msl'] = np.nan  # Flight altitude in meters above sea level
        line_data['agl'] = np.nan  # Height above ground level
        line_data['mode'] = None   # Flight mode (level, climbing, descending)
        line_data['is_inflection'] = False  # Flag for inflection points

        # Set starting altitude
        start_altitude_m = highest_peak_elev + self.nom_flight_height
        start_altitude_ft = start_altitude_m * 3.28084
        print(f"  Starting altitude: {start_altitude_m:.2f}m ({start_altitude_ft:.2f}ft)")

        # STEP 1: Level flight to the highest peak
        for i in range(0, highest_peak_idx + 1):
            line_data.loc[i, 'flight_altitude_msl'] = start_altitude_m
            line_data.loc[i, 'agl'] = start_altitude_m - line_data.loc[i, 'elevation']
            line_data.loc[i, 'mode'] = 'level'

        # Mark the highest peak as an inflection point
        line_data.loc[highest_peak_idx, 'is_inflection'] = True
        print(f"  Inflection point at highest peak: index {highest_peak_idx}")

        # STEP 2: Descent to the valley
        if valley_idx is not None:
            descent_altitude_m = max(valley_elev + self.safety_height, self.min_agl + valley_elev)
            for i in range(highest_peak_idx + 1, valley_idx + 1):
                progress = (line_data.loc[i, 'cum_distance'] - highest_peak_dist) / (valley_dist - highest_peak_dist)
                line_data.loc[i, 'flight_altitude_msl'] = start_altitude_m - progress * (start_altitude_m - descent_altitude_m)
                line_data.loc[i, 'agl'] = line_data.loc[i, 'flight_altitude_msl'] - line_data.loc[i, 'elevation']
                line_data.loc[i, 'mode'] = 'descending'

            # Mark the valley as an inflection point
            line_data.loc[valley_idx, 'is_inflection'] = True
            print(f"  Inflection point at valley: index {valley_idx}")

        # STEP 3: Ascent from the valley to the end
        end_altitude_m = max(line_data['elevation'].iloc[-1] + self.nom_flight_height, self.min_agl + line_data['elevation'].iloc[-1])
        for i in range(valley_idx + 1 if valley_idx is not None else highest_peak_idx + 1, len(line_data)):
            progress = (line_data.loc[i, 'cum_distance'] - (valley_dist if valley_idx is not None else highest_peak_dist)) / \
                       (line_data['cum_distance'].iloc[-1] - (valley_dist if valley_idx is not None else highest_peak_dist))
            line_data.loc[i, 'flight_altitude_msl'] = descent_altitude_m + progress * (end_altitude_m - descent_altitude_m)
            line_data.loc[i, 'agl'] = line_data.loc[i, 'flight_altitude_msl'] - line_data.loc[i, 'elevation']
            line_data.loc[i, 'mode'] = 'ascending'

        # Mark the end as an inflection point
        line_data.loc[len(line_data) - 1, 'is_inflection'] = True
        print(f"  Inflection point at end of line: index {len(line_data) - 1}")

        return line_data

    def plot_flight_lines(self, data: pd.DataFrame, output_dir: str = "png"):
        """
        Create plots for each flight line showing elevation, flight altitude, and inflection points.

        Args:
            data: DataFrame with flight plan data
            output_dir: Directory for saving the PNG files
        """
        import os
        import matplotlib.pyplot as plt

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Plot each flight line
        for line_id in data['line_id'].unique():
            line_data = data[data['line_id'] == line_id]

            plt.figure(figsize=(10, 6))
            plt.plot(line_data['cum_distance'], line_data['elevation'], label='Terrain Elevation', color='brown')
            plt.plot(line_data['cum_distance'], line_data['flight_altitude_msl'], label='Flight Altitude', color='blue')
            plt.scatter(line_data[line_data['is_inflection']]['cum_distance'],
                        line_data[line_data['is_inflection']]['flight_altitude_msl'],
                        color='red', label='Inflection Points')

            plt.title(f"Flight Line {line_id}")
            plt.xlabel("Cumulative Distance (m)")
            plt.ylabel("Altitude (m)")
            plt.legend()
            plt.grid()

            # Save the plot
            plot_path = os.path.join(output_dir, f"flight_line_{line_id}.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved plot for flight line {line_id} to {plot_path}")

    def save_inflection_points(self, data: pd.DataFrame, output_path: str):
        """
        Save inflection points to a CSV file.

        Args:
            data: DataFrame with flight plan data
            output_path: Path for the output CSV file
        """
        # Extract only inflection points
        inflection_points = data[data['is_inflection'] == True].copy()

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Save to CSV
        inflection_points.to_csv(output_path, index=False)
        print(f"Saved inflection points to {output_path}")

def main():
    """
    Main function to execute the flight planning process.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Variable Altitude Flight Planner v2')
    
    # Input/output paths
    parser.add_argument('--input', type=str, required=True,
                      help='Path to input CSV file with elevation data')
    parser.add_argument('--turning-altitudes', type=str, default=None,
                      help='Path to CSV file with turning area altitudes')
    parser.add_argument('--output', type=str, default='inflection_points.csv',
                      help='Path to output CSV file for inflection points')
    parser.add_argument('--plot-dir', type=str, default='png',
                      help='Directory for output plot PNG files')
    parser.add_argument('--input-dir', type=str, default="",
                      help='Directory for input CSV files')
    parser.add_argument('--output-dir', type=str, default="",
                      help='Directory for output files')
    
    # Flight parameters
    parser.add_argument('--min-agl', type=float, default=490.0,
                      help='Minimum height above ground level for eye safety (m)')
    parser.add_argument('--buffer-height', type=float, default=100.0,
                      help='Additional buffer height (m)')
    parser.add_argument('--nom-flight-height', type=float, default=625.0,
                      help='Nominal flight height (m)')
    parser.add_argument('--climb-rate', type=float, default=400.0,
                      help='Climb/descent rate (ft/min)')
    parser.add_argument('--aircraft-speed', type=float, default=97.0,
                      help='Aircraft speed (knots)')
    
    # Specific line parameter
    parser.add_argument('--line-id', type=int, default=None,
                      help='Process only a specific flight line ID (if omitted, all flight lines are processed)')
    
    args = parser.parse_args()
    
    # Print current directory for debugging
    print(f"Current working directory: {os.getcwd()}")
    
    # Adjust output filename based on whether we're processing a specific line or all lines
    if args.line_id is not None:
        # Modify the output filename to include the line_id
        filename, ext = os.path.splitext(args.output)
        output_file = f"{filename}_line_{args.line_id}{ext}"
    else:
        # For all lines, modify to include "all" indicator
        filename, ext = os.path.splitext(args.output)
        output_file = f"{filename}_all{ext}"
    
    # Print configuration
    print("Flight Planner Configuration:")
    print(f"  Input file: {args.input}")
    if args.turning_altitudes:
        print(f"  Turning altitudes file: {args.turning_altitudes}")
    print(f"  Output file: {output_file}")
    print(f"  Plot directory: {args.plot_dir}")
    print(f"  Input directory: {args.input_dir if args.input_dir else 'Using default'}")
    print(f"  Output directory: {args.output_dir if args.output_dir else 'Using default'}")
    if args.line_id is not None:
        print(f"  Processing only flight line: {args.line_id}")
    print("\nFlight parameters:")
    print(f"  Minimum AGL: {args.min_agl} m")
    print(f"  Buffer height: {args.buffer_height} m")
    print(f"  Nominal flight height: {args.nom_flight_height} m")
    print(f"  Climb/descent rate: {args.climb_rate} ft/min")
    print(f"  Aircraft speed: {args.aircraft_speed} knots")
    
    # Create the flight planner with the specified parameters
    # Default paths will be used if not provided
    planner = FlightPlannerV2(
        min_agl=args.min_agl,
        buffer_height=args.buffer_height,
        nom_flight_height=args.nom_flight_height,
        climb_rate=args.climb_rate,
        aircraft_speed=args.aircraft_speed,
        input_dir=args.input_dir if args.input_dir else "C:\\Users\\jmusinsky\\Documents\\Data\\TopoFlight\\Conversions\\pointsToElev",
        output_dir=args.output_dir if args.output_dir else "C:\\Users\\jmusinsky\\Documents\\Data\\TopoFlight\\Conversions\\pointsToElev"
    )
    
    try:
        # Load the input data
        data = planner.load_data(args.input)
        
        # Load turning altitudes if provided
        if args.turning_altitudes:
            print(f"\nLoading turning altitudes from {args.turning_altitudes}...")
            planner.load_turning_altitudes(args.turning_altitudes)
        
        # Calculate the flight plan (for specific line or all lines)
        print("\nCalculating flight plan...")
        flight_plan = planner.calculate_flight_plan(data, args.line_id)
        
        # Plot the flight lines
        print(f"Creating plots in {args.plot_dir} directory...")
        planner.plot_flight_lines(flight_plan, output_dir=args.plot_dir)
        
        # Save the inflection points with the modified filename
        print(f"Saving inflection points to {output_file}...")
        planner.save_inflection_points(flight_plan, output_path=output_file)
        
        print("\nFlight planning completed successfully.")
        
    except Exception as e:
        print(f"\nError during flight planning: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nFlight planning failed.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)