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
    
    def plot_flight_lines(self, data: pd.DataFrame, output_dir: str = "png"):
        """
        Create plots for each flight line showing elevation, flight altitude, and inflection points.
        
        Args:
            data: DataFrame with flight plan data
            output_dir: Directory for saving the PNG files
        """
        # Ensure we have a valid output directory
        # If output_dir is a relative path, create it under self.output_dir
        if not os.path.isabs(output_dir) and self.output_dir:
            full_output_dir = os.path.join(self.output_dir, output_dir)
        else:
            full_output_dir = output_dir
        
        try:
            os.makedirs(full_output_dir, exist_ok=True)
            print(f"Output directory for plots: {full_output_dir}")
        except Exception as e:
            print(f"Warning: Could not create output directory {full_output_dir}: {str(e)}")
            print("Attempting to use current directory instead.")
            full_output_dir = "png"
            os.makedirs(full_output_dir, exist_ok=True)
        
        # Plot each flight line separately
        for line_id in data['line_id'].unique():
            line_data = data[data['line_id'] == line_id].copy()
            
            # Before plotting, make sure there are no NaN values
            if line_data['flight_altitude_msl'].isna().any():
                print(f"  Warning: Flight line {line_id} has NaN values in flight_altitude_msl. Filling with interpolation.")
                # Try to interpolate missing values
                line_data['flight_altitude_msl'] = line_data['flight_altitude_msl'].interpolate()
                
            if line_data['flight_altitude_ft'].isna().any():
                print(f"  Warning: Flight line {line_id} has NaN values in flight_altitude_ft. Recalculating.")
                # Recalculate feet from meters
                line_data['flight_altitude_ft'] = line_data['flight_altitude_msl'] * 3.28084
                
            if line_data['agl'].isna().any():
                print(f"  Warning: Flight line {line_id} has NaN values in agl. Recalculating.")
                # Recalculate AGL
                line_data['agl'] = line_data['flight_altitude_msl'] - line_data['elevation']
            
            # Create the plot
            plt.figure(figsize=(15, 8))
            
            # Plot the terrain
            plt.fill_between(line_data['cum_distance'], 0, line_data['elevation'], 
                             color='saddlebrown', alpha=0.5, label='Terrain')
            
            # Plot the minimum safe height (min_agl)
            plt.plot(line_data['cum_distance'], 
                     line_data['elevation'] + self.min_agl, 
                     'r--', label=f'Eye Safety Limit ({self.min_agl}m)')
            
            # Plot the safety height (min_agl + buffer)
            plt.plot(line_data['cum_distance'], 
                     line_data['elevation'] + self.safety_height, 
                     'r-', label=f'Safety Height ({self.safety_height}m)')
            
            # Plot the nominal flight height (dotted line)
            plt.plot(line_data['cum_distance'], 
                     line_data['elevation'] + self.nom_flight_height, 
                     'k:', label=f'Nominal Height ({self.nom_flight_height}m)')
            
            # Plot the actual flight path
            plt.plot(line_data['cum_distance'], 
                     line_data['flight_altitude_msl'], 
                     'b-', linewidth=2, label='Flight Path')
            
            # Plot inflection points
            inflection_points = line_data[line_data['is_inflection'] == True].copy()
            
            # Ensure no NaN values in inflection points
            inflection_points = inflection_points.dropna(subset=['flight_altitude_msl', 'flight_altitude_ft', 'agl'])
            
            plt.scatter(inflection_points['cum_distance'], 
                        inflection_points['flight_altitude_msl'], 
                        color='blue', s=100, marker='o', 
                        label='Inflection Points')
            
            # Add labels for each inflection point
            for idx, row in inflection_points.iterrows():
                try:
                    # Round flight altitude to nearest 100 feet
                    alt_ft_value = int(round(row['flight_altitude_ft'] / 100) * 100)
                    # Calculate AGL without rounding (just format for display)
                    agl_value = int(round(row['agl']))
                    
                    plt.annotate(f"Mode: {row['mode']}\nAGL: {agl_value}m\nAlt: {alt_ft_value}ft", 
                                (row['cum_distance'], row['flight_altitude_msl']),
                                xytext=(0, 10), textcoords='offset points',
                                ha='center')
                except (ValueError, TypeError) as e:
                    print(f"  Warning: Could not annotate inflection point at index {idx}: {str(e)}")
                    print(f"  Values: flight_altitude_ft={row.get('flight_altitude_ft', 'N/A')}, agl={row.get('agl', 'N/A')}")
            
            # Set title and labels
            plt.title(f'Flight Line {line_id} - Variable Altitude Flight Plan')
            plt.xlabel('Distance along flight line (m)')
            plt.ylabel('Altitude (m above sea level)')
            plt.grid(True)
            plt.legend()
            
            # Adjust margins
            plt.tight_layout()
            
            # Save the plot
            plot_filename = f"flight_line_{line_id}.png"
            plot_path = os.path.join(full_output_dir, plot_filename)
            
            try:
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"  Saved plot for flight line {line_id} to {plot_path}")
            except Exception as e:
                print(f"  Error saving plot for flight line {line_id}: {str(e)}")
                # Try alternative location
                alt_path = plot_filename
                try:
                    plt.savefig(alt_path, dpi=300, bbox_inches='tight')
                    print(f"  Saved plot to alternative path: {alt_path}")
                except Exception as e2:
                    print(f"  Failed to save plot to alternative path: {str(e2)}")
                
            plt.close()
    
    def save_inflection_points(self, data: pd.DataFrame, output_path: str):
        """
        Save inflection points to a CSV file with fixed distance calculation.
        
        Args:
            data: DataFrame with flight plan data
            output_path: Path for the output CSV file
        
        Returns:
            DataFrame with saved inflection points
        """
        # If output_path is a relative path, create it under self.output_dir
        if not os.path.isabs(output_path) and self.output_dir:
            full_path = os.path.join(self.output_dir, output_path)
        else:
            full_path = output_path
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(full_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                print(f"Created output directory: {output_dir}")
            except Exception as e:
                print(f"Warning: Could not create output directory {output_dir}: {str(e)}")
        
        # Extract only inflection points
        inflection_points = data[data['is_inflection'] == True].copy()
        
        print("\nChecking for NaN values in inflection points:")
        for column in inflection_points.columns:
            nan_count = inflection_points[column].isna().sum()
            if nan_count > 0:
                print(f"  Column '{column}': {nan_count} NaN values")
        
        # Make sure flight_altitude_ft is calculated
        if 'flight_altitude_ft' not in inflection_points.columns:
            inflection_points['flight_altitude_ft'] = inflection_points['flight_altitude_msl'] * 3.28084
        
        # Round flight altitude to nearest 100 feet
        if 'flight_altitude_ft' in inflection_points.columns:
            inflection_points['flight_altitude_ft'] = (inflection_points['flight_altitude_ft'] / 100).round() * 100
        
        # Recalculate AGL based on the rounded flight altitude
        if all(col in inflection_points.columns for col in ['flight_altitude_ft', 'elevation']):
            flight_alt_meters = inflection_points['flight_altitude_ft'] / 3.28084
            inflection_points['agl'] = flight_alt_meters - inflection_points['elevation']
        
        # Initialize distance columns - fill with 0 to avoid NaN
        inflection_points['distance_from_south'] = 0
        inflection_points['distance_from_north'] = 0
        inflection_points['distance_to_next'] = 0
        
        print("\nProcessing inflection points by flight line:")
        
        # Process each flight line separately and create a list of processed dataframes
        processed_dfs = []
        
        for line_id in inflection_points['line_id'].unique():
            print(f"  Flight line {line_id}:")
            
            try:
                # Get all points for this flight line (from original data)
                line_data = data[data['line_id'] == line_id].copy()
                
                if len(line_data) == 0:
                    print(f"    No data found for flight line {line_id}")
                    continue
                
                # Get entire flight line for north/south extent determination
                # Caution: this includes ALL points in the flight line, not just inflection points
                
                # Find actual southernmost and northernmost points
                # The y-coordinate is northing in UTM, so higher y = more north
                min_y = line_data['y'].min()
                max_y = line_data['y'].max()
                
                # Find the actual southernmost and northernmost points
                south_idx = line_data[line_data['y'] == min_y]['y'].idxmin()
                north_idx = line_data[line_data['y'] == max_y]['y'].idxmax()
                
                south_point = (line_data.loc[south_idx, 'x'], line_data.loc[south_idx, 'y'])
                north_point = (line_data.loc[north_idx, 'x'], line_data.loc[north_idx, 'y'])
                
                print(f"    Southernmost point: ({south_point[0]:.2f}, {south_point[1]:.2f})")
                print(f"    Northernmost point: ({north_point[0]:.2f}, {north_point[1]:.2f})")
                print(f"    Flight line spans from y={min_y:.2f} to y={max_y:.2f}")
                print(f"    Total north-south span: {max_y - min_y:.2f} meters")
                
                # Get all inflection points for this line
                line_inflections = inflection_points[inflection_points['line_id'] == line_id].copy()
                
                if len(line_inflections) == 0:
                    print(f"    No inflection points found for flight line {line_id}")
                    continue
                
                # Sort by cumulative distance to ensure correct order
                if 'cum_distance' in line_inflections.columns:
                    line_inflections = line_inflections.sort_values('cum_distance').reset_index(drop=True)
                
                # Assign sequential point_ids starting from 0
                # This ensures each inflection point has a unique ID within its flight line
                line_inflections['point_id'] = range(len(line_inflections))
                
                # Calculate distances for each inflection point
                print(f"    Inflection points coordinates (ordered):")
                for i, row in line_inflections.iterrows():
                    x = row['x']
                    y = row['y']
                    print(f"      Point {i}: ({x:.2f}, {y:.2f})")
                    
                    # Calculate distances from south and north reference points
                    dist_south = sqrt((x - south_point[0])**2 + (y - south_point[1])**2)
                    dist_north = sqrt((x - north_point[0])**2 + (y - north_point[1])**2)
                    
                    # Log the results for debugging
                    print(f"      Distance from south point: {dist_south:.2f}m")
                    print(f"      Distance from north point: {dist_north:.2f}m")
                    
                    # Store the distances (rounded to nearest meter)
                    line_inflections.loc[i, 'distance_from_south'] = round(dist_south)
                    line_inflections.loc[i, 'distance_from_north'] = round(dist_north)
                
                # Calculate distances between consecutive points
                print(f"    Distances between consecutive points:")
                for i in range(len(line_inflections) - 1):
                    pt1 = line_inflections.iloc[i]
                    pt2 = line_inflections.iloc[i+1]
                    
                    # Calculate Euclidean distance between consecutive points
                    dist = sqrt((pt2['x'] - pt1['x'])**2 + (pt2['y'] - pt1['y'])**2)
                    print(f"      Distance from point {i} to point {i+1}: {dist:.2f}m")
                    
                    # Store the distance (rounded to nearest meter)
                    line_inflections.loc[i, 'distance_to_next'] = round(dist)
                    
                # Add this line's processed inflection points to our results
                processed_dfs.append(line_inflections)
                    
            except Exception as e:
                print(f"    Error processing flight line {line_id}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Combine all processed flight lines
        if processed_dfs:
            inflection_points = pd.concat(processed_dfs, ignore_index=True)
        else:
            print("No valid inflection points found.")
            return pd.DataFrame()
        
        # Select and order columns for output
        output_cols = [
            'line_id', 'point_id', 'x', 'y', 'elevation', 
            'flight_altitude_ft', 'mode', 'distance_to_next', 'agl',
            'distance_from_south', 'distance_from_north'
        ]
        
        # Filter columns that exist
        existing_cols = [col for col in output_cols if col in inflection_points.columns]
        
        # Create output dataframe
        output_df = inflection_points[existing_cols].copy()
        output_df = output_df.rename(columns={
            'flight_altitude_ft': 'flight_altitude'
        })
        
        print(f"\nSaving {len(output_df)} inflection points to {full_path}")
        
        # Debug: final check of distance values
        print("\nFinal distance values check:")
        for line_id in output_df['line_id'].unique():
            line_points = output_df[output_df['line_id'] == line_id].sort_values('point_id')
            print(f"  Flight line {line_id}:")
            for _, row in line_points.iterrows():
                print(f"    Point {row['point_id']}: distance_to_next={row['distance_to_next']}m, distance_from_south={row['distance_from_south']}m, distance_from_north={row['distance_from_north']}m")
        
        # Save to CSV with error handling
        try:
            output_df.to_csv(full_path, index=False)
            print(f"File saved successfully to {full_path}")
        except Exception as e:
            print(f"Error saving file to {full_path}: {str(e)}")
            # Try alternative save path
            alt_path = "inflection_points_backup.csv"
            print(f"Attempting to save to alternative path: {alt_path}")
            try:
                output_df.to_csv(alt_path, index=False)
                print(f"File saved to alternative path: {alt_path}")
            except Exception as e2:
                print(f"Failed to save to alternative path: {str(e2)}")
        
        return output_df
            
    def _process_flight_line(self, line_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process a single flight line to determine inflection points based on peak elevation logic.
        Follows the revised logic:
        1. Identify which end of the flight line has the highest elevation peak
        2. Set starting altitude based on either turning altitude or elevation + nom_flight_height
        3. Maintain fixed altitude until the highest elevation point
        4. Calculate descent path based on climb rate
        5. Ensure minimum AGL requirements are met during descent
        
        Args:
            line_data: DataFrame with elevation data for a single flight line
        
        Returns:
            DataFrame with inflection points and flight plan for the line
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
        
        total_distance = line_data['cum_distance'].iloc[-1]
        midpoint_distance = total_distance / 2
        
        # Find the midpoint index
        midpoint_idx = 0
        for i in range(len(line_data)):
            if line_data.loc[i, 'cum_distance'] >= midpoint_distance:
                midpoint_idx = i
                break
        
        print(f"  Flight line midpoint at index {midpoint_idx}, distance {midpoint_distance:.2f}m")
        
        # Divide the flight line into halves
        first_half = line_data.iloc[:midpoint_idx+1]
        second_half = line_data.iloc[midpoint_idx:]
        
        # Find the highest elevation in each half
        first_half_highest_idx = first_half['elevation'].idxmax()
        first_half_highest_elev = line_data.loc[first_half_highest_idx, 'elevation']
        first_half_highest_dist = line_data.loc[first_half_highest_idx, 'cum_distance']
        
        second_half_highest_idx = second_half['elevation'].idxmax()
        second_half_highest_elev = line_data.loc[second_half_highest_idx, 'elevation']
        second_half_highest_dist = line_data.loc[second_half_highest_idx, 'cum_distance']
        
        print(f"  First half highest elevation: {first_half_highest_elev:.2f}m at index {first_half_highest_idx}, distance {first_half_highest_dist:.2f}m")
        print(f"  Second half highest elevation: {second_half_highest_elev:.2f}m at index {second_half_highest_idx}, distance {second_half_highest_dist:.2f}m")
        
        # Determine which half has the higher peak and should be the start
        if first_half_highest_elev >= second_half_highest_elev:
            # First half has higher peak
            if abs(first_half_highest_idx - north_idx) <= abs(first_half_highest_idx - south_idx):
                # Peak is closer to north end
                start_idx = north_idx
                end_idx = south_idx
                highest_peak_idx = first_half_highest_idx
                highest_peak_elev = first_half_highest_elev
                highest_peak_dist = first_half_highest_dist
                start_end = "north"
                end_end = "south"
                direction = "north_to_south"
                start_half_highest_elev = first_half_highest_elev
                end_half_highest_elev = second_half_highest_elev
            else:
                # Peak is closer to south end
                start_idx = south_idx
                end_idx = north_idx
                highest_peak_idx = first_half_highest_idx
                highest_peak_elev = first_half_highest_elev
                highest_peak_dist = first_half_highest_dist
                start_end = "south"
                end_end = "north"
                direction = "south_to_north"
                start_half_highest_elev = first_half_highest_elev
                end_half_highest_elev = second_half_highest_elev
        else:
            # Second half has higher peak
            if abs(second_half_highest_idx - north_idx) <= abs(second_half_highest_idx - south_idx):
                # Peak is closer to north end
                start_idx = north_idx
                end_idx = south_idx
                highest_peak_idx = second_half_highest_idx
                highest_peak_elev = second_half_highest_elev
                highest_peak_dist = second_half_highest_dist
                start_end = "north"
                end_end = "south"
                direction = "north_to_south"
                start_half_highest_elev = second_half_highest_elev
                end_half_highest_elev = first_half_highest_elev
            else:
                # Peak is closer to south end
                start_idx = south_idx
                end_idx = north_idx
                highest_peak_idx = second_half_highest_idx
                highest_peak_elev = second_half_highest_elev
                highest_peak_dist = second_half_highest_dist
                start_end = "south"
                end_end = "north"
                direction = "south_to_north"
                start_half_highest_elev = second_half_highest_elev
                end_half_highest_elev = first_half_highest_elev
        
        # If the flight line needs to be reversed to make the start correct
        # (This ensures the start_idx is at index 0)
        if start_idx != 0:
            print(f"  Reversing flight line to make {start_end} the start")
            line_data = line_data.iloc[::-1].reset_index(drop=True)
            
            # Recalculate indices after reversal
            start_idx = 0
            end_idx = len(line_data) - 1
            
            # Recalculate the highest peak index after reversal
            highest_peak_idx = len(line_data) - 1 - highest_peak_idx
            
            # Recalculate cumulative distance
            line_data['cum_distance'] = 0
            for i in range(1, len(line_data)):
                dx = line_data.loc[i, 'x'] - line_data.loc[i-1, 'x']
                dy = line_data.loc[i, 'y'] - line_data.loc[i-1, 'y']
                dist = sqrt(dx**2 + dy**2)
                line_data.loc[i, 'cum_distance'] = line_data.loc[i-1, 'cum_distance'] + dist
                
            highest_peak_dist = line_data.loc[highest_peak_idx, 'cum_distance']
        
        print(f"  Flight direction: {direction}")
        print(f"  Highest peak at index {highest_peak_idx}, distance {highest_peak_dist:.2f}m, elevation: {highest_peak_elev:.2f}m")
        
        # Add columns for flight altitude planning
        line_data['flight_altitude_msl'] = np.nan  # Flight altitude in meters above sea level
        line_data['flight_altitude_ft'] = np.nan   # Flight altitude in feet above sea level
        line_data['agl'] = np.nan  # Height above ground level
        line_data['mode'] = None   # Flight mode (level, climbing, descending)
        line_data['is_inflection'] = False  # Flag for inflection points
        line_data['dist_from_prev'] = 0.0  # Distance from previous inflection point
        
        # Calculate starting flight altitude
        # Option 1: Highest peak elevation + nominal flight height, rounded up to nearest 100 ft
        peak_plus_nominal_m = start_half_highest_elev + self.nom_flight_height
        peak_plus_nominal_ft = peak_plus_nominal_m * 3.28084
        peak_plus_nominal_ft = ceil(peak_plus_nominal_ft / 100) * 100  # Round up to nearest 100ft
        peak_plus_nominal_m = peak_plus_nominal_ft / 3.28084
        
        # Option 2: Turning altitude for this end (start) of the flight line
        has_turning_altitude = line_id in self.turning_altitudes
        
        if has_turning_altitude:
            turning_altitude_ft = self.turning_altitudes[line_id]['start_ft']
            turning_altitude_m = self.turning_altitudes[line_id]['start_m']
            print(f"  Turning altitude for {start_end} (start): {turning_altitude_ft:.0f}ft ({turning_altitude_m:.2f}m)")
        else:
            turning_altitude_ft = 0
            turning_altitude_m = 0
            print(f"  No turning altitude available for {start_end} (start)")
        
        # Use the higher of the two options for starting altitude
        if has_turning_altitude and turning_altitude_ft > peak_plus_nominal_ft:
            start_altitude_ft = turning_altitude_ft
            start_altitude_m = turning_altitude_m
            print(f"  Using turning altitude for start: {start_altitude_ft:.0f}ft ({start_altitude_m:.2f}m)")
        else:
            start_altitude_ft = peak_plus_nominal_ft
            start_altitude_m = peak_plus_nominal_m
            print(f"  Using peak elevation + nominal height for start: {start_altitude_ft:.0f}ft ({start_altitude_m:.2f}m)")
        
        # Calculate end-of-descent altitude
        # Option 1: End half highest peak elevation + (nominal flight height + buffer), rounded up to nearest 100 ft
        end_peak_plus_safety_m = end_half_highest_elev + self.nom_flight_height + self.buffer_height
        end_peak_plus_safety_ft = end_peak_plus_safety_m * 3.28084
        end_peak_plus_safety_ft = ceil(end_peak_plus_safety_ft / 100) * 100  # Round up to nearest 100ft
        end_peak_plus_safety_m = end_peak_plus_safety_ft / 3.28084
        
        # Option 2: Turning altitude for the end of the flight line
        end_turning_altitude_ft = 0
        end_turning_altitude_m = 0
        has_end_turning_altitude = has_turning_altitude and self.turning_altitudes[line_id]['has_end_altitude']
        
        if has_end_turning_altitude:
            end_turning_altitude_ft = self.turning_altitudes[line_id]['end_ft']
            end_turning_altitude_m = self.turning_altitudes[line_id]['end_m']
            print(f"  Turning altitude for {end_end} (end): {end_turning_altitude_ft:.0f}ft ({end_turning_altitude_m:.2f}m)")
        else:
            print(f"  No turning altitude available for {end_end} (end)")
        
        # Use the higher of the two options for ending altitude
        if has_end_turning_altitude and end_turning_altitude_ft > end_peak_plus_safety_ft:
            end_altitude_ft = end_turning_altitude_ft
            end_altitude_m = end_turning_altitude_m
            print(f"  Using turning altitude for end: {end_altitude_ft:.0f}ft ({end_altitude_m:.2f}m)")
        else:
            end_altitude_ft = end_peak_plus_safety_ft
            end_altitude_m = end_peak_plus_safety_m
            print(f"  Using peak elevation + safety height for end: {end_altitude_ft:.0f}ft ({end_altitude_m:.2f}m)")
        
        # STEP 1: Set level flight from start to highest peak (inflection point 1)
        print(f"  Setting level flight from start to highest peak at {start_altitude_ft:.0f}ft")
        for i in range(0, highest_peak_idx + 1):
            line_data.loc[i, 'flight_altitude_msl'] = start_altitude_m
            line_data.loc[i, 'flight_altitude_ft'] = start_altitude_ft
            line_data.loc[i, 'agl'] = start_altitude_m - line_data.loc[i, 'elevation']
            line_data.loc[i, 'mode'] = 'level'
        
        # Mark the start as an inflection point
        line_data.loc[0, 'is_inflection'] = True
        print(f"  Inflection point 1: start of line (level), position: ({line_data.loc[0, 'x']:.2f}, {line_data.loc[0, 'y']:.2f})")
        
        # Mark the highest peak as an inflection point
        line_data.loc[highest_peak_idx, 'is_inflection'] = True
        print(f"  Inflection point 2: highest peak (level->descending), position: ({line_data.loc[highest_peak_idx, 'x']:.2f}, {line_data.loc[highest_peak_idx, 'y']:.2f})")
        
        # STEP 2: Calculate the theoretical descent from highest peak to end
        # How much distance do we need to descend from start_altitude to end_altitude?
        altitude_diff_m = start_altitude_m - end_altitude_m
        
        if altitude_diff_m <= 0:
            # No descent needed, maintain level flight
            print(f"  No descent needed (end altitude >= start altitude), maintaining level flight")
            for i in range(highest_peak_idx, len(line_data)):
                line_data.loc[i, 'flight_altitude_msl'] = start_altitude_m
                line_data.loc[i, 'flight_altitude_ft'] = start_altitude_ft
                line_data.loc[i, 'agl'] = start_altitude_m - line_data.loc[i, 'elevation']
                line_data.loc[i, 'mode'] = 'level'
            
            # Mark the end as an inflection point
            line_data.loc[end_idx, 'is_inflection'] = True
            print(f"  Inflection point 3: end of line (level), position: ({line_data.loc[end_idx, 'x']:.2f}, {line_data.loc[end_idx, 'y']:.2f})")
            
            # No further processing needed
            return line_data
        
        # Calculate how many segments we need to descend at the given rate
        # This is the horizontal distance needed to achieve the descent based on climb rate
        descent_distance_needed = abs(altitude_diff_m / (self.climb_rate / 60) * self.aircraft_speed)
        
        print(f"  Theoretical descent: {altitude_diff_m:.2f}m over {descent_distance_needed:.1f}m")
        
        # Calculate where the descent should end based on the highest peak position + descent distance
        theoretical_end_of_descent_dist = highest_peak_dist + descent_distance_needed
        
        # Find the point closest to this theoretical end distance
        theoretical_end_of_descent_idx = len(line_data) - 1  # Default to end of line
        for i in range(highest_peak_idx + 1, len(line_data)):
            if line_data.loc[i, 'cum_distance'] >= theoretical_end_of_descent_dist:
                theoretical_end_of_descent_idx = i
                break
        
        print(f"  Theoretical end of descent distance: {theoretical_end_of_descent_dist:.2f}m")
        print(f"  Theoretical end of descent index: {theoretical_end_of_descent_idx}")
        
        # Calculate distance between highest peak and theoretical end of descent
        descent_segment_distance = line_data.loc[theoretical_end_of_descent_idx, 'cum_distance'] - line_data.loc[highest_peak_idx, 'cum_distance']
        
        # STEP 3: Calculate actual descent path checking AGL requirements
        # Flag to track if we found any AGL violations
        agl_violation_found = False
        
        # Find the last point where AGL >= nom_flight_height during descent
        last_safe_idx = highest_peak_idx
        
        # Loop through all points from peak to theoretical end of descent
        for i in range(highest_peak_idx + 1, theoretical_end_of_descent_idx + 1):
            # Calculate position along the descent path (0 to 1)
            descent_progress = (line_data.loc[i, 'cum_distance'] - line_data.loc[highest_peak_idx, 'cum_distance']) / descent_segment_distance
            
            # Calculate altitude based on linear interpolation for a smooth diagonal line
            # This gives us a constant descent rate
            current_alt_msl = start_altitude_m - (descent_progress * altitude_diff_m)
            
            # Make sure we don't go below the end altitude
            if current_alt_msl < end_altitude_m:
                current_alt_msl = end_altitude_m
            
            # Calculate AGL at this point
            current_agl = current_alt_msl - line_data.loc[i, 'elevation']
            
            # Store the flight altitude and AGL for this point
            line_data.loc[i, 'flight_altitude_msl'] = current_alt_msl
            line_data.loc[i, 'flight_altitude_ft'] = current_alt_msl * 3.28084
            line_data.loc[i, 'agl'] = current_agl
            line_data.loc[i, 'mode'] = 'descending'
            
            # Check for eye safety buffer violation
            if current_agl < self.safety_height:
                # AGL violation found
                agl_violation_found = True
                print(f"  AGL violation at index {i}: AGL={current_agl:.2f}m < safety_height={self.safety_height:.2f}m")
                break
            
            # If AGL is still >= nominal flight height, update last safe index
            if current_agl >= self.nom_flight_height:
                last_safe_idx = i
        
        # STEP 4: Determine where to place the second inflection point based on AGL violations
        if agl_violation_found:
            # Use the last safe index where AGL >= nom_flight_height
            inflection_point_idx = last_safe_idx
            print(f"  AGL violation found, setting inflection point at last safe index {inflection_point_idx}")
            
            # Recalculate the altitude at this inflection point (round to nearest 100ft)
            inflection_alt_ft = line_data.loc[inflection_point_idx, 'flight_altitude_ft']
            inflection_alt_ft = round(inflection_alt_ft / 100) * 100
            inflection_alt_m = inflection_alt_ft / 3.28084
            
            # Update the altitude at the inflection point
            line_data.loc[inflection_point_idx, 'flight_altitude_msl'] = inflection_alt_m
            line_data.loc[inflection_point_idx, 'flight_altitude_ft'] = inflection_alt_ft
            line_data.loc[inflection_point_idx, 'agl'] = inflection_alt_m - line_data.loc[inflection_point_idx, 'elevation']
        else:
            # No violations, use theoretical end of descent
            inflection_point_idx = theoretical_end_of_descent_idx
            print(f"  No AGL violations, setting inflection point at theoretical end of descent index {inflection_point_idx}")
            
            # Set the altitude to the calculated end altitude
            line_data.loc[inflection_point_idx, 'flight_altitude_msl'] = end_altitude_m
            line_data.loc[inflection_point_idx, 'flight_altitude_ft'] = end_altitude_ft
            line_data.loc[inflection_point_idx, 'agl'] = end_altitude_m - line_data.loc[inflection_point_idx, 'elevation']
        
        # Mark the inflection point
        line_data.loc[inflection_point_idx, 'is_inflection'] = True
        line_data.loc[inflection_point_idx, 'mode'] = 'descending'
        print(f"  Inflection point 3: end of descent (descending->level), position: ({line_data.loc[inflection_point_idx, 'x']:.2f}, {line_data.loc[inflection_point_idx, 'y']:.2f})")
        
        # STEP 5: Set level flight from inflection point to end
        # Get the altitude at the inflection point
        level_altitude_m = line_data.loc[inflection_point_idx, 'flight_altitude_msl']
        level_altitude_ft = line_data.loc[inflection_point_idx, 'flight_altitude_ft']
        
        # Set level flight from inflection point to end
        for i in range(inflection_point_idx + 1, len(line_data)):
            line_data.loc[i, 'flight_altitude_msl'] = level_altitude_m
            line_data.loc[i, 'flight_altitude_ft'] = level_altitude_ft
            line_data.loc[i, 'agl'] = level_altitude_m - line_data.loc[i, 'elevation']
            line_data.loc[i, 'mode'] = 'level'
        
        # Mark the end as an inflection point
        line_data.loc[end_idx, 'is_inflection'] = True
        print(f"  Inflection point 4: end of line (level), position: ({line_data.loc[end_idx, 'x']:.2f}, {line_data.loc[end_idx, 'y']:.2f})")
        
        # Calculate distances between inflection points
        inflection_points = line_data[line_data['is_inflection'] == True]
        print(f"  Distance between inflection points:")
        
        prev_x, prev_y = None, None
        for i, (idx, row) in enumerate(inflection_points.iterrows()):
            if prev_x is not None and prev_y is not None:
                dx = row['x'] - prev_x
                dy = row['y'] - prev_y
                dist = sqrt(dx**2 + dy**2)
                line_data.loc[idx, 'dist_from_prev'] = dist
                print(f"    Distance from point {i} to {i+1}: {dist:.2f}m")
            prev_x, prev_y = row['x'], row['y']
        
        return line_data
        
        # Calculate how many segments we need to descend at the given rate
        segments_needed = abs(altitude_diff_m / self.climb_rate_per_segment)
        theoretical_end_of_descent_idx = min(highest_peak_idx + int(segments_needed), len(line_data) - 1)
        
        print(f"  Theoretical descent: {altitude_diff_m:.2f}m over {segments_needed:.1f} segments")
        print(f"  Theoretical end of descent index: {theoretical_end_of_descent_idx}")
        
        # STEP 3: Calculate actual descent path checking AGL requirements
        current_alt_msl = start_altitude_m
        
        # Initialize variables to track where we might need to level off
        safe_descent_end_idx = highest_peak_idx  # Start with the peak as the safe end point
        
        # Find the last point where AGL >= nom_flight_height during descent
        last_safe_idx = highest_peak_idx
        
        # Flag to track if we found any AGL violations
        agl_violation_found = False
        
        # Calculate the total descent distance and altitude change
        descent_distance = line_data.loc[theoretical_end_of_descent_idx, 'cum_distance'] - line_data.loc[highest_peak_idx, 'cum_distance']
        altitude_change = start_altitude_m - end_altitude_m
        
        # Calculate the total descent distance and altitude change
        descent_distance = line_data.loc[theoretical_end_of_descent_idx, 'cum_distance'] - line_data.loc[highest_peak_idx, 'cum_distance']
        altitude_change = start_altitude_m - end_altitude_m
        
        # Loop through all points from peak to theoretical end of descent
        for i in range(highest_peak_idx + 1, theoretical_end_of_descent_idx + 1):
            # Calculate position along the descent path (0 to 1)
            position = (line_data.loc[i, 'cum_distance'] - line_data.loc[highest_peak_idx, 'cum_distance']) / descent_distance
            
            # Calculate altitude based on linear interpolation for a smooth diagonal line
            current_alt_msl = start_altitude_m - (position * altitude_change)
            
            # Make sure we don't go below the end altitude
            if current_alt_msl < end_altitude_m:
                current_alt_msl = end_altitude_m
            
            # Calculate AGL at this point
            current_agl = current_alt_msl - line_data.loc[i, 'elevation']
            
            if current_agl < self.safety_height:
                # AGL violation found
                agl_violation_found = True
                print(f"  AGL violation at index {i}: AGL={current_agl:.2f}m < safety_height={self.safety_height:.2f}m")
                break
            
            # If AGL is still >= nominal flight height, update last safe index
            if current_agl >= self.nom_flight_height:
                last_safe_idx = i
            
            # Store the flight altitude and AGL for this point
            line_data.loc[i, 'flight_altitude_msl'] = current_alt_msl
            line_data.loc[i, 'flight_altitude_ft'] = current_alt_msl * 3.28084
            line_data.loc[i, 'agl'] = current_agl
            line_data.loc[i, 'mode'] = 'descending'
        
        # STEP 4: Determine where to place the second inflection point
        if agl_violation_found:
            # Use the last safe index where AGL >= nom_flight_height
            inflection_point_idx = last_safe_idx
            print(f"  AGL violation found, setting inflection point at last safe index {inflection_point_idx}")
            
            # Recalculate the altitude at this inflection point (round to nearest 100ft)
            inflection_alt_ft = line_data.loc[inflection_point_idx, 'flight_altitude_ft']
            inflection_alt_ft = round(inflection_alt_ft / 100) * 100
            inflection_alt_m = inflection_alt_ft / 3.28084
            
            # Update the altitude at the inflection point
            line_data.loc[inflection_point_idx, 'flight_altitude_msl'] = inflection_alt_m
            line_data.loc[inflection_point_idx, 'flight_altitude_ft'] = inflection_alt_ft
            line_data.loc[inflection_point_idx, 'agl'] = inflection_alt_m - line_data.loc[inflection_point_idx, 'elevation']
        else:
            # No violations, use theoretical end of descent
            inflection_point_idx = theoretical_end_of_descent_idx
            print(f"  No AGL violations, setting inflection point at theoretical end of descent index {inflection_point_idx}")
            
            # Set the altitude to the calculated end altitude
            line_data.loc[inflection_point_idx, 'flight_altitude_msl'] = end_altitude_m
            line_data.loc[inflection_point_idx, 'flight_altitude_ft'] = end_altitude_ft
            line_data.loc[inflection_point_idx, 'agl'] = end_altitude_m - line_data.loc[inflection_point_idx, 'elevation']
        
        # Mark the inflection point
        line_data.loc[inflection_point_idx, 'is_inflection'] = True
        print(f"  Inflection point 3: end of descent (descending->level), position: ({line_data.loc[inflection_point_idx, 'x']:.2f}, {line_data.loc[inflection_point_idx, 'y']:.2f})")
        
        # STEP 5: Set level flight from inflection point to end
        # Get the altitude at the inflection point
        level_altitude_m = line_data.loc[inflection_point_idx, 'flight_altitude_msl']
        level_altitude_ft = line_data.loc[inflection_point_idx, 'flight_altitude_ft']
        
        # Set level flight from inflection point to end
        for i in range(inflection_point_idx + 1, len(line_data)):
            line_data.loc[i, 'flight_altitude_msl'] = level_altitude_m
            line_data.loc[i, 'flight_altitude_ft'] = level_altitude_ft
            line_data.loc[i, 'agl'] = level_altitude_m - line_data.loc[i, 'elevation']
            line_data.loc[i, 'mode'] = 'level'
        
        # Mark the end as an inflection point
        line_data.loc[end_idx, 'is_inflection'] = True
        print(f"  Inflection point 4: end of line (level), position: ({line_data.loc[end_idx, 'x']:.2f}, {line_data.loc[end_idx, 'y']:.2f})")
        
        # Calculate distances between inflection points
        inflection_points = line_data[line_data['is_inflection'] == True]
        print(f"  Distance between inflection points:")
        
        prev_x, prev_y = None, None
        for i, (idx, row) in enumerate(inflection_points.iterrows()):
            if prev_x is not None and prev_y is not None:
                dx = row['x'] - prev_x
                dy = row['y'] - prev_y
                dist = sqrt(dx**2 + dy**2)
                line_data.loc[idx, 'dist_from_prev'] = dist
                print(f"    Distance from point {i} to {i+1}: {dist:.2f}m")
            prev_x, prev_y = row['x'], row['y']
        
        return line_data


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