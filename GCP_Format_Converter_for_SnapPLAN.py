#!/usr/bin/env python3
"""
GCP Format Converter for Unsupported Software
Converts CSV coordinate files to the specific GCP format required by unsupported software

Usage:
    python gcp_formatter.py input.csv output.txt "Flight_Plan_Name"
    python gcp_formatter.py CRBU_inflection_points_A_XYTableToPoint_v2.csv CRBU_GCPs.txt "D13_CRBU_inflection_points"
"""

import pandas as pd
import argparse
import sys
from pathlib import Path

class GCPFormatter:
    def __init__(self):
        pass
    
    def read_csv_file(self, input_file):
        """Read the CSV file and extract coordinate data"""
        try:
            df = pd.read_csv(input_file)
            print(f"Successfully read {len(df)} rows from {input_file}")
            print("Columns found:", list(df.columns))
            
            # Check for required columns
            if 'POINT_X' not in df.columns or 'POINT_Y' not in df.columns:
                print("Error: Required columns POINT_X and POINT_Y not found in CSV")
                print("Available columns:", list(df.columns))
                return None
            
            print(f"\nFirst few rows of coordinate data:")
            print(df[['POINT_X', 'POINT_Y']].head())
            
            return df
            
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None
    
    def format_gcp_output(self, df, flight_plan_name):
        """Format the data according to the required GCP format"""
        output_lines = []
        
        # Add header lines
        output_lines.append(f"Ground control points for the flight plan: {flight_plan_name}")
        output_lines.append("Coordinates are in WGS84")
        output_lines.append("XY = horizontal point, Z = height point, XYZ = horizontal + height point)")
        output_lines.append("[xyz controls]")
        
        # Process each row
        for index, row in df.iterrows():
            # Create sequential ID starting from 1
            point_id = index + 1
            
            # Zero-pad the point name to 3 digits (001, 002, etc.)
            point_name = f"{point_id:03d}"
            
            # Get coordinates (POINT_Y = latitude, POINT_X = longitude)
            latitude = row['POINT_Y']
            longitude = row['POINT_X']
            
            # Format as: ID=point_name,XYZ,latitude,longitude
            gcp_line = f"{point_id}={point_name},XYZ,{latitude},{longitude}"
            output_lines.append(gcp_line)
        
        return output_lines
    
    def convert(self, input_file, output_file, flight_plan_name):
        """Main conversion function"""
        print(f"Converting {input_file} to GCP format for flight plan: {flight_plan_name}")
        
        # Read the CSV file
        df = self.read_csv_file(input_file)
        if df is None:
            return False
        
        # Format the output
        output_lines = self.format_gcp_output(df, flight_plan_name)
        
        # Write output file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(output_lines))
            
            print(f"\nConversion complete! Output written to: {output_file}")
            print(f"Converted {len(df)} points")
            
            # Show preview of output
            print("\nOutput preview:")
            for i, line in enumerate(output_lines):
                print(line)
                if i >= 7:  # Show header + first few points
                    print("...")
                    break
            
            return True
            
        except Exception as e:
            print(f"Error writing output file: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Convert CSV coordinates to GCP format for unsupported software')
    parser.add_argument('input_csv', help='Input CSV file with POINT_X and POINT_Y columns')
    parser.add_argument('output_txt', help='Output text file in GCP format')
    parser.add_argument('flight_plan_name', help='Name of the flight plan for the header')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input_csv).exists():
        print(f"Error: Input file {args.input_csv} does not exist")
        sys.exit(1)
    
    # Perform conversion
    formatter = GCPFormatter()
    success = formatter.convert(args.input_csv, args.output_txt, args.flight_plan_name)
    
    if success:
        print("Conversion completed successfully!")
    else:
        print("Conversion failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()