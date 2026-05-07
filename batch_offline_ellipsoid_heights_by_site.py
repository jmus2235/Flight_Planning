#!/usr/bin/env python3
"""
Query geoid heights for NEON sites using pyproj

This uses the pyproj library which has built-in geoid models (EGM96/EGM2008).
No need to download NGS binary files!

Requirements:
    conda install -c conda-forge pyproj pandas

Usage:
    python neon_geoid_query_pyproj.py input.csv [output.csv] [--model egm2008]
    
Example:
    python neon_geoid_query_pyproj.py NEON_site_tower_coordinates.csv
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

try:
    from pyproj import Transformer, CRS
except ImportError:
    print("ERROR: pyproj is not installed.", file=sys.stderr)
    print("Please install it with: conda install -c conda-forge pyproj", file=sys.stderr)
    sys.exit(1)


def get_geoid_height(lat, lon, model='egm2008'):
    """
    Get geoid height (ellipsoid-geoid separation) using pyproj
    
    Parameters:
    -----------
    lat : float
        Latitude in decimal degrees
    lon : float
        Longitude in decimal degrees (negative for West)
    model : str
        Geoid model: 'egm96' or 'egm2008' (default)
    
    Returns:
    --------
    float or None : Geoid height in meters
                   Positive = geoid above ellipsoid
                   Negative = geoid below ellipsoid
    """
    try:
        # Select the appropriate geoid model
        if model.lower() == 'egm96':
            # EGM96 geoid model
            geoid_crs = "EPSG:5773"  # EGM96 height
        else:  # egm2008
            # EGM2008 geoid model (more accurate, similar to GEOID18)
            geoid_crs = "EPSG:3855"  # EGM2008 height
        
        # Create transformer from ellipsoid to geoid
        # We transform a point at ellipsoid height = 0
        # The resulting height is the geoid separation
        transformer = Transformer.from_crs(
            "EPSG:4979",  # WGS84 3D (lat, lon, ellipsoid height)
            f"EPSG:4326+{geoid_crs}",  # WGS84 + geoid model
            always_xy=True
        )
        
        # Transform: input is (lon, lat, 0) for ellipsoid
        # Output z-coordinate is the difference
        x, y, z = transformer.transform(lon, lat, 0.0)
        
        # The z value represents the geoid height
        # Negative z means we need to flip the sign
        geoid_height = -z
        
        return geoid_height
        
    except Exception as e:
        print(f"  ERROR: Could not compute geoid height for {lat}, {lon}: {e}", 
              file=sys.stderr)
        return None


def process_neon_sites(input_csv, output_csv=None, model='egm2008'):
    """
    Process NEON sites CSV and compute geoid heights
    
    Parameters:
    -----------
    input_csv : str
        Path to input CSV with LAT and LON columns
    output_csv : str, optional
        Path to output CSV (default: input_with_geoid.csv)
    model : str
        Geoid model to use ('egm96' or 'egm2008')
    """
    
    # Read input CSV
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"ERROR: Could not read input file '{input_csv}': {e}", file=sys.stderr)
        sys.exit(1)
    
    # Check for required columns
    required_cols = ['LAT', 'LON']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)
    
    # Set default output filename
    if output_csv is None:
        input_path = Path(input_csv)
        output_csv = input_path.parent / f"{input_path.stem}_with_geoid{input_path.suffix}"
    
    model_upper = model.upper()
    print(f"Processing {len(df)} NEON sites...")
    print(f"Using geoid model: {model_upper}")
    print(f"Output file: {output_csv}")
    print("=" * 80)
    
    # Query geoid heights
    geoid_heights = []
    ellipsoid_minus_geoid = []
    
    for idx, row in df.iterrows():
        site = row.get('SITE', f'Site_{idx}')
        domain = row.get('DOMAIN', '')
        lat = row['LAT']
        lon = row['LON']
        
        # Get geoid height
        geoid_ht = get_geoid_height(lat, lon, model=model)
        
        if geoid_ht is not None:
            geoid_heights.append(geoid_ht)
            ellipsoid_minus_geoid.append(-geoid_ht)
            
            # Determine relationship
            if geoid_ht > 0:
                rel = "above"
                sign = "+"
            elif geoid_ht < 0:
                rel = "below"
                sign = ""
            else:
                rel = "equal"
                sign = " "
            
            print(f"  {domain:3} | {site:12} | {lat:9.5f}°, {lon:10.5f}° | "
                  f"{sign}{geoid_ht:6.2f}m ({rel} ellipsoid)")
        else:
            geoid_heights.append(None)
            ellipsoid_minus_geoid.append(None)
            print(f"  {domain:3} | {site:12} | {lat:9.5f}°, {lon:10.5f}° | ERROR")
    
    # Add results to dataframe
    df['geoid_height_m'] = geoid_heights
    df['ellipsoid_minus_geoid_m'] = ellipsoid_minus_geoid
    df['geoid_model'] = model_upper
    
    # Calculate statistics for successful queries
    valid_heights = [h for h in geoid_heights if h is not None]
    
    print("=" * 80)
    if valid_heights:
        print(f"\nSUMMARY:")
        print(f"  Successfully processed: {len(valid_heights)}/{len(df)} sites")
        print(f"  Minimum geoid height: {min(valid_heights):+.2f}m (most below ellipsoid)")
        print(f"  Maximum geoid height: {max(valid_heights):+.2f}m (most above ellipsoid)")
        print(f"  Mean geoid height:    {sum(valid_heights)/len(valid_heights):+.2f}m")
        print(f"  Range across sites:   {max(valid_heights) - min(valid_heights):.2f}m")
        
        print(f"\nNote: {model_upper} and GEOID18 typically differ by <5cm for CONUS sites")
    else:
        print("\nWARNING: No successful queries!")
    
    # Save output
    try:
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to: {output_csv}")
    except Exception as e:
        print(f"\nERROR: Could not save output file '{output_csv}': {e}", file=sys.stderr)
        sys.exit(1)
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Query geoid heights using pyproj (EGM96/EGM2008 models)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s NEON_site_tower_coordinates.csv
  %(prog)s input.csv output.csv
  %(prog)s input.csv --model egm96

Available geoid models:
  egm2008  - Earth Gravitational Model 2008 (recommended, similar to GEOID18)
  egm96    - Earth Gravitational Model 1996 (older, less accurate)

Note: EGM2008 and GEOID18 are very similar for CONUS sites (typically <5cm difference).
      GEOID18 is a hybrid model based on EGM2008 with additional gravimetric data.

Requirements:
  conda install -c conda-forge pyproj pandas
        """
    )
    
    parser.add_argument('input_csv', 
                        help='Input CSV file with LAT and LON columns')
    parser.add_argument('output_csv', nargs='?', default=None,
                        help='Output CSV file (default: input_with_geoid.csv)')
    parser.add_argument('--model', default='egm2008',
                        choices=['egm96', 'egm2008'],
                        help='Geoid model to use (default: egm2008)')
    
    args = parser.parse_args()
    
    # Process the sites
    process_neon_sites(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        model=args.model
    )


if __name__ == '__main__':
    main()