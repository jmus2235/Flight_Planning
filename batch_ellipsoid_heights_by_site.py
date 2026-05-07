#!/usr/bin/env python3
"""
Query GEOID18 heights for NEON tower sites

Usage:
    python neon_geoid_query.py input.csv [output.csv] [--model geoid18]
    
Example:
    python neon_geoid_query.py NEON_site_tower_coordinates.csv
    python neon_geoid_query.py NEON_site_tower_coordinates.csv results.csv --model geoid18
"""

import sys
import argparse
import pandas as pd
import requests
import time
from pathlib import Path


def get_geoid_height(lat, lon, model='geoid18'):
    """
    Get geoid height (ellipsoid-geoid separation) for a specific location
    
    Parameters:
    -----------
    lat : float
        Latitude in decimal degrees
    lon : float
        Longitude in decimal degrees (negative for West)
    model : str
        Geoid model to use (default: 'geoid18')
        Options: geoid18, geoid12b, geoid12a, geoid09, geoid03, geoid99, geoid96
    
    Returns:
    --------
    float or None : Geoid height in meters
                   Positive = geoid above ellipsoid
                   Negative = geoid below ellipsoid
    """
    url = "https://geodesy.noaa.gov/api/geoid/ght"
    params = {
        'lat': lat,
        'lon': lon,
        'model': model
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('geoidHeight')
        else:
            print(f"  WARNING: API returned status {response.status_code} for {lat}, {lon}", 
                  file=sys.stderr)
            return None
    except requests.exceptions.RequestException as e:
        print(f"  ERROR: Network error for {lat}, {lon}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  ERROR: Unexpected error for {lat}, {lon}: {e}", file=sys.stderr)
        return None


def process_neon_sites(input_csv, output_csv=None, model='geoid18', delay=0.5):
    """
    Process NEON sites CSV and query geoid heights
    
    Parameters:
    -----------
    input_csv : str
        Path to input CSV with DOMAIN, SITE, LAT, LON columns
    output_csv : str, optional
        Path to output CSV (default: input_with_geoid.csv)
    model : str
        Geoid model to use
    delay : float
        Delay in seconds between API calls (be nice to the server)
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
    
    print(f"Processing {len(df)} NEON sites...")
    print(f"Using geoid model: {model.upper()}")
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
        
        # Query API
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
        
        # Rate limiting
        if idx < len(df) - 1:  # Don't sleep after last request
            time.sleep(delay)
    
    # Add results to dataframe
    df['geoid_height_m'] = geoid_heights
    df['ellipsoid_minus_geoid_m'] = ellipsoid_minus_geoid
    df['geoid_model'] = model.upper()
    
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
        description='Query GEOID heights for NEON tower sites',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s NEON_site_tower_coordinates.csv
  %(prog)s input.csv output.csv
  %(prog)s input.csv --model geoid12b
  %(prog)s input.csv output.csv --model geoid18 --delay 1.0

Available geoid models:
  geoid18  - GEOID18 (2019, recommended for CONUS)
  geoid12b - GEOID12B (2012)
  geoid12a - GEOID12A (2012)
  geoid09  - GEOID09 (2009)
  geoid03  - GEOID03 (2003)
  geoid99  - GEOID99 (1999)
  geoid96  - GEOID96 (1996)
        """
    )
    
    parser.add_argument('input_csv', 
                        help='Input CSV file with LAT and LON columns')
    parser.add_argument('output_csv', nargs='?', default=None,
                        help='Output CSV file (default: input_with_geoid.csv)')
    parser.add_argument('--model', default='geoid18',
                        choices=['geoid18', 'geoid12b', 'geoid12a', 'geoid09', 
                                'geoid03', 'geoid99', 'geoid96'],
                        help='Geoid model to use (default: geoid18)')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between API calls in seconds (default: 0.5)')
    
    args = parser.parse_args()
    
    # Process the sites
    process_neon_sites(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        model=args.model,
        delay=args.delay
    )


if __name__ == '__main__':
    main()