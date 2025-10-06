import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer
from scipy.interpolate import interp1d
import os
import glob

# ============================================================================
# USER DEFINED PARAMETERS
# ============================================================================

# Site name to display in plot title
SITE = "Upper Taylor Watershed (UPTA)"

# Path to the CSV file containing flight line information
FLIGHT_LINES_CSV = r"C:\Users\jmusinsky\Documents\Data\TopoFlight\Conversions\pointsToElev\UPTA_GPS_start_end_times.csv"

# UTM Zone
UTM_ZONE = "13N"

# Planned flight line path
# Requires the planned flight tracks from the variable altitude flight plans
PLANNED_PATH = r"C:\Users\jmusinsky\Documents\Data\TopoFlight\Conversions\pointsToElev\export_output\D13_UPTA_S2_P3_625m_max_v7_VQ-780"

# Actual flight track path (where CSV files from shapefile extraction are saved)
ACTUAL_PATH = r"C:\Users\jmusinsky\Documents\Data\TopoFlight\Conversions\pointsToElev\data_out"

# Output path for plots
OUTPUT_PATH = r"C:\Users\jmusinsky\Documents\Data\TopoFlight\Conversions\pointsToElev\data_out\png"

# ============================================================================
# FUNCTIONS
# ============================================================================

def determine_flight_direction(north_time, south_time):
    """
    Determine flight direction based on GPS times.
    If north_time < south_time, flight is North to South.
    If south_time < north_time, flight is South to North.
    """
    if north_time < south_time:
        return "NtoS"
    else:
        return "StoN"

def find_actual_track_file(actual_path, line_number):
    """
    Find the actual track CSV file for a given line number.
    Searches for files matching pattern: *_line{line_number}.csv
    """
    pattern = os.path.join(actual_path, f"*_line{line_number}.csv")
    matching_files = glob.glob(pattern)
    
    if len(matching_files) == 0:
        return None
    elif len(matching_files) == 1:
        return matching_files[0]
    else:
        # If multiple files match, return the most recently modified
        return max(matching_files, key=os.path.getmtime)

def latlon_to_utm(lat, lon, utm_zone):
    """Convert latitude/longitude to UTM coordinates."""
    epsg_code = f"326{utm_zone[:-1]:0>2}" if utm_zone[-1] == 'N' else f"327{utm_zone[:-1]:0>2}"
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y

def calculate_distance_along_track(x, y):
    """Calculate cumulative distance along track from x, y coordinates."""
    dx = np.diff(x)
    dy = np.diff(y)
    distances = np.sqrt(dx**2 + dy**2)
    cumulative_distance = np.concatenate([[0], np.cumsum(distances)])
    return cumulative_distance

def load_and_process_planned_track(planned_filepath):
    """Load planned flight track and process coordinates."""
    df = pd.read_csv(planned_filepath)
    df['aircraft_altitude'] = df['elevation'] + df['agl_m']
    df['distance_m'] = calculate_distance_along_track(df['x'].values, df['y'].values)
    return df

def load_and_process_actual_track(actual_filepath, utm_zone, flight_direction):
    """Load actual flight track, convert coordinates, and calculate distances."""
    df = pd.read_csv(actual_filepath)
    
    # Convert lat/lon to UTM
    utm_coords = [latlon_to_utm(lat, lon, utm_zone) 
                  for lat, lon in zip(df['LATITUDE'], df['LONGITUDE'])]
    df['x'] = [coord[0] for coord in utm_coords]
    df['y'] = [coord[1] for coord in utm_coords]
    
    # If flight direction is south to north, reverse the order of points
    if flight_direction == "StoN":
        df = df.iloc[::-1].reset_index(drop=True)
    
    # Calculate distance along track
    df['distance_m'] = calculate_distance_along_track(df['x'].values, df['y'].values)
    
    return df

def truncate_actual_track(actual_df, min_distance, max_distance):
    """Truncate actual track to match planned track distance range."""
    mask = (actual_df['distance_m'] >= min_distance) & (actual_df['distance_m'] <= max_distance)
    truncated = actual_df[mask].copy()
    truncated['distance_m'] = truncated['distance_m'] - min_distance
    return truncated

def interpolate_planned_altitude(planned_df, actual_distances):
    """Interpolate planned altitude at actual flight track distances."""
    f = interp1d(planned_df['distance_m'], planned_df['aircraft_altitude'], 
                 kind='linear', fill_value='extrapolate')
    return f(actual_distances)

def interpolate_planned_coordinates(planned_df, actual_distances):
    """Interpolate planned x, y coordinates at actual flight track distances."""
    f_x = interp1d(planned_df['distance_m'], planned_df['x'], 
                   kind='linear', fill_value='extrapolate')
    f_y = interp1d(planned_df['distance_m'], planned_df['y'], 
                   kind='linear', fill_value='extrapolate')
    return f_x(actual_distances), f_y(actual_distances)

def calculate_perpendicular_distance(actual_x, actual_y, planned_x, planned_y):
    """
    Calculate perpendicular distance from actual track points to planned track line.
    Uses point-to-line-segment distance for each segment of the planned track.
    Returns signed distance (positive = right of planned line, negative = left).
    """
    n_actual = len(actual_x)
    perp_distances = np.zeros(n_actual)
    
    # For each actual point, find the perpendicular distance to the nearest planned segment
    for i in range(n_actual):
        point = np.array([actual_x[i], actual_y[i]])
        min_dist = float('inf')
        signed_dist = 0
        
        # Check distance to each segment of the planned line
        for j in range(len(planned_x) - 1):
            p1 = np.array([planned_x[j], planned_y[j]])
            p2 = np.array([planned_x[j + 1], planned_y[j + 1]])
            
            # Calculate perpendicular distance to line segment
            dist, sign = point_to_segment_distance(point, p1, p2)
            
            if dist < min_dist:
                min_dist = dist
                signed_dist = sign * dist
        
        perp_distances[i] = signed_dist
    
    return perp_distances

def point_to_segment_distance(point, seg_start, seg_end):
    """
    Calculate perpendicular distance from point to line segment.
    Returns (distance, sign) where sign indicates which side of the line.
    """
    # Vector from segment start to end
    seg_vec = seg_end - seg_start
    seg_length = np.linalg.norm(seg_vec)
    
    if seg_length == 0:
        return np.linalg.norm(point - seg_start), 1
    
    # Normalized segment vector
    seg_unit = seg_vec / seg_length
    
    # Vector from segment start to point
    point_vec = point - seg_start
    
    # Project point onto line (may be outside segment)
    projection_length = np.dot(point_vec, seg_unit)
    
    # Clamp projection to segment
    projection_length = np.clip(projection_length, 0, seg_length)
    
    # Closest point on segment
    closest_point = seg_start + projection_length * seg_unit
    
    # Distance vector from closest point to actual point
    dist_vec = point - closest_point
    distance = np.linalg.norm(dist_vec)
    
    # Determine sign using cross product (positive = right, negative = left)
    # In 2D: cross product z-component = seg_vec[0] * dist_vec[1] - seg_vec[1] * dist_vec[0]
    cross_z = seg_vec[0] * dist_vec[1] - seg_vec[1] * dist_vec[0]
    sign = 1 if cross_z >= 0 else -1
    
    return distance, sign

def create_combined_analysis_plot(planned_df, actual_df, line_number, site_name, output_filepath):
    """Create combined plot with comparison, vertical variance, and horizontal variance."""
    # Calculate variations
    planned_alt_interp = interpolate_planned_altitude(planned_df, actual_df['distance_m'])
    vertical_variation = actual_df['ELEVATION'].values - planned_alt_interp
    
    planned_x_interp, planned_y_interp = interpolate_planned_coordinates(
        planned_df, actual_df['distance_m'])
    horizontal_variation = calculate_perpendicular_distance(
        actual_df['x'].values, actual_df['y'].values,
        planned_x_interp, planned_y_interp)
    
    # Create figure with 3 subplots using GridSpec for individual spacing control
    fig = plt.figure(figsize=(14, 13))
    
    # Add overall figure title at the top with line break and extra spacing
    fig.suptitle(f'{site_name}\n\nFlight Line {line_number}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.35, top=0.92)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    # To move only the 3rd plot down, adjust its position after creation
    pos3 = ax3.get_position()
    ax3.set_position([pos3.x0, pos3.y0 - 0.02, pos3.width, pos3.height])
    
    # ========================================================================
    # TOP PLOT: Comparison of planned vs actual flight paths
    # ========================================================================
    ax1.fill_between(planned_df['distance_m'], 0, planned_df['elevation'], 
                     color='tan', alpha=0.6, label='Terrain')
    ax1.plot(planned_df['distance_m'], planned_df['aircraft_altitude'], 
            color='blue', linewidth=2, label='Planned Flight Path', linestyle='--')
    ax1.plot(actual_df['distance_m'], actual_df['ELEVATION'], 
            color='red', linewidth=2, label='Actual Flight Path')
    
    ax1.set_xlabel('Distance along flight line (m)', fontsize=12)
    ax1.set_ylabel('Altitude (m above sea level)', fontsize=12)
    ax1.set_title('Variable Altitude Flight Plan Comparison', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    ax1.set_ylim(bottom=0)
    ax1.text(0.98, 0.06, 'S', transform=ax1.transAxes,
            fontsize=16, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black', linewidth=2))
    
    # ========================================================================
    # MIDDLE PLOT: Vertical variation
    # ========================================================================
    ax2.axhline(y=0, color='black', linewidth=2, label='Planned Flight Path', zorder=2)
    ax2.fill_between(actual_df['distance_m'], 0, vertical_variation, 
                     where=(vertical_variation >= 0), color='green', alpha=0.6, 
                     interpolate=True, label='Actual flight path above planned line')
    ax2.fill_between(actual_df['distance_m'], 0, vertical_variation, 
                     where=(vertical_variation < 0), color='gold', alpha=0.6, 
                     interpolate=True, label='Actual flight path below planned line')
    ax2.plot(actual_df['distance_m'], vertical_variation, color='black', linewidth=1, alpha=0.5, zorder=3)
    
    ax2.set_xlabel('Distance along flight line (m)', fontsize=12)
    ax2.set_ylabel('Vertical Variation (m)', fontsize=12)
    ax2.set_title('Vertical Deviation from Planned Path', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='both')
    
    handles2, labels2 = ax2.get_legend_handles_labels()
    by_label2 = dict(zip(labels2, handles2))
    ax2.legend(by_label2.values(), by_label2.keys(), loc='upper center', 
              bbox_to_anchor=(0.5, -0.20), fontsize=10, framealpha=0.9, ncol=3)
    
    mean_vert = np.mean(vertical_variation)
    std_vert = np.std(vertical_variation)
    max_vert = np.max(vertical_variation)
    min_vert = np.min(vertical_variation)
    
    stats_text_vert = f'Mean: {mean_vert:.1f}m\nStd: {std_vert:.1f}m\nMax: {max_vert:.1f}m\nMin: {min_vert:.1f}m'
    ax2.text(0.02, 0.98, stats_text_vert, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    ax2.text(0.98, 0.12, 'S', transform=ax2.transAxes,
            fontsize=16, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black', linewidth=2))
    
    # ========================================================================
    # BOTTOM PLOT: Horizontal variation
    # ========================================================================
    ax3.axhline(y=0, color='black', linewidth=2, label='Planned Flight Path', zorder=2)
    ax3.fill_between(actual_df['distance_m'], 0, horizontal_variation, 
                     where=(horizontal_variation >= 0), color='purple', alpha=0.6, 
                     interpolate=True, label='Actual flight path right of planned line')
    ax3.fill_between(actual_df['distance_m'], 0, horizontal_variation, 
                     where=(horizontal_variation < 0), color='orange', alpha=0.6, 
                     interpolate=True, label='Actual flight path left of planned line')
    ax3.plot(actual_df['distance_m'], horizontal_variation, color='black', linewidth=1, alpha=0.5, zorder=3)
    
    ax3.set_xlabel('Distance along flight line (m)', fontsize=12)
    ax3.set_ylabel('Horizontal Variation (m)', fontsize=12)
    ax3.set_title('Horizontal Deviation from Planned Path', 
                 fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--', axis='both')
    
    handles3, labels3 = ax3.get_legend_handles_labels()
    by_label3 = dict(zip(labels3, handles3))
    ax3.legend(by_label3.values(), by_label3.keys(), loc='upper center', 
              bbox_to_anchor=(0.5, -0.20), fontsize=10, framealpha=0.9, ncol=3)
    
    mean_horiz = np.mean(horizontal_variation)
    std_horiz = np.std(horizontal_variation)
    max_horiz = np.max(horizontal_variation)
    min_horiz = np.min(horizontal_variation)
    abs_mean_horiz = np.mean(np.abs(horizontal_variation))
    
    stats_text_horiz = (f'Mean: {mean_horiz:.1f}m\nAbs Mean: {abs_mean_horiz:.1f}m\n'
                        f'Std: {std_horiz:.1f}m\nMax: {max_horiz:.1f}m\nMin: {min_horiz:.1f}m')
    ax3.text(0.02, 0.98, stats_text_horiz, transform=ax3.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    ax3.text(0.98, 0.12, 'S', transform=ax3.transAxes,
            fontsize=16, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black', linewidth=2))
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return vertical_variation, horizontal_variation

def process_single_line(line_number, north_time, south_time, site_name):
    """Process a single flight line."""
    print(f"\nProcessing Line {line_number}:")
    
    # Determine flight direction
    flight_direction = determine_flight_direction(north_time, south_time)
    print(f"  Flight direction: {flight_direction}")
    
    # Find actual track file
    actual_filepath = find_actual_track_file(ACTUAL_PATH, line_number)
    if actual_filepath is None:
        print(f"  Error: No actual track file found for line {line_number}")
        return False
    print(f"  Found actual track: {os.path.basename(actual_filepath)}")
    
    # Find planned track file - check both NtoS and StoN naming conventions
    planned_file_ntos = f"flight_track_line_{line_number}_NtoS.csv"
    planned_file_ston = f"flight_track_line_{line_number}_StoN.csv"
    
    planned_filepath_ntos = os.path.join(PLANNED_PATH, planned_file_ntos)
    planned_filepath_ston = os.path.join(PLANNED_PATH, planned_file_ston)
    
    if os.path.exists(planned_filepath_ntos):
        planned_filepath = planned_filepath_ntos
        planned_file = planned_file_ntos
    elif os.path.exists(planned_filepath_ston):
        planned_filepath = planned_filepath_ston
        planned_file = planned_file_ston
    else:
        print(f"  Error: Planned track file not found (checked both {planned_file_ntos} and {planned_file_ston})")
        return False
    print(f"  Found planned track: {planned_file}")
    
    try:
        # Load planned track first to determine its orientation
        planned_df = load_and_process_planned_track(planned_filepath)
        
        # Determine planned track orientation from filename
        if '_StoN.csv' in planned_file:
            planned_direction = "StoN"
        else:
            planned_direction = "NtoS"
        
        # Load actual track with NO reversal initially
        actual_df = load_and_process_actual_track(actual_filepath, UTM_ZONE, "NtoS")
        
        # Now check if we need to reverse: only if actual flight direction differs from planned
        if flight_direction != planned_direction:
            # Reverse the actual track to match planned orientation
            actual_df = actual_df.iloc[::-1].reset_index(drop=True)
            # Recalculate distances after reversal
            actual_df['distance_m'] = calculate_distance_along_track(actual_df['x'].values, actual_df['y'].values)
        
        # Truncate actual track
        min_distance = planned_df['distance_m'].min()
        max_distance = planned_df['distance_m'].max()
        actual_df_truncated = truncate_actual_track(actual_df, min_distance, max_distance)
        
        if len(actual_df_truncated) == 0:
            print(f"  Warning: No actual track points within planned range")
            return False
        
        # Create output filename
        output_combined = os.path.join(OUTPUT_PATH, f"flight_analysis_line_{line_number}_complete.png")
        
        # Generate combined plot
        vert_var, horiz_var = create_combined_analysis_plot(
            planned_df, actual_df_truncated, line_number, site_name, output_combined)
        
        # Print statistics
        print(f"  Vertical Statistics:")
        print(f"    Mean: {np.mean(vert_var):.1f}m, Std: {np.std(vert_var):.1f}m, "
              f"Max: {np.max(vert_var):.1f}m, Min: {np.min(vert_var):.1f}m")
        print(f"  Horizontal Statistics:")
        print(f"    Mean: {np.mean(horiz_var):.1f}m, Abs Mean: {np.mean(np.abs(horiz_var)):.1f}m, "
              f"Std: {np.std(horiz_var):.1f}m")
        print(f"    Max: {np.max(horiz_var):.1f}m, Min: {np.min(horiz_var):.1f}m")
        print(f"  ✓ Generated complete analysis plot")
        
        return True
        
    except Exception as e:
        print(f"  Error processing line {line_number}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main execution function."""
    try:
        # Check if CSV file exists
        if not os.path.exists(FLIGHT_LINES_CSV):
            print(f"Error: Flight lines CSV not found: {FLIGHT_LINES_CSV}")
            return
        
        # Read flight lines CSV
        print(f"Reading flight lines from: {FLIGHT_LINES_CSV}")
        print(f"Site: {SITE}")
        df = pd.read_csv(FLIGHT_LINES_CSV)
        
        # Validate required columns
        required_columns = ['Line', 'North', 'South']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns in CSV: {missing_columns}")
            return
        
        print(f"Found {len(df)} flight lines to process")
        print("=" * 70)
        
        # Process each line
        success_count = 0
        fail_count = 0
        
        for idx, row in df.iterrows():
            line_number = int(row['Line'])
            north_time = float(row['North'])
            south_time = float(row['South'])
            
            success = process_single_line(line_number, north_time, south_time, SITE)
            
            if success:
                success_count += 1
            else:
                fail_count += 1
        
        # Summary
        print("\n" + "=" * 70)
        print("Processing Summary")
        print("=" * 70)
        print(f"Total lines: {len(df)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {fail_count}")
        print(f"\nPlots saved to: {OUTPUT_PATH}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()