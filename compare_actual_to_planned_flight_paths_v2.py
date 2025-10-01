import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer
from scipy.interpolate import interp1d
import os

# ============================================================================
# USER DEFINED PARAMETERS
# ============================================================================

# Line number (used throughout for file names and plot title)
LINE_NUMBER = 7

# Actual flight direction: "NtoS" (north to south) or "StoN" (south to north)
ACTUAL_FLIGHT_DIRECTION = "NtoS"
#ACTUAL_FLIGHT_DIRECTION = "StoN"

# UTM Zone
UTM_ZONE = "13N"

# Planned flight line path and file
PLANNED_PATH = r"C:\Users\jmusinsky\Documents\Data\TopoFlight\Conversions\pointsToElev\export_output\D13_ALMO_S2_P4_625m_max_v7_VQ-780"
PLANNED_FILE = f"flight_track_line_{LINE_NUMBER}_NtoS.csv"

# Actual flight track path and file
ACTUAL_PATH = r"C:\Users\jmusinsky\Documents\Data\TopoFlight\Conversions\pointsToElev\data_out"
ACTUAL_FILE = f"flight_trajectory_CRBU_P1C1_L1_GPSIMU_sbet_2025061614_subset_line{LINE_NUMBER}.csv"

# Output path for plots
OUTPUT_PATH = r"C:\Users\jmusinsky\Documents\Data\TopoFlight\Conversions\pointsToElev\data_out\png"
OUTPUT_FILE_COMPARISON = f"flight_comparison_line_{LINE_NUMBER}.png"
OUTPUT_FILE_VARIATION = f"flight_comparison_line_{LINE_NUMBER}_flightline_variation.png"
OUTPUT_FILE_COMBINED = f"flight_comparison_line_{LINE_NUMBER}_combined.png"

# ============================================================================
# FUNCTIONS
# ============================================================================

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

def load_and_process_planned_track(planned_filepath, utm_zone):
    """Load planned flight track and process coordinates."""
    print(f"Loading planned flight track: {planned_filepath}")
    df = pd.read_csv(planned_filepath)
    
    # Calculate aircraft altitude (elevation + AGL)
    df['aircraft_altitude'] = df['elevation'] + df['agl_m']
    
    # Calculate distance along track
    df['distance_m'] = calculate_distance_along_track(df['x'].values, df['y'].values)
    
    print(f"  Loaded {len(df)} planned waypoints")
    print(f"  Distance range: {df['distance_m'].min():.1f} to {df['distance_m'].max():.1f} m")
    
    return df

def load_and_process_actual_track(actual_filepath, utm_zone, flight_direction):
    """Load actual flight track, convert coordinates, and calculate distances."""
    print(f"Loading actual flight track: {actual_filepath}")
    df = pd.read_csv(actual_filepath)
    
    # Convert lat/lon to UTM
    print("  Converting coordinates to UTM...")
    utm_coords = [latlon_to_utm(lat, lon, utm_zone) 
                  for lat, lon in zip(df['LATITUDE'], df['LONGITUDE'])]
    df['x'] = [coord[0] for coord in utm_coords]
    df['y'] = [coord[1] for coord in utm_coords]
    
    # If flight direction is south to north, reverse the order of points
    if flight_direction == "StoN":
        print("  Reversing flight direction (South to North)")
        df = df.iloc[::-1].reset_index(drop=True)
    
    # Calculate distance along track
    df['distance_m'] = calculate_distance_along_track(df['x'].values, df['y'].values)
    
    print(f"  Loaded {len(df)} actual track points")
    print(f"  Flight direction: {flight_direction}")
    
    return df

def truncate_actual_track(actual_df, min_distance, max_distance):
    """Truncate actual track to match planned track distance range."""
    mask = (actual_df['distance_m'] >= min_distance) & (actual_df['distance_m'] <= max_distance)
    truncated = actual_df[mask].copy()
    truncated['distance_m'] = truncated['distance_m'] - min_distance
    
    print(f"  Truncated actual track: {len(truncated)} points within planned range")
    
    return truncated

def create_comparison_plot(planned_df, actual_df, line_number, output_filepath):
    """Create comparison plot of planned vs actual flight tracks."""
    print("\nCreating comparison plot...")
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot ground elevation (filled area)
    ax.fill_between(planned_df['distance_m'], 0, planned_df['elevation'], 
                     color='tan', alpha=0.6, label='Terrain')
    
    # Plot planned flight track
    ax.plot(planned_df['distance_m'], planned_df['aircraft_altitude'], 
            color='blue', linewidth=2, label='Planned Flight Path', linestyle='--')
    
    # Plot actual flight track
    ax.plot(actual_df['distance_m'], actual_df['ELEVATION'], 
            color='red', linewidth=2, label='Actual Flight Path')
    
    # Labels and formatting
    ax.set_xlabel('Distance along flight line (m)', fontsize=12)
    ax.set_ylabel('Altitude (m above sea level)', fontsize=12)
    ax.set_title(f'Flight Line {line_number} - Variable Altitude Flight Plan Comparison', 
                 fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    # Don't invert x-axis - north (start, distance=0) is on left, south (end) is on right
    # This is the natural orientation since distance increases from north to south
    
    # Add circled S in bottom right corner to indicate south
    ax.text(0.98, 0.02, 'S', transform=ax.transAxes,
            fontsize=16, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black', linewidth=2))
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_filepath}")
    
    # Close the figure
    plt.close(fig)

def interpolate_planned_altitude(planned_df, actual_distances):
    """Interpolate planned altitude at actual flight track distances."""
    f = interp1d(planned_df['distance_m'], planned_df['aircraft_altitude'], 
                 kind='linear', fill_value='extrapolate')
    interpolated_altitudes = f(actual_distances)
    return interpolated_altitudes

def create_variation_plot(planned_df, actual_df, line_number, output_filepath):
    """Create plot showing variation of actual flight path from planned."""
    print("\nCreating variation plot...")
    
    # Interpolate planned altitude at actual track distances
    planned_interpolated = interpolate_planned_altitude(planned_df, actual_df['distance_m'])
    
    # Calculate variation (actual - planned)
    variation = actual_df['ELEVATION'].values - planned_interpolated
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Plot zero line (planned flight path)
    ax.axhline(y=0, color='black', linewidth=2, label='Planned Flight Path', zorder=2)
    
    # Plot variation as filled area with different colors
    ax.fill_between(actual_df['distance_m'], 0, variation, 
                     where=(variation >= 0), color='green', alpha=0.6, 
                     interpolate=True, label='Actual flight path above planned line')
    ax.fill_between(actual_df['distance_m'], 0, variation, 
                     where=(variation < 0), color='gold', alpha=0.6, 
                     interpolate=True, label='Actual flight path below planned line')
    
    # Plot variation line
    ax.plot(actual_df['distance_m'], variation, color='black', linewidth=1, alpha=0.5, zorder=3)
    
    # Labels and formatting
    ax.set_xlabel('Distance along flight line (m)', fontsize=12)
    ax.set_ylabel('Altitude Variation (m)', fontsize=12)
    ax.set_title(f'Flight Line {line_number} - Actual vs Planned Altitude Variation', 
                 fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='both')
    
    # Legend - place below the plot
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper center', 
              bbox_to_anchor=(0.5, -0.20), fontsize=10, framealpha=0.9, ncol=3)
    
    # Add statistics text box in upper left
    mean_var = np.mean(variation)
    std_var = np.std(variation)
    max_var = np.max(variation)
    min_var = np.min(variation)
    
    stats_text = f'Mean: {mean_var:.1f}m\nStd: {std_var:.1f}m\nMax: {max_var:.1f}m\nMin: {min_var:.1f}m'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Add circled S in bottom right corner (higher up to avoid covering x-axis label)
    ax.text(0.98, 0.08, 'S', transform=ax.transAxes,
            fontsize=16, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black', linewidth=2))
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    print(f"Variation plot saved to: {output_filepath}")
    
    # Close the figure
    plt.close(fig)
    
    return variation

def create_combined_plot(planned_df, actual_df, line_number, output_filepath):
    """Create combined plot with comparison on top and variation on bottom."""
    print("\nCreating combined plot...")
    
    # Interpolate planned altitude for variation calculation
    planned_interpolated = interpolate_planned_altitude(planned_df, actual_df['distance_m'])
    variation = actual_df['ELEVATION'].values - planned_interpolated
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                     gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.3})
    
    # ========== TOP PLOT: Comparison ==========
    # Plot ground elevation (filled area)
    ax1.fill_between(planned_df['distance_m'], 0, planned_df['elevation'], 
                     color='tan', alpha=0.6, label='Terrain')
    
    # Plot planned flight track
    ax1.plot(planned_df['distance_m'], planned_df['aircraft_altitude'], 
            color='blue', linewidth=2, label='Planned Flight Path', linestyle='--')
    
    # Plot actual flight track
    ax1.plot(actual_df['distance_m'], actual_df['ELEVATION'], 
            color='red', linewidth=2, label='Actual Flight Path')
    
    # Labels and formatting for top plot
    ax1.set_xlabel('Distance along flight line (m)', fontsize=12)
    ax1.set_ylabel('Altitude (m above sea level)', fontsize=12)
    ax1.set_title(f'Flight Line {line_number} - Variable Altitude Flight Plan Comparison', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    ax1.set_ylim(bottom=0)
    
    # Add circled S in bottom right corner of top plot
    ax1.text(0.98, 0.02, 'S', transform=ax1.transAxes,
            fontsize=16, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black', linewidth=2))
    
    # ========== BOTTOM PLOT: Variation ==========
    # Plot zero line (planned flight path)
    ax2.axhline(y=0, color='black', linewidth=2, label='Planned Flight Path', zorder=2)
    
    # Plot variation as filled area with different colors
    ax2.fill_between(actual_df['distance_m'], 0, variation, 
                     where=(variation >= 0), color='green', alpha=0.6, 
                     interpolate=True, label='Actual flight path above planned line')
    ax2.fill_between(actual_df['distance_m'], 0, variation, 
                     where=(variation < 0), color='gold', alpha=0.6, 
                     interpolate=True, label='Actual flight path below planned line')
    
    # Plot variation line
    ax2.plot(actual_df['distance_m'], variation, color='black', linewidth=1, alpha=0.5, zorder=3)
    
    # Labels and formatting for bottom plot
    ax2.set_xlabel('Distance along flight line (m)', fontsize=12)
    ax2.set_ylabel('Altitude Variation (m)', fontsize=12)
    ax2.set_title(f'Flight Line {line_number} - Actual vs Planned Altitude Variation', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='both')
    
    # Legend for bottom plot - place below the plot
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='upper center', 
              bbox_to_anchor=(0.5, -0.20), fontsize=10, framealpha=0.9, ncol=3)
    
    # Add statistics text box in upper left of bottom plot
    mean_var = np.mean(variation)
    std_var = np.std(variation)
    max_var = np.max(variation)
    min_var = np.min(variation)
    
    stats_text = f'Mean: {mean_var:.1f}m\nStd: {std_var:.1f}m\nMax: {max_var:.1f}m\nMin: {min_var:.1f}m'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Add circled S in bottom right corner of bottom plot
    ax2.text(0.98, 0.08, 'S', transform=ax2.transAxes,
            fontsize=16, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black', linewidth=2))
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to: {output_filepath}")
    
    # Close the figure
    plt.close(fig)
    
    return variation

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    try:
        # Construct file paths
        planned_filepath = os.path.join(PLANNED_PATH, PLANNED_FILE)
        actual_filepath = os.path.join(ACTUAL_PATH, ACTUAL_FILE)
        output_comparison = os.path.join(OUTPUT_PATH, OUTPUT_FILE_COMPARISON)
        output_variation = os.path.join(OUTPUT_PATH, OUTPUT_FILE_VARIATION)
        output_combined = os.path.join(OUTPUT_PATH, OUTPUT_FILE_COMBINED)
        
        # Check if input files exist
        if not os.path.exists(planned_filepath):
            print(f"Error: Planned flight file not found: {planned_filepath}")
            return
        if not os.path.exists(actual_filepath):
            print(f"Error: Actual flight file not found: {actual_filepath}")
            return
        
        print("=" * 70)
        print(f"Processing Flight Line {LINE_NUMBER}")
        print("=" * 70)
        
        # Load and process planned track
        planned_df = load_and_process_planned_track(planned_filepath, UTM_ZONE)
        
        # Load and process actual track
        actual_df = load_and_process_actual_track(actual_filepath, UTM_ZONE, ACTUAL_FLIGHT_DIRECTION)
        
        # Get distance range from planned track
        min_distance = planned_df['distance_m'].min()
        max_distance = planned_df['distance_m'].max()
        
        # Truncate actual track to planned range
        actual_df_truncated = truncate_actual_track(actual_df, min_distance, max_distance)
        
        # Create comparison plot
        create_comparison_plot(planned_df, actual_df_truncated, LINE_NUMBER, output_comparison)
        
        # Create variation plot
        variation = create_variation_plot(planned_df, actual_df_truncated, LINE_NUMBER, output_variation)
        
        # Create combined plot
        create_combined_plot(planned_df, actual_df_truncated, LINE_NUMBER, output_combined)
        
        print("\n" + "=" * 70)
        print("Processing Complete!")
        print("=" * 70)
        print(f"\nAltitude variation statistics:")
        print(f"  Mean: {np.mean(variation):.1f} m")
        print(f"  Std Dev: {np.std(variation):.1f} m")
        print(f"  Max (above): {np.max(variation):.1f} m")
        print(f"  Min (below): {np.min(variation):.1f} m")
        print(f"\nThree plots saved to: {OUTPUT_PATH}")
        print(f"  1. {OUTPUT_FILE_COMPARISON}")
        print(f"  2. {OUTPUT_FILE_VARIATION}")
        print(f"  3. {OUTPUT_FILE_COMBINED}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()