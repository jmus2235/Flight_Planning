import os
import glob
import pandas as pd

def combine_csv_files(folder_path):
    # Get all CSV files in the folder matching the pattern
    csv_files = glob.glob(os.path.join(folder_path, "inflection_points_line_*.csv"))
    
    if not csv_files:
        print("No CSV files found in the specified folder.")
        return

    combined_data = []

    # Read each CSV file and append its data to the list
    for file in csv_files:
        df = pd.read_csv(file)
        combined_data.append(df)

    # Concatenate all dataframes
    combined_df = pd.concat(combined_data)

    # Sort by line_id and point_id
    combined_df.sort_values(by=["line_id", "point_id"], inplace=True)

    # Save the combined dataframe to a new CSV file
    output_file = os.path.join(folder_path, "inflection_points_all.csv")
    combined_df.to_csv(output_file, index=False)

    print(f"Combined CSV file saved as: {output_file}")

if __name__ == "__main__":
    folder_path = input("Enter the folder path containing the CSV files: ").strip()
    if os.path.isdir(folder_path):
        combine_csv_files(folder_path)
    else:
        print("Invalid folder path. Please try again.")