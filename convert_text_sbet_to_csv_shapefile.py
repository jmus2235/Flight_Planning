import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def txt_to_csv(input_path, output_csv_path):
    # Read space-delimited text file
    df = pd.read_csv(input_path, delim_whitespace=True)
    df.to_csv(output_csv_path, index=False)
    print(f"CSV file saved to: {output_csv_path}")
    return output_csv_path

def csv_to_shapefile(csv_path, output_shp_path):
    # Read CSV
    df = pd.read_csv(csv_path)
    # Create geometry from LATITUDE and LONGITUDE
    geometry = [Point(xy) for xy in zip(df['LONGITUDE'], df['LATITUDE'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    gdf.to_file(output_shp_path)
    print(f"Shapefile saved to: {output_shp_path}")

if __name__ == "__main__":
    input_path = input("Enter the full path to the input text file: ").strip()
    output_dir = input("Enter the output directory: ").strip()
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_csv = os.path.join(output_dir, base_name + ".csv")
    output_shp = os.path.join(output_dir, base_name + ".shp")

    txt_to_csv(input_path, output_csv)
    csv_to_shapefile(output_csv, output_shp)