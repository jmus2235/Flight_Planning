#!/usr/bin/env python3
"""
KML Converter: ArcGIS Pro to Proprietary Software Format (Point Output)
Converts point-based KML from ArcGIS Pro to match target format structure while preserving points
Anaconda Command line: python kml_format_converter.py input_arcgis.kml target_format.kml output.kml
"""


import xml.etree.ElementTree as ET
import argparse
import sys
from pathlib import Path
import re

class KMLConverter:
    def __init__(self):
        self.kml_ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        
    def parse_arcgis_kml(self, input_file):
        """Parse the ArcGIS Pro generated KML and extract point data"""
        try:
            # Parse with namespace handling
            tree = ET.parse(input_file)
            root = tree.getroot()
            
            # Get actual namespaces from the document
            namespaces = {}
            # Extract namespace from root element
            if root.tag.startswith('{'):
                ns_uri = root.tag[1:root.tag.find('}')]
                namespaces['kml'] = ns_uri
            else:
                # Try common KML namespace
                namespaces['kml'] = 'http://www.opengis.net/kml/2.2'
            
            points = []
            
            # Try with detected namespaces first
            placemarks = root.findall('.//kml:Placemark', namespaces)
            
            # If that doesn't work, try without namespaces
            if not placemarks:
                placemarks = root.findall('.//Placemark')
            
            print(f"Found {len(placemarks)} placemarks")
            
            for placemark in placemarks:
                # Try with namespace first
                name_elem = placemark.find('kml:name', namespaces)
                if name_elem is None:
                    name_elem = placemark.find('name')
                
                point_elem = placemark.find('.//kml:Point/kml:coordinates', namespaces)
                if point_elem is None:
                    point_elem = placemark.find('.//Point/coordinates')
                
                if name_elem is not None and point_elem is not None:
                    name = name_elem.text
                    coords = point_elem.text.strip()
                    
                    print(f"Processing point: {name} with coordinates: {coords}")
                    
                    # Parse coordinates (lon,lat,alt)
                    coord_parts = coords.split(',')
                    if len(coord_parts) >= 2:
                        lon = float(coord_parts[0])
                        lat = float(coord_parts[1])
                        alt = float(coord_parts[2]) if len(coord_parts) > 2 else 0
                        
                        # Try to extract numeric order from name
                        order = 0
                        if name and name.isdigit():
                            order = int(name)
                        else:
                            # Try to extract number from name
                            match = re.search(r'\d+', name or '')
                            if match:
                                order = int(match.group())
                        
                        points.append({
                            'name': name,
                            'lon': lon,
                            'lat': lat,
                            'alt': alt,
                            'order': order
                        })
            
            # Sort points by their order
            points.sort(key=lambda x: x['order'])
            print(f"Successfully parsed {len(points)} points")
            return points
            
        except ET.ParseError as e:
            print(f"XML Parse Error: {e}")
            print("Trying alternative parsing method...")
            return self._parse_kml_alternative(input_file)
        except Exception as e:
            print(f"Error parsing ArcGIS KML: {e}")
            return []

    def _parse_kml_alternative(self, input_file):
        """Alternative parsing method using string manipulation"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            points = []
            
            # Use regex to find placemarks
            placemark_pattern = r'<Placemark[^>]*id="([^"]*)"[^>]*>(.*?)</Placemark>'
            placemarks = re.findall(placemark_pattern, content, re.DOTALL)
            
            for placemark_id, placemark_content in placemarks:
                # Extract name
                name_match = re.search(r'<name>([^<]*)</name>', placemark_content)
                name = name_match.group(1) if name_match else placemark_id
                
                # Extract coordinates
                coord_match = re.search(r'<coordinates>([^<]*)</coordinates>', placemark_content)
                if coord_match:
                    coords = coord_match.group(1).strip()
                    coord_parts = coords.split(',')
                    
                    if len(coord_parts) >= 2:
                        try:
                            lon = float(coord_parts[0])
                            lat = float(coord_parts[1])
                            alt = float(coord_parts[2]) if len(coord_parts) > 2 else 0
                            
                            # Try to extract numeric order from name
                            order = 0
                            if name and name.isdigit():
                                order = int(name)
                            else:
                                # Try to extract number from name
                                order_match = re.search(r'\d+', name or '')
                                if order_match:
                                    order = int(order_match.group())
                            
                            points.append({
                                'name': name,
                                'lon': lon,
                                'lat': lat,
                                'alt': alt,
                                'order': order
                            })
                            
                            print(f"Alternative parsing found point: {name} at {lon},{lat},{alt}")
                            
                        except ValueError as e:
                            print(f"Error parsing coordinates for {name}: {e}")
                            continue
            
            points.sort(key=lambda x: x['order'])
            print(f"Alternative parsing found {len(points)} points")
            return points
            
        except Exception as e:
            print(f"Alternative parsing failed: {e}")
            return []

    def analyze_target_format(self, target_file):
        """Analyze the target KML format to understand its structure"""
        try:
            tree = ET.parse(target_file)
            root = tree.getroot()
            
            # Get actual namespaces from the document
            namespaces = {}
            if root.tag.startswith('{'):
                ns_uri = root.tag[1:root.tag.find('}')]
                namespaces['kml'] = ns_uri
            else:
                namespaces['kml'] = 'http://www.opengis.net/kml/2.2'
            
            format_info = {
                'has_folders': False,
                'folder_naming': None,
                'uses_snippets': False,
                'uses_descriptions': False,
                'description_format': None,
                'styles': {},
                'main_folder_name': 'Points'
            }
            
            # Check for folders
            folders = root.findall('.//kml:Folder', namespaces)
            if not folders:
                folders = root.findall('.//Folder')
                
            if folders:
                format_info['has_folders'] = True
                
                # Get main folder name
                main_folder = folders[0] if folders else None
                if main_folder is not None:
                    main_name_elem = main_folder.find('kml:name', namespaces)
                    if main_name_elem is None:
                        main_name_elem = main_folder.find('name')
                    if main_name_elem is not None:
                        format_info['main_folder_name'] = main_name_elem.text
            
            # Check for snippets
            snippets = root.findall('.//kml:snippet', namespaces)
            if not snippets:
                snippets = root.findall('.//snippet')
            if snippets:
                format_info['uses_snippets'] = True
            
            # Check for descriptions and their format
            descriptions = root.findall('.//kml:description', namespaces)
            if not descriptions:
                descriptions = root.findall('.//description')
                
            if descriptions:
                format_info['uses_descriptions'] = True
                # Analyze description format from first non-empty description
                for desc in descriptions:
                    if desc.text and desc.text.strip():
                        # Check if it's HTML formatted
                        if '<html' in desc.text.lower() or '<table' in desc.text.lower():
                            format_info['description_format'] = 'html_table'
                        else:
                            format_info['description_format'] = 'simple'
                        break
            
            # Extract styles
            styles = root.findall('.//kml:Style', namespaces)
            if not styles:
                styles = root.findall('.//Style')
                
            for style in styles:
                style_id = style.get('id')
                if style_id:
                    # Store the entire style element
                    format_info['styles'][style_id] = ET.tostring(style, encoding='unicode')
            
            return format_info
            
        except Exception as e:
            print(f"Error analyzing target KML: {e}")
            return {
                'has_folders': True,
                'uses_snippets': True,
                'uses_descriptions': True,
                'description_format': 'html_table',
                'styles': {},
                'main_folder_name': 'Points'
            }
    
    def create_html_table_description(self, point_name):
        """Create HTML table description matching the target format"""
        return f'''<html xmlns:fo="http://www.w3.org/1999/XSL/Format" xmlns:msxsl="urn:schemas-microsoft-com:xslt">

<head>

<META http-equiv="Content-Type" content="text/html">

<meta http-equiv="content-type" content="text/html; charset=UTF-8">

</head>

<body style="margin:0px 0px 0px 0px;overflow:auto;background:#FFFFFF;">

<table style="font-family:Arial,Verdana,Times;font-size:12px;text-align:left;width:100%;border-collapse:collapse;padding:3px 3px 3px 3px">

<tr style="text-align:center;font-weight:bold;background:#9CBCE2">

<td>{point_name}</td>

</tr>

<tr>

<td>

<table style="font-family:Arial,Verdana,Times;font-size:12px;text-align:left;width:100%;border-spacing:0px; padding:3px 3px 3px 3px"></table>

</td>

</tr>

</table>

</body><script type="text/javascript">

				function changeImage(attElement, nameElement) {{

				document.getElementById('imageAttachment').src = attElement;

				document.getElementById('imageName').innerHTML = nameElement;}}

			</script></html>'''
    
    def generate_target_kml(self, points, format_info, output_file):
        """Generate KML in the target format with points"""
        # Create root KML element
        root = ET.Element('kml')
        root.set('xmlns', 'http://www.opengis.net/kml/2.2')
        root.set('xmlns:gx', 'http://www.google.com/kml/ext/2.2')
        root.set('xmlns:kml', 'http://www.opengis.net/kml/2.2')
        root.set('xmlns:atom', 'http://www.w3.org/2005/Atom')
        
        document = ET.SubElement(root, 'Document')
        document.set('id', format_info.get('main_folder_name', 'ConvertedPoints'))
        document.set('xsi:schemaLocation', 
                    'http://www.opengis.net/kml/2.2 http://schemas.opengis.net/kml/2.2.0/ogckml22.xsd http://www.google.com/kml/ext/2.2 http://code.google.com/apis/kml/schema/kml22gx.xsd')
        
        # Add document name and snippet
        doc_name = ET.SubElement(document, 'name')
        doc_name.text = format_info.get('main_folder_name', 'ConvertedPoints')
        
        if format_info.get('uses_snippets', False):
            doc_snippet = ET.SubElement(document, 'snippet')
            doc_snippet.text = ''
        
        # Add atom link if present in target format
        atom_link = ET.SubElement(document, '{http://www.w3.org/2005/Atom}link')
        atom_link.set('rel', 'app')
        atom_link.set('href', 'https://www.google.com/earth/about/versions/#earth-pro')
        atom_link.set('title', 'Google Earth Pro 7.3.6.10201')
        
        # Add styles
        if format_info.get('styles'):
            for style_id, style_xml in format_info['styles'].items():
                try:
                    style_elem = ET.fromstring(style_xml)
                    document.append(style_elem)
                except ET.ParseError:
                    pass
        else:
            # Create default style if none found
            default_style = ET.SubElement(document, 'Style')
            default_style.set('id', 'IconStyle00')
            
            icon_style = ET.SubElement(default_style, 'IconStyle')
            icon = ET.SubElement(icon_style, 'Icon')
            href = ET.SubElement(icon, 'href')
            href.text = 'Layer0_Symbol_8733f4c0_0.png'
            
            label_style = ET.SubElement(default_style, 'LabelStyle')
            label_color = ET.SubElement(label_style, 'color')
            label_color.text = '00000000'
            label_scale = ET.SubElement(label_style, 'scale')
            label_scale.text = '0'
            
            poly_style = ET.SubElement(default_style, 'PolyStyle')
            poly_color = ET.SubElement(poly_style, 'color')
            poly_color.text = 'ff000000'
            poly_outline = ET.SubElement(poly_style, 'outline')
            poly_outline.text = '0'
        
        # Create main folder if target uses folders
        if format_info.get('has_folders', False):
            main_folder = ET.SubElement(document, 'Folder')
            main_folder.set('id', 'FeatureLayer0')
            folder_name = ET.SubElement(main_folder, 'name')
            folder_name.text = format_info.get('main_folder_name', 'ConvertedPoints')
            
            if format_info.get('uses_snippets', False):
                folder_snippet = ET.SubElement(main_folder, 'snippet')
                folder_snippet.text = ''
            
            parent_element = main_folder
        else:
            parent_element = document
        
        # Create point placemarks
        for i, point in enumerate(points):
            placemark = ET.SubElement(parent_element, 'Placemark')
            placemark.set('id', f'ID_{i:05d}')
            
            # Name
            name = ET.SubElement(placemark, 'name')
            name.text = point['name']
            
            # Snippet
            if format_info.get('uses_snippets', False):
                snippet = ET.SubElement(placemark, 'snippet')
                snippet.text = ''
            
            # Description
            if format_info.get('uses_descriptions', False):
                description = ET.SubElement(placemark, 'description')
                if format_info.get('description_format') == 'html_table':
                    description.text = self.create_html_table_description(point['name'])
                else:
                    description.text = f"Point {point['name']}"
            
            # Style reference
            style_url = ET.SubElement(placemark, 'styleUrl')
            # Use first available style ID or default
            available_styles = list(format_info.get('styles', {}).keys())
            if available_styles:
                style_url.text = f"#{available_styles[0]}"
            else:
                style_url.text = "#IconStyle00"
            
            # Point geometry
            point_elem = ET.SubElement(placemark, 'Point')
            coordinates = ET.SubElement(point_elem, 'coordinates')
            coordinates.text = f"{point['lon']},{point['lat']},{point['alt']}"
        
        # Write to file with proper formatting
        # First create the tree
        tree = ET.ElementTree(root)
        
        # Format the XML with proper indentation
        self._indent_xml(root)
        
        # Write with XML declaration
        with open(output_file, 'wb') as f:
            tree.write(f, encoding='utf-8', xml_declaration=True)
    
    def _indent_xml(self, elem, level=0):
        """Add proper indentation to XML elements"""
        i = "\n" + level * "\t"
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "\t"
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                self._indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
    
    def convert(self, input_file, target_file, output_file):
        """Main conversion function"""
        print(f"Converting {input_file} to format similar to {target_file}")
        
        # Parse input points
        points = self.parse_arcgis_kml(input_file)
        if not points:
            print("No points found in input file")
            return False
        
        print(f"Found {len(points)} points")
        
        # Analyze target format
        format_info = self.analyze_target_format(target_file)
        print(f"Target format analysis: {format_info}")
        
        # Generate output KML
        self.generate_target_kml(points, format_info, output_file)
        print(f"Conversion complete: {output_file}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Convert ArcGIS Pro KML points to proprietary software format')
    parser.add_argument('input_kml', help='Input KML file from ArcGIS Pro')
    parser.add_argument('target_kml', help='Target KML file (format reference)')
    parser.add_argument('output_kml', help='Output KML file')
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.input_kml).exists():
        print(f"Error: Input file {args.input_kml} does not exist")
        sys.exit(1)
    
    if not Path(args.target_kml).exists():
        print(f"Error: Target file {args.target_kml} does not exist")
        sys.exit(1)
    
    # Perform conversion
    converter = KMLConverter()
    success = converter.convert(args.input_kml, args.target_kml, args.output_kml)
    
    if success:
        print("Conversion completed successfully!")
    else:
        print("Conversion failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()