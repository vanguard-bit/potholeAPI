import xml.etree.ElementTree as ET
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db  # For Realtime Database
from firebase_admin import firestore # For Firestore

def extract_placemarks(kml_file):
    tree = ET.parse(kml_file)
    root = tree.getroot()
    placemarks = root.findall('.//{*}Placemark')
    return placemarks

def print_placemark_details(placemark_element):
    """Prints the detailed information of a Placemark element."""
    print("--- Placemark ---")

    # Extract SimpleData from ExtendedData
    extended_data = placemark_element.find('.//{*}ExtendedData')
    if extended_data is not None:
        print("  --- ExtendedData ---")
        schema_data = extended_data.find('.//{*}SchemaData')
        if schema_data is not None:
            for simple_data in schema_data.findall('.//{*}SimpleData'):
                name = simple_data.get('name')
                value = simple_data.text.strip() if simple_data.text else None # Strip whitespace
                print(f"    {name}: {value}")

    # Extract Coordinates from MultiGeometry/Polygon
    multi_geometry = placemark_element.find('.//{*}MultiGeometry')
    if multi_geometry is not None:
        polygon = multi_geometry.find('.//{*}Polygon')
        if polygon is not None:
            outer_boundary = polygon.find('.//{*}outerBoundaryIs')
            if outer_boundary is not None:
                linear_ring = outer_boundary.find('.//{*}LinearRing')
                if linear_ring is not None:
                    coordinates_element = linear_ring.find('.//{*}coordinates')
                    if coordinates_element is not None:
                        coordinates = coordinates_element.text.strip()
                        print("  --- Coordinates ---")
                        print(f"    {coordinates}")

# Initialize Firebase Admin SDK (replace with your credentials)
cred = credentials.Certificate("serviceAccountKey.json")
try:
    firebase_admin.initialize_app(cred)
except ValueError as e:
    print(f"Firebase app already initialized: {e}")
# firebase_admin.initialize_app(cred, {
#     'databaseURL': 'YOUR_DATABASE_URL'  # Replace with your Firebase Realtime Database URL
#     # For Firestore, no databaseURL is needed here
# })

# Get a reference to the database (Realtime Database)
# ref = db.reference('wards')

# Or for Firestore:
db = firestore.client()
wards_collection = db.collection('wards')

def store_placemark_data(placemark_element):
    """Extracts data from a Placemark element and stores it in Firestore.

    Args:
        placemark_element (ET.Element): A Placemark Element object.
    """
    data = {}
    extended_data = placemark_element.find('.//{*}ExtendedData')
    if extended_data is not None:
        schema_data = extended_data.find('.//{*}SchemaData')
        if schema_data is not None:
            for simple_data in schema_data.findall('.//{*}SimpleData'):
                data[simple_data.get('name')] = simple_data.text.strip() if simple_data.text else None

    polygon = placemark_element.find('.//{*}Polygon')
    if polygon is not None:
        coordinates_element = polygon.find('.//{*}coordinates')
        if coordinates_element is not None:
            coordinates_str = coordinates_element.text.strip()
            coordinates_list = [
                float(coord)
                for point in coordinates_str.split()
                for coord in point.split(',')[:2]
            ]
            data['coordinates'] = coordinates_list

    # Store in Firestore
    if data.get('id'):
        wards_collection.document(data['id']).set(data)
        print(f"Stored data for ward ID: {data['id']}")

# Process each placemark and store its data

# Example usage:
kml_file = '/home/loki/Downloads/bbmp_final_new_wards.kml'
placemark_elements = extract_placemarks(kml_file)
#
# for placemark in placemark_elements:
#     print_placemark_details(placemark)
# for placemark in placemark_elements:
    # store_placemark_data(placemark)
print(len(placemark_elements))
print("Finished storing data in Firebase.")
