from firebase_admin import db  # For Realtime Database (if you used that)
from firebase_admin import firestore

def is_point_in_polygon(point, polygon_coords):
    """
    Determines if a point (lon, lat) is inside a polygon (list of flat lon, lat values)
    using the Winding Number Algorithm for higher accuracy.
    """
    x, y = point
    winding_number = 0
    n = len(polygon_coords) // 2

    for i in range(n):
        j = (i + 1) % n
        p1x, p1y = polygon_coords[i * 2], polygon_coords[i * 2 + 1]
        p2x, p2y = polygon_coords[j * 2], polygon_coords[j * 2 + 1]

        if p1y <= y:
            if p2y > y and (p2y - p1y) * (x - p1x) - (p2x - p1x) * (y - p1y) > 0:
                winding_number += 1
        else:
            if p2y <= y and (p2y - p1y) * (x - p1x) - (p2x - p1x) * (y - p1y) < 0:
                winding_number -= 1

    # print(winding_number)
    return winding_number != 0

def search_ward_by_coordinates_firestore(longitude, latitude, db=None):
    """Searches the Firestore database for the ward containing the given coordinates.
    Returns both the ward data and the document ID.

    Args:
        longitude (float): The longitude of the point.
        latitude (float): The latitude of the point.

    Returns:
        tuple (dict, str) or None: A tuple containing the ward data and document ID if found,

                                     otherwise None.  Returns (None, None) if not found.
    """

    try:
        if db != None:
            wards_collection = db.collection('wards')
        else:
            wards_collection = firestore.client().collection('wards')
        wards = wards_collection.get()

        for ward_doc in wards:
            ward_data = ward_doc.to_dict()
            coordinates = ward_data.get('coordinates')
            if coordinates:
                if is_point_in_polygon((longitude, latitude), coordinates):
                    return ward_data, ward_doc.id  # Return both data and doc ID
        return None, None  # Return None, None if not found
    except Exception as e:
        print(f"Error searching Firestore: {e}")
        return None, None  # Return None, None on erro

# Example usage:
def get_coordinates_input():
    """Gets latitude and longitude input from the user."""
    try:
        latitude = float(input("Enter latitude: "))
        longitude = float(input("Enter longitude: "))
        return longitude, latitude
    except ValueError:
        print("Invalid input. Please enter numeric values for latitude and longitude.")
        return None

if __name__ == "__main__":
    # Initialize Firebase Admin SDK (if you haven't already)
    import firebase_admin
    from firebase_admin import credentials

    cred = credentials.Certificate("serviceAccountKey.json")  # Replace with your key path
    try:
        firebase_admin.initialize_app(cred)
    except ValueError as e:
        print(f"Firebase app already initialized: {e}")

    search_coords = get_coordinates_input()

    if search_coords:
        longitude, latitude = search_coords
        found_ward = search_ward_by_coordinates_firestore(longitude, latitude)

        if found_ward:
            print("\n--- Found Ward Details ---")
            for key, value in found_ward.items():
                print(f"{key}: {value}")
        else:
            print("No ward found for the given coordinates.")
