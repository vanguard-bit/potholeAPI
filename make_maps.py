from firebase_admin import firestore
import folium # Import folium

def get_ward_details_by_id(db, ward_id):
    """
    Fetches the details of a ward from Firestore given its document ID.

    Args:
        ward_id (str): The document ID of the ward in Firestore.

    Returns:
        dict or None: The ward data as a dictionary if found, None otherwise.
    """
    try:
        ward_ref = db.collection('wards').document(ward_id)  # Get a reference to the specific document
        ward_doc = ward_ref.get()  # Fetch the document

        if ward_doc.exists:
            return ward_doc.to_dict()  # Convert the document data to a dictionary
        else:
            print(f"Ward with ID '{ward_id}' not found.")
            return None
    except Exception as e:
        print(f"Error fetching ward details: {e}")
        return None

def generate_category_map(db, category_name):
    """
    Fetch issues of a category from Firebase and generate a Folium map.
    
    Args:
        db: Firebase database connection.
        category_name (str): Category to filter issues.
    
    Returns:
        str or None: HTML string of the map, or None if no issues found.
    """
    issues_ref = db.collection('issues')
    issues = issues_ref.where('category', '==', category_name).stream()
    locations = []
    for issue in issues:
        issue_data = issue.to_dict()
        lat = issue_data['latitude']
        lon = issue_data['longitude']
        description = issue_data.get('description', '')
        item = [lat, lon, description]
        ward_details = []
        if "ward_id" in issue_data and issue_data['ward_id']:
            ward_id = issue_data['ward_id'] #changed
            ward_data = get_ward_details_by_id(db, ward_id) #changed
            if ward_data: # added this check
                ward_details.append("Assembly Name: " + ward_data.get("assembly_constituency_name_en","N/A")) # added .get and default value
                ward_details.append("Parliamentary Name: " + ward_data.get("parliamentary_constituency_name_en","N/A")) # added .get and default value
                ward_details.append("Ward Name: " + ward_data.get("proposed_ward_name_en","N/A")) # added .get and default value
        locations.append(item + ward_details)

    if not locations:
        return None

    m = folium.Map(location=[locations[0][0], locations[0][1]], zoom_start=13)

    for lat, lon, description, *ward_details in locations: #changed
        popup_text = description + "<br>" + "<br>".join(ward_details) if ward_details else description #changed
        folium.Marker(
            location=[lat, lon],
            popup=popup_text,
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)

    return m.get_root().render()
