# map_utils.py

import folium

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
        locations.append((lat, lon, description))

    if not locations:
        return None

    m = folium.Map(location=[locations[0][0], locations[0][1]], zoom_start=13)

    for lat, lon, description in locations:
        folium.Marker(
            location=[lat, lon],
            popup=description,
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)

    return m.get_root().render()
