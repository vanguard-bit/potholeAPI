
import requests
import json

# API endpoint
url = "https://kgis.ksrsac.in:9000/genericwebservices/ws/getKGISAdminCodes2"

# Ask user for input
lat = input("Enter Latitude: ").strip()
lon = input("Enter Longitude: ").strip()

try:
    # Convert inputs to float to validate
    lat = float(lat)
    lon = float(lon)
except ValueError:
    print("Invalid latitude or longitude. Please enter numeric values.")
    exit(1)

# Create payload
payload =  [ { "ID": 198, "Gps_Lat": 13.03063015, "Gps_Lon": 77.61886905 }, { "ID": 200, "Gps_Lat": 13.03138907, "Gps_Lon": 77.61987634 } ] 
# Set headers
headers = {
    "Content-Type": "application/json"
}

# Make POST request
try:
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Raise an error for bad status codes
    print("Response JSON:")
    print(json.dumps(response.json(), indent=4))
except requests.exceptions.RequestException as e:
    print(f"Error occurred: {e}")
