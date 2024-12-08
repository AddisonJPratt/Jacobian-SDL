# %%
import requests

# Define the API endpoint and query
api_endpoint = "https://openpaymentsdata.cms.gov/api/1/datastore/sql"
query = "[SELECT Change_Type][LIMIT 1000]"

# Send the GET request
response = requests.get(api_endpoint, params={'query': query})

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    # Process the data as needed
else:
    print(f"Error: {response.status_code}")
# %%


