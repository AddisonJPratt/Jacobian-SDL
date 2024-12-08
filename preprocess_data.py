# %%
import os
import urllib.request
import zipfile

# Define URLs and paths
dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
dataset_path = "UCI_HAR_Dataset.zip"
extract_path = "UCI_HAR_Dataset"

# Download the dataset
if not os.path.exists(dataset_path):
    print("Downloading UCI HAR Dataset...")
    urllib.request.urlretrieve(dataset_url, dataset_path)
    print("Download complete.")

# Extract the dataset
if not os.path.exists(extract_path):
    print("Extracting UCI HAR Dataset...")
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")
# %%
import pandas as pd
import numpy as np

# Paths to data
base_path = os.path.join(extract_path, "UCI HAR Dataset")
train_path = os.path.join(base_path, "train")
test_path = os.path.join(base_path, "test")

# Load feature names
feature_names = pd.read_csv(os.path.join(base_path, "features.txt"), 
                            sep='\s+', header=None, names=["index", "feature"])
feature_names = feature_names['feature'].tolist()

# Load activity labels
activity_labels = pd.read_csv(os.path.join(base_path, "activity_labels.txt"), 
                              sep='\s+', header=None, names=["index", "activity"])
activity_labels_dict = dict(zip(activity_labels.index + 1, activity_labels['activity']))
# %%
# Load training data
X_train = pd.read_csv(os.path.join(train_path, "X_train.txt"), sep='\s+', header=None, names=feature_names)
y_train = pd.read_csv(os.path.join(train_path, "y_train.txt"), sep='\s+', header=None, names=["Activity"])
subject_train = pd.read_csv(os.path.join(train_path, "subject_train.txt"), sep='\s+', header=None, names=["Subject"])


# %%
# Check for duplicate feature names
from collections import Counter

feature_counts = Counter(feature_names)
duplicates = [item for item, count in feature_counts.items() if count > 1]

if duplicates:
    print(f"Duplicate feature names found: {duplicates}")
else:
    print("No duplicate feature names found.")

# Function to make feature names unique
def make_unique(names):
    from collections import defaultdict
    name_counts = defaultdict(int)
    unique_names = []
    for name in names:
        if name_counts[name]:
            unique_names.append(f"{name}_{name_counts[name]}")
        else:
            unique_names.append(name)
        name_counts[name] += 1
    return unique_names

# Apply the function
unique_feature_names = make_unique(feature_names)

# Verify uniqueness
if len(unique_feature_names) == len(set(unique_feature_names)):
    print("All feature names are now unique.")
else:
    print("There are still duplicate feature names.")
# Load training data with unique feature names
X_train = pd.read_csv(
    os.path.join(train_path, "X_train.txt"), 
    sep='\s+', 
    header=None, 
    names=unique_feature_names
)

y_train = pd.read_csv(
    os.path.join(train_path, "y_train.txt"), 
    sep='\s+', 
    header=None, 
    names=["Activity"]
)

subject_train = pd.read_csv(
    os.path.join(train_path, "subject_train.txt"), 
    sep='\s+', 
    header=None, 
    names=["Subject"]
)

# Load testing data similarly
X_test = pd.read_csv(
    os.path.join(test_path, "X_test.txt"), 
    sep='\s+', 
    header=None, 
    names=unique_feature_names
)

y_test = pd.read_csv(
    os.path.join(test_path, "y_test.txt"), 
    sep='\s+', 
    header=None, 
    names=["Activity"]
)

subject_test = pd.read_csv(
    os.path.join(test_path, "subject_test.txt"), 
    sep='\s+', 
    header=None, 
    names=["Subject"]
)

# Combine train and test data
data = pd.concat([X_train, y_train], axis=1)
train_data = pd.concat([X_test], axis=1)
# %%

data.to_csv("preprocessed_data.csv")

# %%

train_data.to_csv("test_data.csv")
# %%
# %%
import requests

# Define the API endpoint and query
api_endpoint = "https://openpaymentsdata.cms.gov/api/1/datastore/sql"
query = "[SELECT Change_Type, Covered_Recipient_Type, Teaching_Hospital_Name, Covered_Recipient_First_Name, Covered_Recipient_Last_Name, Recipient_City, Recipient_State, Total_Amount_of_Payment_USDollars, Date_of_Payment, Nature_of_Payment_or_Transfer_of_Value FROM fdc3c773-018a-412c-8a81-d7b8a13a037b][LIMIT 1000]"
# %%
# Send the GET request
response = requests.get(api_endpoint, params={'query': query})

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    # Process the data as needed
else:
    print(f"Error: {response.status_code}")


# %%
import pandas as pd
open_payments = pd.read_csv("open_payments.csv").head()
# %%
print(list(open_payments.columns))
# %%
