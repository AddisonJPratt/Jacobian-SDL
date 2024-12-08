# %%
import polars as pl
import pandas as pd
from sqlalchemy import create_engine

# Database connection parameters
username = "postgres"
password = "annejames"
host = "localhost"
port = 5432
database = "open_payments"

# Create SQLAlchemy engine
engine = create_engine(f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}")

# %%
# Query the database with all columns aliased to lowercase
query = """
WITH numbered AS (
    SELECT 
        Covered_Recipient_Type AS covered_recipient_type, 
        Covered_Recipient_Specialty_1 AS covered_recipient_specialty_1, 
        Submitting_Applicable_Manufacturer_or_Applicable_GPO_Name AS submitting_applicable_manufacturer_or_applicable_gpo_name, 
        Indicate_Drug_or_Biological_or_Device_or_Medical_Supply_1 AS indicate_drug_or_biological_or_device_or_medical_supply_1, 
        Product_Category_or_Therapeutic_Area_1 AS product_category_or_therapeutic_area_1, 
        Program_Year AS program_year, 
        Total_Amount_of_Payment_USDollars AS total_amount_of_payment_usdollars, 
        Number_of_Payments_Included_in_Total_Amount AS number_of_payments_included_in_total_amount,
        ROW_NUMBER() OVER (ORDER BY (SELECT 1)) AS rn
    FROM payments_2024
)
SELECT *
FROM numbered
WHERE rn <= 1000000;
"""

with engine.connect() as connection:
    result = connection.execute(query)
    data = result.fetchall()
    columns = result.keys()

df = pl.DataFrame({col: [row[i] for row in data] for i, col in enumerate(columns)})

# %%
# Handle missing values
df = df.drop_nulls()

# Convert categorical columns to strings (UTF-8)
categorical_cols = [
    "covered_recipient_type", 
    "covered_recipient_specialty_1", 
    "submitting_applicable_manufacturer_or_applicable_gpo_name", 
    "indicate_drug_or_biological_or_device_or_medical_supply_1", 
    "product_category_or_therapeutic_area_1"
]

df = df.with_columns([pl.col(col).cast(pl.Utf8) for col in categorical_cols])

# Numeric columns
numeric_cols = ["total_amount_of_payment_usdollars", "number_of_payments_included_in_total_amount"]

# Convert to pandas for get_dummies
pdf = df.to_pandas()

# Scale numeric columns
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
pdf[numeric_cols] = scaler.fit_transform(pdf[numeric_cols])

# One-hot encode categorical variables
pdf = pd.get_dummies(pdf, columns=categorical_cols, drop_first=True)

# Ensure no NaNs remain after encoding/scaling
pdf = pdf.fillna(0)

# %%
# Separate features and target
target_col = "total_amount_of_payment_usdollars"
X = pdf.drop(target_col, axis=1)
y = pdf[target_col]

# Split into train/validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X.values, y.values, 
                                                  test_size=0.2, 
                                                  random_state=42)

import numpy as np
X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)
y_train = y_train.astype(np.float32)
y_val = y_val.astype(np.float32)

# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Convert to torch tensors
X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train)
X_val_t = torch.from_numpy(X_val)
y_val_t = torch.from_numpy(y_val)

# Create DataLoader
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define a simple neural network
class PaymentModel(nn.Module):
    def __init__(self, input_dim):
        super(PaymentModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)
        return x

model = PaymentModel(input_dim=X_train.shape[1])

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
# Training loop with validation
num_epochs = 20
model.train()
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        preds = model(batch_X).squeeze()
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()

    # Validation loss
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t).squeeze()
        val_loss = criterion(val_preds, y_val_t)
    model.train()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# %%
# Compute the Jacobian of the model output w.r.t. the inputs for a given sample
model.eval()

from torch.autograd.functional import jacobian

# Select a single sample from validation set for demonstration
x_sample = X_val_t[0].unsqueeze(0)  # shape: (1, input_dim)
x_sample.requires_grad_(True)

def model_forward(inp):
    return model(inp)

J = jacobian(model_forward, x_sample)
print("Jacobian shape:", J.shape)
jacobian_vector = J[0,0,:]  # Extract the vector of partial derivatives
print("Jacobian for the selected sample:\n", jacobian_vector)



# %%
test_query = """
WITH numbered AS (
    SELECT 
        Covered_Recipient_Type AS covered_recipient_type, 
        Covered_Recipient_Specialty_1 AS covered_recipient_specialty_1, 
        Submitting_Applicable_Manufacturer_or_Applicable_GPO_Name AS submitting_applicable_manufacturer_or_applicable_gpo_name, 
        Indicate_Drug_or_Biological_or_Device_or_Medical_Supply_1 AS indicate_drug_or_biological_or_device_or_medical_supply_1, 
        Product_Category_or_Therapeutic_Area_1 AS product_category_or_therapeutic_area_1, 
        Program_Year AS program_year, 
        Total_Amount_of_Payment_USDollars AS total_amount_of_payment_usdollars, 
        Number_of_Payments_Included_in_Total_Amount AS number_of_payments_included_in_total_amount,
        ROW_NUMBER() OVER (ORDER BY (SELECT 1)) AS rn
    FROM payments_2024
)
SELECT *
FROM numbered
WHERE rn > 100000
AND rn <= 150000;
"""

with engine.connect() as connection:
    test_result = connection.execute(test_query)
    test_data = test_result.fetchall()
    test_columns = test_result.keys()

test_df = pl.DataFrame({col: [row[i] for row in test_data] for i, col in enumerate(test_columns)})

# 1. Drop nulls (same as training)
test_df = test_df.drop_nulls()

# 2. Cast categorical columns to UTF8 (same as training)
test_df = test_df.with_columns([pl.col(col).cast(pl.Utf8) for col in categorical_cols])

# Convert to pandas for further processing
test_pdf = test_df.to_pandas()

# Extract target once
y_test = test_pdf[target_col].values.astype(np.float32)
test_pdf = test_pdf.drop(target_col, axis=1)

# Remove target_col from numeric_cols if present
numeric_cols = [col for col in numeric_cols if col != target_col]

# Now scale only the remaining numeric columns
test_pdf[numeric_cols] = scaler.transform(test_pdf[numeric_cols])

# One-hot encode categorical variables
test_pdf = pd.get_dummies(test_pdf, columns=categorical_cols, drop_first=True)

# Ensure test_pdf has the same columns as X
missing_cols = set(X.columns) - set(test_pdf.columns)
for c in missing_cols:
    test_pdf[c] = 0

extra_cols = set(test_pdf.columns) - set(X.columns)
if extra_cols:
    test_pdf = test_pdf.drop(columns=extra_cols)

test_pdf = test_pdf[X.columns]

X_test = test_pdf.values.astype(np.float32)
X_test_t = torch.from_numpy(X_test)

model.eval()
with torch.no_grad():
    test_preds = model(X_test_t).squeeze().numpy()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test, test_preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, test_preds)
r2 = r2_score(y_test, test_preds)

print("Test MSE:", mse)
print("Test RMSE:", rmse)
print("Test MAE:", mae)
print("Test R²:", r2)
# %%
