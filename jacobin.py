# %%
import os
import polars as pl
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, log_loss
import tensorflow as tf

#############################################
#       DATABASE CONNECTION AND QUERY
#############################################
# %%
# Database connection parameters - adjust as needed
username = "postgres"
password = "annejames"
host = "localhost"
port = 5432
database = "open_payments"

engine = create_engine(f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}")

# Example query fetching data from 2024 dataset:
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
WHERE rn <= 1500000;
"""

with engine.connect() as connection:
    result = connection.execute(query)
    columns = result.keys()
    data = result.fetchall()

df = pl.DataFrame({col: [row[i] for row in data] for i, col in enumerate(columns)})

# Drop nulls
df = df.drop_nulls()
# %%
#############################################
#       PREPROCESSING
#############################################
categorical_cols = [
    "covered_recipient_type", 
    "covered_recipient_specialty_1", 
    "submitting_applicable_manufacturer_or_applicable_gpo_name", 
    "indicate_drug_or_biological_or_device_or_medical_supply_1", 
    "product_category_or_therapeutic_area_1"
]

df = df.with_columns([pl.col(col).cast(pl.Utf8) for col in categorical_cols])
pdf = df.to_pandas()

# Classification target
target_clf = "indicate_drug_or_biological_or_device_or_medical_supply_1"

y_clf = pdf[target_clf].values
pdf = pdf.drop(columns=["total_amount_of_payment_usdollars", target_clf])

# Identify numeric columns
numeric_cols = ["number_of_payments_included_in_total_amount", "program_year"]

# Scale numeric columns
scaler = StandardScaler()
if numeric_cols:
    pdf[numeric_cols] = scaler.fit_transform(pdf[numeric_cols])

# Encode remaining categorical columns (except the classification target which is already dropped)
remaining_cats = [c for c in categorical_cols if c != target_clf]
pdf = pd.get_dummies(pdf, columns=remaining_cats, drop_first=True)

# Encode classification target
unique_types = np.unique(y_clf)
type_to_idx = {t: i for i, t in enumerate(unique_types)}
y_clf_idx = np.array([type_to_idx[val] for val in y_clf], dtype=np.int64)

X = pdf.values.astype(np.float32)

# Split into train/validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_clf_idx, test_size=0.2, random_state=42)
# %%
#############################################
#       TENSORFLOW MODEL DEFINITION
#############################################
num_features = X_train.shape[1]
num_classes = len(unique_types)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(num_features,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# %%
#############################################
#       TRAINING
#############################################
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=10, batch_size=64, verbose=1)

#############################################
#       EVALUATION
#############################################
val_preds = model.predict(X_val)
val_preds_classes = np.argmax(val_preds, axis=1)

acc = accuracy_score(y_val, val_preds_classes)
ll = log_loss(y_val, val_preds)
report = classification_report(y_val, val_preds_classes, target_names=unique_types, zero_division=0)

print("Validation Accuracy:", acc)
print("Log Loss:", ll)
print("Classification Report:\n", report)
# %%
#############################################
#       JACOBIAN / SENSITIVITY ANALYSIS
#############################################
# We'll pick one sample from the validation set and compute gradients
x_sample = X_val[0:1]  # shape (1, num_features)
x_sample_tf = tf.convert_to_tensor(x_sample)

with tf.GradientTape() as tape:
    tape.watch(x_sample_tf)
    preds = model(x_sample_tf)  # shape (1, num_classes)
    # We can choose a particular output neuron to analyze. 
    # For a full Jacobian, compute gradient w.r.t. each class. Let's pick all classes.
    # We'll sum up the absolute values or just take one class for demonstration.
    # For full Jacobian: we need gradients for each output w.r.t inputs.
    
    # Let's compute gradients of each class logit w.r.t input:
    # We need the logits before softmax for a more direct interpretation:
    # We'll redefine a small function to get logits by removing the last activation:
    
# A more direct approach: create a sub-model to get logits (the layer before softmax)
logit_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-1].output)
with tf.GradientTape() as tape:
    tape.watch(x_sample_tf)
    logits = logit_model(x_sample_tf)  # shape (1, num_classes)

# Compute gradients of all outputs w.r.t. inputs
grads = tape.gradient(logits, x_sample_tf)  # shape (1, num_features, ) for each output (this returns sum for all?)
# grads is shape (1, num_classes, num_features)? Actually, gradient is computed at once.
# By default, `tape.gradient` returns the gradient of sum of outputs w.r.t input.
# To get full Jacobian, we can loop over classes:
jacobian = []
for c in range(num_classes):
    with tf.GradientTape() as tape:
        tape.watch(x_sample_tf)
        # isolate logit for class c
        logit_c = logit_model(x_sample_tf)[:, c]
    grad_c = tape.gradient(logit_c, x_sample_tf)  # shape (1, num_features)
    jacobian.append(grad_c.numpy()[0])

jacobian = np.array(jacobian)  # shape (num_classes, num_features)

# Now `jacobian` is the matrix of partial derivatives: classes x features
feature_sensitivity = np.mean(np.abs(jacobian), axis=0)

print("Feature sensitivity (avg absolute gradient across classes):\n", feature_sensitivity)

# You can visualize or sort this sensitivity to understand which features are most influential.
feature_names = pdf.columns
sens_df = pd.DataFrame({"feature": feature_names, "sensitivity": feature_sensitivity})
sens_df = sens_df.sort_values("sensitivity", ascending=False)
print("Top Features by Sensitivity:\n", sens_df.head(10))