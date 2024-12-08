# %%
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Define the path to your preprocessed data
data_path = 'preprocessed_data.csv'  # Update this path if necessary

# Load the preprocessed dataset
preprocessed_dataset = pd.read_csv(data_path)
# Exclude 'Subject' and 'Activity' columns
feature_columns = preprocessed_dataset.columns[1:-1]
target_column = preprocessed_dataset.columns[-1]

# Define your target variable and features
X = preprocessed_dataset[feature_columns]
y = preprocessed_dataset[target_column]

# Split the data (e.g., 80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Combine back into DataFrames for the Dataset class
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train_df['Activity'] = le.fit_transform(train_df['Activity'])
test_df['Activity'] = le.transform(test_df['Activity'])

print(f"\nTraining Samples: {len(train_df)}")
print(f"Testing Samples: {len(test_df)}")

# %%
import torch
from torch.utils.data import Dataset

class HARRawDatasetCNN(Dataset):
    def __init__(self, dataframe, num_channels, sequence_length):
        """
        Args:
            dataframe (pd.DataFrame): The preprocessed dataset without 'Subject'.
            num_channels (int): Number of sensor channels (e.g., 3).
            sequence_length (int): Number of time steps per channel (e.g., 187).
        """
        self.X = torch.tensor(dataframe[feature_columns].values, dtype=torch.float32)
        self.y = torch.tensor(dataframe[target_column].values, dtype=torch.long)
        
        # Reshape X to [batch_size, channels, sequence_length]
        self.X = self.X.view(-1, num_channels, sequence_length)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# %%
from torch.utils.data import DataLoader

# Define parameters based on Option 1
num_channels = 3
sequence_length = 561 // num_channels  # 561 / 3 = 187

# Instantiate the Dataset
train_dataset_cnn = HARRawDatasetCNN(train_df, num_channels, sequence_length)
test_dataset_cnn = HARRawDatasetCNN(test_df, num_channels, sequence_length)

# Define batch size
batch_size = 64

# Create DataLoader instances
train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=batch_size, shuffle=True)
test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=batch_size, shuffle=False)

# Verify the shapes
for X_batch, y_batch in train_loader_cnn:
    print(f"Batch X shape: {X_batch.shape}")  # Expected: [batch_size, num_channels, sequence_length]
    print(f"Batch y shape: {y_batch.shape}")  # Expected: [batch_size]
    break
# %%
import torch.nn as nn

class HAR_CNN(nn.Module):
    def __init__(self, num_channels, num_classes, sequence_length):
        super(HAR_CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # Reduces sequence_length to 93
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # Reduces sequence_length to 46
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # Reduces sequence_length to 23
        )
        self.fc1 = nn.Linear(256 * (sequence_length // 8), 512)  # 3 MaxPool layers reduce sequence_length by 8
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        out = self.layer1(x)  # [batch_size, 64, 93]
        out = self.layer2(out)  # [batch_size, 128, 46]
        out = self.layer3(out)  # [batch_size, 256, 23]
        out = out.view(out.size(0), -1)  # Flatten to [batch_size, 256*23]
        out = self.fc1(out)  # [batch_size, 512]
        out = self.dropout(out)
        out = self.fc2(out)  # [batch_size, num_classes]
        return out
# %%
import torch.optim as optim

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# Initialize the model
num_classes = len(preprocessed_dataset['Activity'].unique())
model_cnn = HAR_CNN(num_channels=num_channels, num_classes=num_classes, sequence_length=sequence_length)
model_cnn.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_cnn.parameters(), lr=0.001, weight_decay=1e-4)  # weight_decay for regularization
# %%
# Define number of epochs
num_epochs = 30  # Adjust based on convergence and computational resources

from sklearn.metrics import accuracy_score

best_accuracy = 0.0
patience = 5
trigger_times = 0

for epoch in range(num_epochs):
    model_cnn.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader_cnn:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # Forward pass
        outputs = model_cnn(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader_cnn)
    
    # Evaluate on test set
    model_cnn.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader_cnn:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model_cnn(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    
    # Check for improvement
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        trigger_times = 0
        # Save the best model
        torch.save(model_cnn.state_dict(), 'har_cnn_baseline_model.pth')
        print("Best model saved.")
    else:
        trigger_times += 1
        print(f'No improvement in accuracy for {trigger_times} epochs.')
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

# %%
# Save the trained model
torch.save(model_cnn.state_dict(), 'har_cnn_baseline_model.pth')
print("\nBaseline CNN model saved as 'har_cnn_baseline_model.pth'")
# %%
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_cnn(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    
    return accuracy, report, cm
# %%
# Evaluate baseline CNN model
baseline_cnn_accuracy, baseline_cnn_report, baseline_cnn_cm = evaluate_cnn(model_cnn, test_loader_cnn, device)

print(f'\nBaseline CNN Model Accuracy: {baseline_cnn_accuracy:.2f}%')
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=le.classes_))
# %%
import torch
from torch.utils.data import Dataset, DataLoader

class TestDataset(Dataset):
    def __init__(self, dataframe, num_channels, sequence_length):
        self.X = torch.tensor(dataframe.values, dtype=torch.float32)
        # Ensure features fit into num_channels * sequence_length
        total_features = num_channels * sequence_length
        if self.X.shape[1] != total_features:
            self.X = self.X[:, :total_features]
        # Reshape to [batch_size, num_channels, sequence_length]
        self.X = self.X.view(-1, num_channels, sequence_length)
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx]
    
# Define parameters
num_channels = 3
sequence_length = 561 // num_channels  # Ensure it matches your model definition

test_data = pd.read_csv("test_data.csv")
# Instantiate the dataset
test_dataset = TestDataset(test_data, num_channels, sequence_length)

# Create a DataLoader
batch_size = 64  # Adjust batch size as needed
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# %%
# Define the model architecture
model_cnn = HAR_CNN(num_channels=num_channels, num_classes=num_classes, sequence_length=sequence_length)
model_cnn.load_state_dict(torch.load('har_cnn_baseline_model.pth'))
model_cnn.to(device)
model_cnn.eval()  # Set to evaluation mode
# %%
all_preds = []

with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model_cnn(X_batch)  # Forward pass
        _, predicted = torch.max(outputs, 1)  # Get class predictions
        all_preds.extend(predicted.cpu().numpy())  # Move predictions to CPU and store

# Convert predictions to a DataFrame
predictions = pd.DataFrame(all_preds, columns=["Predicted_Activity"])
print(predictions.head())
# %%
