import pandas as pd
import torch
import numpy as np
from torch import nn
from LSTM import LSTM


# Print found devices
devices = [d for d in range(torch.cuda.device_count())] 
device_names  = [torch.cuda.get_device_name(d) for d in devices]
print(device_names)

# Cuda device to use .to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load the data
data = pd.read_csv('test.csv')

# Convert the string representation of lists to actual lists
data['feature'] = data['feature'].apply(eval)
data['label'] = data['label'].apply(eval)

# Convert the features and labels to PyTorch tensors
feature_tensor = torch.tensor((np.array(data['feature'].tolist())), dtype=torch.float32).to(device)
label_tensor = torch.tensor((np.array(data['label'].tolist())), dtype=torch.float32).to(device)

# Initialize the LSTM model
model = LSTM(input_size=1, hidden_size=127, output_size=5)
# Move the model to the device
model.to(device)

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Number of epochs
epochs = 100000

# Train the model
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(feature_tensor.view(feature_tensor.shape[0], -1, 1))
    loss = criterion(output, label_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

# Switch to evaluation mode
model.eval()

# Make predictions
predictions = model(feature_tensor.view(feature_tensor.shape[0], -1, 1))

# Calculate the loss
loss = criterion(predictions, label_tensor)

print(f'Evaluation loss: {loss.item()}')

#accuracy
# Calculate the accuracy
def calculate_accuracy(predictions, labels):
    # Convert the predictions and labels to numpy arrays
    predictions =(predictions.detach().cpu().numpy())
    labels = (labels.cpu().numpy())

    # Round the predictions to the nearest integer
    rounded_predictions = np.round(predictions)

    # Calculate the number of correct predictions
    correct_predictions = np.sum(rounded_predictions == labels)

    # Calculate the accuracy
    accuracy = correct_predictions / len(labels) * 100

    return accuracy

accuracy = calculate_accuracy(predictions, label_tensor)
print(f'Accuracy: {accuracy}%')

# Compare the predictions to the actual labels
print('Predictions:\n', (predictions.detach().cpu().numpy()))
print('Actual labels:\n', (label_tensor.cpu().numpy()))

# Assuming 'model' is your trained model
print(model.state_dict().keys())
torch.save(model.state_dict(), 'LSTMTest.pth')