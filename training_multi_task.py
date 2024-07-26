import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
from torch.utils.data import Dataset, DataLoader
# from shapely.geometry import Polygon, Point

test_tr_list = np.load('tr_list.npy')
test_ta_list = np.load('ta_list.npy')
test_re_list = np.load('re_list.npy')

test_tr_value = np.load('tr_value.npy')
test_ta_value = np.load('ta_value.npy')
test_re_value = np.load('re_value.npy')
print("Arrays loaded successfully.")

tr_list = test_tr_list.tolist()
ta_list = test_ta_list.tolist()  
re_list = test_re_list.tolist()

tr_value = test_tr_value.tolist()
ta_value = test_ta_value.tolist()
re_value = test_re_value.tolist()

print("finish generating data")
print("start training")

class shapeDetector(nn.Module):
    def __init__(self):
        super(shapeDetector, self).__init__()
        
        # For convolutional layers
        self.conv1 = nn.Conv1d(2, 64, 1)
        self.a1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.a2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.a3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv1d(256, 1024, 1)
        
        # Fully-connected layers: add complexity
        self.fc1 = nn.Linear(1024, 1024)
        self.a4 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 1024)
        self.a5 = nn.ReLU(inplace=True)
        
        # Classification layer
        self.fc3_1 = nn.Linear(1024, 1024)
        self.a_class = nn.ReLU(inplace=True)
        self.fc3_2 = nn.Linear(1024, 3)  
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        # Regression layers for parameters
        self.fcta1 = nn.Linear(1024, 1024)
        self.a_ta1 = nn.ReLU(inplace=True)
        self.fcta2 = nn.Linear(1024, 1024)
        self.a_ta2 = nn.ReLU(inplace=True)
        self.fcta3 = nn.Linear(1024, 4)  
        self.a_ta3 = nn.Softplus()
        
        self.fcre1 = nn.Linear(1024, 1024)
        self.a_re1 = nn.ReLU(inplace=True)
        self.fcre2 = nn.Linear(1024, 1024)
        self.a_re2 = nn.ReLU(inplace=True)
        self.fcre3 = nn.Linear(1024, 3)  
        self.a_re3 = nn.Softplus()
        
        self.fctr1 = nn.Linear(1024, 1024)
        self.a_tr1 = nn.ReLU(inplace=True)
        self.fctr2 = nn.Linear(1024, 1024)
        self.a_tr2 = nn.ReLU(inplace=True)
        self.fctr3 = nn.Linear(1024, 4)  
        self.a_tr3 = nn.Softplus()
        
        # Dropout layers to be implemented between every two layers
        self.d1 = nn.Dropout(p=0.15)
        self.d2 = nn.Dropout(p=0.15)
        self.d3 = nn.Dropout(p=0.15)
        self.d4 = nn.Dropout(p=0.15)
        self.d5 = nn.Dropout(p=0.15)
        # For classification layer
        self.d_class = nn.Dropout(p=0.15)
        # For parameter layer:
        self.d_ta1 = nn.Dropout(p=0.15)
        self.d_ta2 = nn.Dropout(p=0.15)
        
        self.d_re1 = nn.Dropout(p=0.15)
        self.d_re2 = nn.Dropout(p=0.15)
        
        self.d_tr1 = nn.Dropout(p=0.15)
        self.d_tr2 = nn.Dropout(p=0.15)

    def forward(self, x):
        # Putting together convolutional layers and dropout layers
        x = self.d1(self.a1(self.conv1(x)))
        x = self.d2(self.a2(self.conv2(x)))
        x = self.d3(self.a3(self.conv3(x)))
        x = self.d4(self.conv4(x))
        # Maxpooling to "conclude" patterns
        x = nn.MaxPool1d(x.size(-1))(x)
        x_latent = nn.Flatten(1)(x)
        x_latent = self.d5(self.a4(self.fc1(x_latent)))
        x_latent = self.a5(self.fc2(x_latent))
        
        # For classification: 
        x_class = self.d_class(self.a_class(self.fc3_1(x_latent)))
        x_class = self.logsoftmax(self.fc3_2(x_class))
        
        # For regression:
        # For trapezoid:
        x_ta = self.d_ta1(self.a_ta1(self.fcta1(x_latent)))
        x_ta = self.d_ta2(self.a_ta2(self.fcta2(x_ta)))
        x_ta = self.a_ta3(self.fcta3(x_ta))
        # For rectangle:
        x_re = self.d_re1(self.a_re1(self.fcre1(x_latent)))
        x_re = self.d_re2(self.a_re2(self.fcre2(x_re)))
        x_re = self.a_re3(self.fcre3(x_re))
        # For triangle:
        x_tr = self.d_tr1(self.a_tr1(self.fctr1(x_latent)))
        x_tr = self.d_tr2(self.a_tr2(self.fctr2(x_tr)))
        x_tr = self.a_tr3(self.fctr3(x_tr))
        
        return x_class, x_ta, x_re, x_tr

# Create a combined dataset class
class CombinedDataset(Dataset):
    def __init__(self, tr_list, ta_list, re_list, tr_value, ta_value, re_value):
        self.data = tr_list + ta_list + re_list
        self.labels = [2] * len(tr_list) + [1] * len(ta_list) + [0] * len(re_list)
        self.values = tr_value + ta_value + re_value

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        point_cloud = np.array(self.data[idx], dtype=np.float32).T
        label = self.labels[idx]
        value = np.array(self.values[idx], dtype=np.float32)
        return torch.tensor(point_cloud), torch.tensor(label), torch.tensor(value)

# Create the dataset and dataloader
combined_dataset = CombinedDataset(tr_list, ta_list, re_list, tr_value, ta_value, re_value)
dataloader = DataLoader(combined_dataset, batch_size=16, shuffle=True)

# Initialize the model, loss functions, and optimizer
model = shapeDetector()
classification_loss = nn.CrossEntropyLoss()
regression_loss = nn.MSELoss()
optimizer = optimizers.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Training loop
for epoch in range(25):  # Number of epochs
    total_loss = 0
    total_classification_loss = 0
    total_regression_loss = 0
    
    for point_clouds, labels, values in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(point_clouds)
        
        class_output = outputs[0]
        ta_output, re_output, tr_output = outputs[1], outputs[2], outputs[3]
        
        # Calculate classification loss
        class_loss = classification_loss(class_output, labels)
        
        # Calculate regression loss
        regression_losses = []
        for i in range(len(labels)):
            if labels[i] == 2:  # Trapezoid
                regression_losses.append(regression_loss(ta_output[i], values[i]))
            elif labels[i] == 0:  # Rectangle
                regression_losses.append(regression_loss(re_output[i], values[i][:3]))
            elif labels[i] == 1:  # Triangle
                regression_losses.append(regression_loss(tr_output[i], values[i]))
        
        regression_loss_value = torch.stack(regression_losses).mean()
        
        # Combine the losses
        loss = class_loss + regression_loss_value
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_classification_loss += class_loss.item()
        total_regression_loss += regression_loss_value.item()
    
    print(f"Epoch [{epoch+1}], Total Loss: {total_loss:.4f}, Classification Loss: {total_classification_loss:.4f}, Regression Loss: {total_regression_loss:.4f}")
    torch.save(model.state_dict(), "reg epoch"+str(epoch))

print('Finished Training')
