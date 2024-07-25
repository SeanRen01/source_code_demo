import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
from torch.utils.data import Dataset, DataLoader
from shapely.geometry import Polygon, Point

# Define the lists to store point clouds for each shape
tr_list, ta_list, re_list = [], [], []

# For the purpose of uncertainty analysis, we need to store the values of length
# List to hold values
tr_value, ta_value, re_value = [],[],[]

# Some variables to be used throughout the point cloud generation
N_pts = 1000
NUM_TEMP = 1000
origin = np.array([[0, 0]])

# Function to generate points inside polygons

def generate_points(vertices, num_points):
    polygon = Polygon(vertices)
    
    min_x, min_y, max_x, max_y = polygon.bounds
    
    points = []
    num_generated = 0
    # To make sure we generate expected number of points
    while num_generated < num_points:
        # Generate a random point within the bounding box
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        point = Point(x, y)
        
        # Check if the point is inside the polygon
        if polygon.contains(point):
            points.append((x, y))
            num_generated += 1
    
    return points

# Generate trapezoid point clouds
for trial in range(NUM_TEMP):
    # The range for each values
    w1, h1, h2 = 95 * np.random.uniform(0.5, 2), 20 * np.random.uniform(0.5, 2), 60 * np.random.uniform(0.5, 2)
    
    # Add the redundant parameter, the top width
    w2 = np.sqrt(w1*w1 + (h1-h2)*(h1-h2))
    temp = [w1, w2, h1, h2]
    
    tr_value.append(temp)
    p2, p3, p4 = np.array([[w1, 0]]), np.array([[w1, h2]]), np.array([[0, h1]])
    vertices = np.concatenate([origin, p2, p3, p4], axis=0)
    tr_list.append(generate_points(vertices, N_pts))

# Generate rectangle point clouds
for trial in range(NUM_TEMP):
    w, h = 45 * np.random.uniform(0.5, 2), 20 * np.random.uniform(0.5, 2)
    
    # The redundant parameter: diagonal
    d =np.sqrt( w*w + h*h)
    temp = [w, h, d, 0]
    re_value.append(temp)
    p2, p3 = np.array([[w, 0]]), np.array([[w, h]])
    vertices = np.concatenate([origin, p2, p3], axis=0)
    re_list.append(generate_points(vertices, N_pts))

# Generate triangle point clouds
for trial in range(NUM_TEMP):
    x1, x2 = 20 * np.random.uniform(0.5, 2), 20 * np.random.uniform(0.5, 2)
    
    while x1 == x2:
        x2 = 20 * np.random.uniform(0.5, 2)
        
    y1, y2 = 20 * np.random.uniform(0.5, 2), 20 * np.random.uniform(0.5, 2)
    
    while y1 == y2:
        y2 = 20 * np.random.uniform(0.5, 2)
        
    a1 = np.sqrt(x1*x1 + y1*y1)
    a2 = np.sqrt(x2*x2 + y2*y2)
    a3 = np.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))
    
    # Using Heron's formula to compute the height
    s = (a1 + a2 + a3) / 2
    A = np.sqrt(s * (s - a1) * (s - a2) * (s - a3))
    h = 2 * A / a1  # Height corresponding to side a1
    temp = [a1, a2, a3, h]
    
    ta_value.append(temp)
    p2, p3 = np.array([[x1, y1]]), np.array([[x2, y2]])
    vertices = np.concatenate([origin, p2, p3], axis=0)
    ta_list.append(generate_points(vertices, N_pts))


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
        self.fctr3 = nn.Linear(1024, 1024) 
        self.a_tr3 = nn.ReLU(inplace=True)
        self.fctr4 = nn.Linear(1024, 4)
        self.a_tr4 = nn.Softplus()
        
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
        self.d_tr3 = nn.Dropout(p=0.15)

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
        return x_latent

    def classify(self, x):
        x = self.d_class(self.a_class(self.fc3_1(x)))
        x = self.fc3_2(x)
        return self.logsoftmax(x)
    
    def reg(self, xs_pre, labels):
        
        xs_pre_ta = self.d_ta1(self.a_ta1(self.fcta1(xs_pre)))
        xs_pre_ta = self.d_ta2(self.a_ta2(self.fcta2(xs_pre_ta)))
        params_ta = self.a_ta3(self.fcta3(xs_pre_ta))

        xs_pre_re = self.d_re1(self.a_re1(self.fcre1(xs_pre)))
        xs_pre_re = self.d_re2(self.a_re2(self.fcre2(xs_pre_re)))
        params_re = self.a_re3(self.fcre3(xs_pre_re))

        xs_pre_tr = self.d_tr1(self.a_tr1(self.fctr1(xs_pre)))
        xs_pre_tr = self.d_tr2(self.a_tr2(self.fctr2(xs_pre_tr)))
        xs_pre_tr = self.d_tr3(self.a_tr3(self.fctr3(xs_pre_tr)))
        params_tr = self.a_tr3(self.fctr3(xs_pre_tr))
        
        return params_tr[:, :4], params_ta[:, :4], params_re[:, :3]
    
# Same for two models
model = shapeDetector()
num_epochs = 20

# Loss function for classification
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optimizers.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)



# Combine point clouds and create labels
all_point_clouds = tr_list + ta_list + re_list
labels = [2] * len(tr_list) + [1] * len(ta_list) + [0] * len(re_list)

all_point_clouds_reg = tr_list + ta_list + re_list
labels_reg = [2] * len(tr_list) + [1] * len(ta_list) + [0] * len(re_list)

# Convert to tensors
tensor_point_clouds = [torch.tensor(pc, dtype=torch.float32).transpose(0, 1) for pc in all_point_clouds]
tensor_labels = torch.tensor(labels, dtype=torch.long)



# Create dataset and dataloader
class PointCloudDataset(Dataset):
    def __init__(self, point_clouds, labels):
        self.point_clouds = point_clouds
        self.labels = labels

    def __len__(self):
        
        return len(self.point_clouds)

    def __getitem__(self, idx):
        return self.point_clouds[idx], self.labels[idx]

dataset = PointCloudDataset(tensor_point_clouds, tensor_labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


print('start training for classification')
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(dataloader, 0):
        
        inputs, labels = inputs.float(), labels.long()
        optimizer.zero_grad()
        outputs = model(inputs)
        
        predictions = model.classify(outputs)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    running_loss /= len(dataloader)
    print("epoch:{}, train_loss:{:.4f}".format(epoch+1, running_loss))
    torch.save(model.state_dict(), "cla epoch"+str(epoch))

print('Finished Training for classification')







# Now we train for parameters
# Prepare the data for parameters
# Convert to PyTorch tensors
class PointCloudDataset2(Dataset):
    def __init__(self, inputs, targets, labels):
        self.inputs = inputs
        self.targets = targets
        self.labels = labels
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        input_data = self.inputs[index]
        target = self.targets[index]
        label = self.labels[index]
        return input_data, target, label

# Feature of input data
tensor_point_clouds_reg = [torch.tensor(pc, dtype=torch.float32).transpose(0, 1) for pc in all_point_clouds_reg]

# Feature of parameters
all_parameters = tr_value + ta_value + re_value

# Generate the tensor from list/array
tensor_parameter_reg = torch.tensor(all_parameters, dtype=torch.float32)



# Dataset and DataLoader for batching
dataset2 = PointCloudDataset2(tensor_point_clouds_reg, tensor_parameter_reg, labels_reg)
dataloader = DataLoader(dataset2, batch_size=32, shuffle=True)

# Optimizer and loss function
optimizer = optimizers.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
criterion = nn.MSELoss()

print('start training for regression')

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, targets, labels) in enumerate(dataloader, 0):
        inputs, targets, labels = inputs.float(), targets.float(), labels.long()
        
        # Forward pass to get the results
        latent_features = model(inputs)
        params_tr, params_ta, params_re = model.reg(latent_features, labels)
        
        loss = 0
        
        for j in range(len(labels)):
            if labels[j] == 0:  # Rectangle
                target_re = targets[j][:3]  # Use only the first 2 values
                pred_re = params_re[j]
                loss += criterion(pred_re, target_re)
            elif labels[j] == 1:  # Triangle
                target_ta = targets[j][:4]
                pred_ta = params_ta[j]
                loss += criterion(pred_ta, target_ta)
            elif labels[j] == 2:  # Trapezoid
                target_tr = targets[j][:4]
                pred_tr = params_tr[j]
                loss += criterion(pred_tr, target_tr)
        
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        
    # Print average loss per epoch
    running_loss /= len(dataloader)
    print("epoch:{}, train_loss:{:.4f}".format(epoch + 1, running_loss))
    torch.save(model.state_dict(), "reg epoch"+str(epoch))


print('Finished Training for regression')
