import torch
import pandas as pd
import networkx as nx
import numpy as np
from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
from sklearn.model_selection import KFold
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.metrics import f1_score


# Function to hash IP addresses
def hash_ip(ip):
    # Replace this with your hashing logic if needed
    return hash(ip)


# Function to create NetworkX graphs from CSV data
def create_graphs_from_csv(df):
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %I:%M:%S %p")
    df["timestamp_minute"] = df["Timestamp"].dt.floor("min")
    df = df.sort_values(by=["Timestamp"])
    grouped = df.groupby("timestamp_minute")

    pyg_graphs = []
    scaler = StandardScaler()
    all_features = []

    for timestamp_minute, group in grouped:
        index = 0
        G = nx.DiGraph()
        previous_backward_node = None
        
        for idx, row in group.iterrows():
            src_ip = hash_ip(row["Src IP"])
            dst_ip = hash_ip(row["Dst IP"])
            
            forward_features = {
                "total_pkts": row["Total Fwd Packet"],
                "pkts_per_s": row["Fwd Packets/s"],
                "iat_std": row["Fwd IAT Std"],
                "total_len_pkt": row["Total Length of Fwd Packet"],
                "pkt_len_std": row["Fwd Packet Length Std"],
                "seg_size_avg": row["Fwd Segment Size Avg"],
                "init_win_bytes": row["FWD Init Win Bytes"],
                "pkt_len_mean": row["Fwd Packet Length Mean"],
                "iat_max": row["Fwd IAT Max"],
                "avg_pkt_size": row["Average Packet Size"],
                "subflow_bytes": row["Subflow Fwd Bytes"],
                "ip": src_ip,
                "direction": 0,  # 0 for forward
            }
            forward_node_id = f"{index}_fwd"
            G.add_node(forward_node_id, **forward_features)
            all_features.append(list(forward_features.values()))

            backward_features = {
                "total_pkts": row["Total Bwd packets"],
                "pkts_per_s": row["Bwd Packets/s"],
                "iat_std": row["Bwd IAT Std"],
                "total_len_pkt": row["Total Length of Bwd Packet"],
                "pkt_len_std": row["Bwd Packet Length Std"],
                "seg_size_avg": row["Bwd Segment Size Avg"],
                "init_win_bytes": row["Bwd Init Win Bytes"],
                "pkt_len_mean": row["Bwd Packet Length Mean"],
                "iat_max": row["Bwd IAT Max"],
                "avg_pkt_size": row["Average Packet Size"],
                "subflow_bytes": row["Subflow Bwd Bytes"],
                "ip": dst_ip,
                "direction": 1,  # 1 for backward
            }
            backward_node_id = f"{index}_bkd"
            G.add_node(backward_node_id, **backward_features)
            all_features.append(list(backward_features.values()))

            G.add_edge(forward_node_id, backward_node_id)
            if previous_backward_node is not None:
                G.add_edge(previous_backward_node, forward_node_id)

            previous_backward_node = backward_node_id
            index += 1

        labels = group["Label"].apply(lambda x: 0 if x.lower() in ["nonvpn", "non-tor"] else 1).tolist()
        graph_label = max(set(labels), key=labels.count)

        data = from_networkx(G)

        node_features = []
        for node_id in G.nodes:
            node_features.append(list(G.nodes[node_id].values()))

        data.x = torch.tensor(node_features, dtype=torch.float)
        data.y = torch.tensor([graph_label], dtype=torch.long)

        pyg_graphs.append(data)

    scaler.fit(all_features)
    for data in pyg_graphs:
        data.x = torch.tensor(scaler.transform(data.x.numpy()), dtype=torch.float)

    return pyg_graphs
''' Referenced the Code from Official GIN code site: https://github.com/weihua916/powerful-gnns/blob/master/README.md 
'''

class GINModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GINModel, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(input_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(),
                       Linear(hidden_dim, hidden_dim), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(),
                       Linear(hidden_dim, hidden_dim), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(),
                       Linear(hidden_dim, hidden_dim), ReLU()))
        self.fc1 = Linear(hidden_dim * 3, hidden_dim * 3)
        self.fc2 = Linear(hidden_dim * 3, output_dim)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index, batch):
        h1 = self.conv1(x, edge_index)
        h1 = h1.relu()
        h2 = self.conv2(h1, edge_index)
        h2 = h2.relu()
        h3 = self.conv3(h2, edge_index)
        h3 = h3.relu()

        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)

        h = torch.cat([h1, h2, h3], dim=1)
        h = self.fc1(h)
        h = h.relu()
        h = self.dropout(h)
        h = self.fc2(h)

        return torch.nn.functional.log_softmax(h, dim=-1)


def train_model(model, train_loader, test_loader, optimizer, criterion, device, epochs=50):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
            print(f"Epoch {epoch+1}/{epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    return model


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    for batch in test_loader:
        batch.to(device)
        output = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(output, batch.y)
        test_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += batch.y.size(0)
        correct += (predicted == batch.y).sum().item()

    accuracy = correct / total
    avg_loss = test_loss / len(test_loader)

    return avg_loss, accuracy


# Load your dataset from CSV
df = pd.read_csv("/home/jseo/DGN-Graph/Baseline_Models/GraphSAGE/graph_creation/data/Darknet.CSV")

# Create graphs from dataset
pyg_graphs = create_graphs_from_csv(df)

# Define model parameters
input_dim = pyg_graphs[0].x.shape[1]  # Input dimension based on features
hidden_dim = 64
output_dim = 2

# Define K-Fold Cross Validation
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Use GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize lists to store results
fold_losses = []
fold_accuracies = []

# Start cross-validation
for fold_idx, (train_index, test_index) in enumerate(kf.split(pyg_graphs)):
    print(f"Processing fold {fold_idx + 1}/{n_splits}")

    # Split data into train and test sets
    train_graphs = [pyg_graphs[i] for i in train_index]
    test_graphs = [pyg_graphs[i] for i in test_index]

    # Create DataLoaders
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

    # Initialize model
    model = GINModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    # Define optimizer and loss function
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()

    # Train model
    trained_model = train_model(model, train_loader, test_loader, optimizer, criterion, device)

    # Evaluate on test set
    test_loss, test_acc = evaluate_model(trained_model, test_loader, criterion, device)
    fold_losses.append(test_loss)
    fold_accuracies.append(test_acc)

    print(f"Fold {fold_idx + 1} Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Calculate average metrics across all folds
avg_loss = np.mean(fold_losses)
avg_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies)

print(f"Average Test Loss: {avg_loss:.4f}, Average Test Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
