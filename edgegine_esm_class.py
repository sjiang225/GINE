# %%
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINEConv, PDNConv, HEATConv, Linear, BatchNorm, GINConv
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import to_hetero
from pytorchtools_update import EarlyStopping
import torch_geometric.transforms as T
import pandas as pd
import numpy as np
import os
from typing import List
from datetime import datetime
import random
from torchinfo import summary

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# %%
# 
def create_result_dir(base_dir: str, conv_type: str, epsilon: float, train_epsilon: bool, random_analyst_feat: bool, data_name: str) -> str:
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(base_dir, f'{current_time}_{conv_type}_{epsilon}_{train_epsilon}_{"random" if random_analyst_feat else "analyst"}_{data_name}')
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def print_args(args):
    print("Hyperparameters:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("\n")

# %%

# 
parser = argparse.ArgumentParser(description='Bipartite Graph Neural Network')
parser.add_argument('--data_name', type=str, default='0925', help='Data name')
parser.add_argument('--seed', type=int, default=777, help='Random seed')
parser.add_argument('--hidden_channels', type=int, nargs='+', default=[128, 256], help='Hidden channels for each layer')
parser.add_argument('--out_channels', type=int, default=1, help='Output channels in GNN')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs')
parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
parser.add_argument('--patience', type=int, default=500, help='Patience for early stopping')
parser.add_argument('--conv_type', type=str, default='GIN', choices=['GAT', 'GINE', 'GIN', 'HEAT'], help='Type of graph convolution')
parser.add_argument('--use_edge_features', action='store_true', help='Whether to use edge features')
parser.add_argument('--dropout', type=float, default=0.7, help='Dropout rate')
parser.add_argument('--train_ratio', type=float, default=0.6, help='Training data ratio')
parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation data ratio')
parser.add_argument('--test_ratio', type=float, default=0.2, help='Test data ratio')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--base_result_dir', type=str, default='/project/zhiwei/sj225/EPS/Result', help='Base directory for results')
parser.add_argument('--epsilon', type=float, default=0.5, help='Epsilon value for GINEConv')
parser.add_argument('--train_epsilon', action='store_true', help='Whether to train epsilon in GINEConv')
parser.add_argument('--random_analyst_feat', action='store_true', help='Use random initialization for analyst features')
parser.add_argument('--analyst_feat_dim', type=int, default=13, help='Dimension of analyst features (random or not)')
parser.add_argument('--num_initializations', type=int, default=5, help='Number of random initializations')
parser.add_argument('--mode', type=str, default='min', choices=['min', 'max'], help='Mode for early stopping')
parser.add_argument('--monitor', type=str, default='loss', choices=['loss', 'acc'], help='Metric to monitor')
parser.add_argument('--target', type=str, default='ESP_B', help='Target column name')
args = parser.parse_args()
print_args(args)

# set_seed(args.seed)
# 
result_dir = create_result_dir(args.base_result_dir, args.conv_type, args.epsilon, args.train_epsilon, args.random_analyst_feat, args.data_name)

# 
args.model_path = os.path.join(result_dir, f'model_{args.conv_type.lower()}.pt')
args.picture_path = os.path.join(result_dir, 'training_metrics.png')
args.csv_path = os.path.join(result_dir, 'test_results.csv')

# 
print(f"Model will be saved to: {args.model_path}")
print(f"Training metrics plot will be saved to: {args.picture_path}")
#print('NO estimator')


# 在Jupyter Notebook
#args = parser.parse_args([])

# %%
edge = pd.read_csv(f'/project/zhiwei/sj225/EPS/Data/1218/edge{args.data_name}.csv')
analyst = pd.read_csv(f'/project/zhiwei/sj225/EPS/Data/1218/analyst{args.data_name}.csv')
company = pd.read_csv(f'/project/zhiwei/sj225/EPS/Data/1218/company{args.data_name}.csv')
data=edge.sort_values(by="date", ascending=True)
date_list=data['date'].unique()

# %%
edge = edge.sort_values(by=['date','analys', 'ticker'])
analyst = analyst.sort_values(by=['date','analys'])
company = company.sort_values(by=['date','ticker'])


# %%
def Roos(y_true,y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    diff = np.sum((np.square(y_true-y_pred))) / np.sum(np.square(y_true))
    result = 1-diff
    return result

# %%
def embed_estimator(data, embedding_dim):
    unique_estimators = data['estimator'].unique()
    estimator_mapping = {estimator: i for i, estimator in enumerate(unique_estimators)}
    data['estimator'] = data['estimator'].map(estimator_mapping)
    embedding_matrix = torch.nn.Embedding(len(unique_estimators), embedding_dim)
    data['estimator_emb'] = data['estimator'].apply(lambda x: embedding_matrix(torch.tensor(x)).detach().numpy())
    return data, embedding_matrix

# %%
#analyst, estimator_embedding_matrix = embed_estimator(analyst, embedding_dim=10)  # Set embedding_dim to 10

# %%
def generate_dynamic_graph_data(date_list, edge, company, analyst, args):
    dataset = []
    
    for time_step in date_list:
        current_edge = edge[edge['date'] == time_step]
        current_company = company[company['date'] == time_step]
        current_analyst = analyst[analyst['date'] == time_step]
        
        # Extracting features
        company_feat = current_company[['atq', 'ni', 'dv', 'acc', 'invest', 'mc', 'bm', 'dinvt', 'dar', 'capx', 'gm', 'sga','mean_value']]
        company_mapping = {index: i for i, index in enumerate(current_company.ticker.unique())}
        
        # Extracting analyst features
        if args.random_analyst_feat:
            analyst_feat = np.random.randn(len(current_analyst), args.analyst_feat_dim)
        else:
            analyst_feat = current_analyst[['firms_followed','overall_performance', 'experience_with_any_firm_quarters']]
            #estimator_emb = np.vstack(current_analyst['estimator_emb'].values)  # Expand the estimator embedding
            #analyst_feat = np.hstack([analyst_feat.to_numpy(dtype=np.float32), estimator_emb])
        
        analyst_mapping = {index: i for i, index in enumerate(current_analyst.analys.unique())}
        
        current_edge = current_edge[current_edge['ticker'].isin(current_company['ticker'])]
        # Extracting edge features
        edge_feat = current_edge[['value', 'first_time_follow','firm_performance','experience_with_firm']]

        # Convert to tensor
        company_x = torch.tensor(company_feat.to_numpy(dtype=np.float32))
        #analyst_x = torch.tensor(analyst_feat, dtype=torch.float32)
        analyst_x = torch.tensor(analyst_feat.to_numpy(dtype=np.float32))
        edge_attr = torch.tensor(edge_feat.to_numpy(dtype=np.float32))
        
        # filter ticker
        #current_edge = current_edge[current_edge['ticker'].isin(current_company['ticker'])]
        
        # Create edge index
        src = [company_mapping[index] for index in current_edge['ticker']]
        dst = [analyst_mapping[index] for index in current_edge['analys']]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        
        # Define target
        #target = current_company[['actual']]
        target = current_company[args.target]
        #company_y = torch.tensor(target.to_numpy(dtype=np.int64)).squeeze()
        company_y = torch.tensor(target.to_numpy(dtype=np.float32)).unsqueeze(1)
        
        # Create heterograph
        hetero_data = HeteroData()
        hetero_data['company'].x = company_x
        hetero_data['analyst'].x = analyst_x
        hetero_data['company'].y = company_y
        hetero_data['company', 'interacts', 'analyst'].edge_index = edge_index
        hetero_data['company', 'interacts', 'analyst'].edge_attr = edge_attr
        hetero_data = T.ToUndirected()(hetero_data)
        
        hetero_data['company'].date = [time_step] * len(current_company)
        hetero_data['company'].ticker = current_company['ticker'].values
        
        
        
        dataset.append(hetero_data)

    return dataset


# %%
dataset = generate_dynamic_graph_data(date_list, edge, company, analyst, args)
 
    
class GNNModel(nn.Module):
    def __init__(self, in_channels: dict, hidden_channels: List[int], out_channels: int, use_edge_features: bool =True, eps: float = 0.5, train_eps: bool = True):
        super(GNNModel, self).__init__()
        self.use_edge_features = use_edge_features
        self.num_layers = len(hidden_channels) + 1

        self.convs = nn.ModuleList()
        self.company_lins = nn.ModuleList()
        self.analyst_lins = nn.ModuleList()
        
        for i, hidden_dim in enumerate(hidden_channels):
            if i == 0:
                self.company_lins.append(Linear(in_channels['company'], hidden_dim))
                self.analyst_lins.append(Linear(in_channels['analyst'], hidden_dim))
            else:
                self.company_lins.append(Linear(hidden_channels[i-1], hidden_dim))
                self.analyst_lins.append(Linear(hidden_channels[i-1], hidden_dim))

            if use_edge_features:
                edge_dim = 4  # 
                self.convs.append(GINEConv(nn.Sequential(
                    Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    Linear(hidden_dim, hidden_dim)
                ), eps=eps, train_eps=train_eps, edge_dim=edge_dim))
            else:
                self.convs.append(GINConv(nn.Sequential(
                    Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    Linear(hidden_dim, hidden_dim)
                ), eps=eps, train_eps=train_eps))

        self.company_lins.append(Linear(hidden_channels[-1], out_channels))
        self.analyst_lins.append(Linear(hidden_channels[-1], out_channels))
        
        if use_edge_features:
            self.convs.append(GINEConv(nn.Sequential(
                Linear(hidden_channels[-1], hidden_channels[-1]),
                nn.ReLU(),
                Linear(hidden_channels[-1], out_channels)
            ), eps=eps, train_eps=train_eps, edge_dim=edge_dim))
        else:
            self.convs.append(GINConv(nn.Sequential(
                Linear(hidden_channels[-1], hidden_channels[-1]),
                nn.ReLU(),
                Linear(hidden_channels[-1], out_channels)
            ), eps=eps, train_eps=train_eps))

        self.bns = nn.ModuleList([BatchNorm(hidden_dim) for hidden_dim in hidden_channels])

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        for i, conv in enumerate(self.convs[:-1]):
            x_dict['company'] = self.company_lins[i](x_dict['company'])
            x_dict['analyst'] = self.analyst_lins[i](x_dict['analyst'])

            if self.use_edge_features:
                # 
                for edge_type, edge_index in edge_index_dict.items():
                    src, _, dst = edge_type
                    edge_attr = edge_attr_dict[edge_type]
                    out = conv((x_dict[src], x_dict[dst]), edge_index, edge_attr)
                    x_dict[dst] = out

            else:
                # 
                for edge_type, edge_index in edge_index_dict.items():
                    src, _, dst = edge_type
                    out = conv((x_dict[src], x_dict[dst]), edge_index)
                    x_dict[dst] = out

            for node_type in x_dict:
                x_dict[node_type] = self.bns[i](x_dict[node_type])
                x_dict[node_type] = F.relu(x_dict[node_type])
                x_dict[node_type] = F.dropout(x_dict[node_type], p=0.5, training=self.training)


        if self.use_edge_features:
            for edge_type, edge_index in edge_index_dict.items():
                src, _, dst = edge_type
                edge_attr = edge_attr_dict[edge_type]
                out = self.convs[-1]((x_dict[src], x_dict[dst]), edge_index, edge_attr)
                x_dict[dst] = out
        else:
            for edge_type, edge_index in edge_index_dict.items():
                src, _, dst = edge_type
                out = self.convs[-1]((x_dict[src], x_dict[dst]), edge_index)
                x_dict[dst] = out
                
        #  sigmoid 
        for node_type in x_dict:
            x_dict[node_type] = torch.sigmoid(x_dict[node_type])
        

        return x_dict
    
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_channels = {'company': dataset[0]['company'].x.shape[1], 'analyst': dataset[0]['analyst'].x.shape[1]}    
#criterion = nn.MSELoss().to(device)
criterion = nn.BCELoss().to(device)




#using adamw
#optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)



# %%
from torch_geometric.loader import DataLoader

def split_time_series_dataset(dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    
    n = len(dataset)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_dataset = dataset[:train_end]
    val_dataset = dataset[train_end:val_end]
    test_dataset = dataset[val_end:]
    
    return train_dataset, val_dataset, test_dataset

# 
batch_size = args.batch_size  # 
train_ratio = args.train_ratio  # 
val_ratio = args.val_ratio
test_ratio = args.test_ratio

# 
train_dataset, val_dataset, test_dataset = split_time_series_dataset(dataset, 
                                                                     train_ratio=train_ratio, 
                                                                     val_ratio=val_ratio, 
                                                                     test_ratio=test_ratio)

# 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# 
print(f"Train period: {train_dataset[0]['company'].date[0]} to {train_dataset[-1]['company'].date[0]}")
print(f"Validation period: {val_dataset[0]['company'].date[0]} to {val_dataset[-1]['company'].date[0]}")
print(f"Test period: {test_dataset[0]['company'].date[0]} to {test_dataset[-1]['company'].date[0]}")

# %%
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_mse = 0
    #total_mae = 0
    all_targets = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            #out = model(batch.x_dict, batch.edge_index_dict)['company']
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)['company']
            targets = batch['company'].y
            loss = criterion(out, targets)
            
            total_loss += loss.item() * batch.num_graphs
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(out.cpu().numpy())
            all_predictions.extend((out > 0.5).float().cpu().numpy())


    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)
    auc = roc_auc_score(all_targets, all_probabilities)

    return avg_loss, accuracy, precision, recall, f1, auc

def test_model(model, test_loader, criterion, device, args):
    model.eval()
    test_loss = 0
    predictions = []
    probabilities = []
    targets = []
    tickers = []
    dates = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            #out = model(batch.x_dict, batch.edge_index_dict)['company']
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)['company']
            loss = criterion(out, batch['company'].y)


            test_loss += loss.item() * batch.num_graphs
            probabilities.extend(out.cpu().numpy().flatten().tolist())
            predictions.extend((out > 0.5).float().cpu().numpy().flatten().tolist())
            targets.extend(batch['company'].y.cpu().numpy().flatten().tolist())
            tickers.extend(np.concatenate(batch['company'].ticker).tolist())
            dates.extend(np.concatenate(batch['company'].date).tolist())

    test_loss /= len(test_loader.dataset)
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions)
    recall = recall_score(targets, predictions)
    f1 = f1_score(targets, predictions)
    auc = roc_auc_score(targets, probabilities)

    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}')
    # 
    df_results = pd.DataFrame({
        'Date': dates,
        'Ticker': tickers,
        'Prediction': predictions,
        'Probability': probabilities,
        'Target': targets
    })

    # 
    
    df_results.to_csv(args.csv_path, index=False)
    print(f"Test results saved to: {args.csv_path}")

    return test_loss, accuracy, precision, recall, f1, auc

# %%
# 
train_losses, val_losses, test_losses = [], [], []
train_accuracies, val_accuracies, test_accuracies = [], [], []
train_precisions, val_precisions, test_precisions = [], [], []
train_recalls, val_recalls, test_recalls = [], [], []
train_f1s, val_f1s, test_f1s = [], [], []
train_aucs, val_aucs, test_aucs = [], [], []

# %%
# 
best_val_loss = float('inf')
best_val_acc = 0
best_model_state = None
initialization_results = []

for init in range(args.num_initializations):
    print(f"\nInitialization {init + 1}/{args.num_initializations}")

    # 
    temp_model_path = os.path.join(result_dir, f'temp_model_{init}.pt')
    model = GNNModel(in_channels, args.hidden_channels, args.out_channels, use_edge_features=args.use_edge_features, eps=args.epsilon, train_eps=args.train_epsilon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=args.model_path)
    if args.mode == 'min':  
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=temp_model_path, mode='min', monitor='loss')
    else:
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=temp_model_path, mode='max', monitor='acc')
    
    for epoch in tqdm(range(args.epochs)):
        model.train()
        train_loss = 0
        all_train_targets = []
        all_train_predictions = []
        all_train_probabilities = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            #out = model(batch.x_dict, batch.edge_index_dict)['company']
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)['company']
            loss = criterion(out, batch['company'].y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch.num_graphs
            all_train_targets.extend(batch['company'].y.cpu().numpy())
            all_train_probabilities.extend(out.detach().cpu().numpy())
            all_train_predictions.extend((out > 0.5).float().detach().cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_accuracy = accuracy_score(all_train_targets, all_train_predictions)
        train_precision = precision_score(all_train_targets, all_train_predictions)
        train_recall = recall_score(all_train_targets, all_train_predictions)
        train_f1 = f1_score(all_train_targets, all_train_predictions)
        train_auc = roc_auc_score(all_train_targets, all_train_probabilities)
        
        # 
        val_loss, val_accuracy, val_precision, val_recall, val_f1, val_auc = evaluate(model, val_loader, criterion, device)

        # 
        test_loss, test_accuracy, test_precision, test_recall, test_f1, test_auc = evaluate(model, test_loader, criterion, device)

        # 
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)
        train_precisions.append(train_precision)
        val_precisions.append(val_precision)
        test_precisions.append(test_precision)
        train_recalls.append(train_recall)
        val_recalls.append(val_recall)
        test_recalls.append(test_recall)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        test_f1s.append(test_f1)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)
        test_aucs.append(test_auc)

        # 
        print(f'Epoch: {epoch + 1}')
        print(f'Train - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}')
        print(f'Val   - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}')
        print(f'Test  - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}')

            
        # Early stopping
        if args.mode == 'min':
            early_stopping(val_loss, model)
        else:
            early_stopping(val_accuracy, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break
        
    # 
    model.load_state_dict(torch.load(temp_model_path))
    
    # 
    val_loss, val_accuracy, val_precision, val_recall, val_f1, val_auc = evaluate(model, val_loader, criterion, device)
    
    # 
    initialization_results.append({
    'initialization': init + 1,
    'final_val_loss': val_loss,
    'final_val_accuracy': val_accuracy,
    'final_val_f1': val_f1,
    'final_val_auc': val_auc
})
    # 
    if args.mode == 'min':
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            print(f"New best model found at initialization {init + 1} with validation loss: {best_val_loss:.4f}")
    else:
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_state = model.state_dict()
            print(f"New best model found at initialization {init + 1} with validation accuracy: {best_val_acc:.4f}")

    # 
    os.remove(temp_model_path)
    

# 
print("\nResults for each initialization:")
for result in initialization_results:
    print(f"Initialization {result['initialization']}:")
    print(f"  Final validation loss: {result['final_val_loss']:.4f}")
    print(f"  Final validation accuracy: {result['final_val_accuracy']:.4f}")
    print(f"  Final validation F1: {result['final_val_f1']:.4f}")
    print(f"  Final validation AUC: {result['final_val_auc']:.4f}")

# 
best_init = min(initialization_results, key=lambda x: x['final_val_loss'])
print(f"\nBest model found at initialization {best_init['initialization']} with validation loss: {best_init['final_val_loss']:.4f}")

    
# 
torch.save(best_model_state, args.model_path)
print(f"Best model saved to {args.model_path}")
# %%
# 
# plt.figure(figsize=(20, 15))

# # Loss
# plt.subplot(3, 2, 1)
# plt.plot(train_losses, label='Train')
# plt.plot(val_losses, label='Validation')
# plt.plot(test_losses, label='Test')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Loss')
# plt.legend()

# # RMSE
# plt.subplot(3, 2, 2)
# plt.plot(train_rmses, label='Train')
# plt.plot(val_rmses, label='Validation')
# plt.plot(test_rmses, label='Test')
# plt.xlabel('Epochs')
# plt.ylabel('RMSE')
# plt.title('Root Mean Squared Error (RMSE)')
# plt.legend()

# # MAE
# plt.subplot(3, 2, 3)
# plt.plot(train_maes, label='Train')
# plt.plot(val_maes, label='Validation')
# plt.plot(test_maes, label='Test')
# plt.xlabel('Epochs')
# plt.ylabel('MAE')
# plt.title('Mean Absolute Error (MAE)')
# plt.legend()

# # R2 Score
# plt.subplot(3, 2, 4)
# plt.plot(train_r2s, label='Train')
# plt.plot(val_r2s, label='Validation')
# plt.plot(test_r2s, label='Test')
# plt.xlabel('Epochs')
# plt.ylabel('R2')
# plt.title('R2 Score')
# plt.legend()

# # Roos2 Score
# plt.subplot(3, 2, 5)
# plt.plot(train_roos2s, label='Train')
# plt.plot(val_roos2s, label='Validation')
# plt.plot(test_roos2s, label='Test')
# plt.xlabel('Epochs')
# plt.ylabel('Roos2')
# plt.title('Roos2 Score')
# plt.legend()

# plt.tight_layout()
# plt.savefig(args.picture_path)
# plt.show()

# %%
# 
model.load_state_dict(torch.load(args.model_path))
final_test_loss, final_test_accuracy, final_test_precision, final_test_recall, final_test_f1, final_test_auc = test_model(model, test_loader, criterion, device, args)
print("Final Test Results:")
print(f'Loss: {final_test_loss:.4f}, Accuracy: {final_test_accuracy:.4f}, Precision: {final_test_precision:.4f}, Recall: {final_test_recall:.4f}, F1: {final_test_f1:.4f}, AUC: {final_test_auc:.4f}')


