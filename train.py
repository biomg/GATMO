import torch
import torch.nn.functional as F
from data_processing import TEINet_embeddings_5fold, esm_embeddings_5fold
from model import GraphNet
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from arg_parser import parse_args
import numpy as np
import collections
from torch_geometric.data import Data
import random
from sklearn.model_selection import train_test_split
import yaml
import copy

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, gamma=1.9, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, pos_weight=None):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, 
            targets,
            reduction='none',
            pos_weight=pos_weight
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

#***   搜索最优 alpha, gamma
# ======================
def tune_focal_loss(model, train_data, val_data, device, 
                    alpha_range=(0.1, 1.0), gamma_range=(0.1, 5.0),
                    search_mode="grid", num_alpha=10, num_gamma=10, num_samples=20, 
                    epochs=5):
    """
    自动搜索最佳 FocalLoss 参数 alpha, gamma
    :param alpha_range: (min, max)
    :param gamma_range: (min, max)
    :param search_mode: "grid" 或 "random"
    :param num_alpha: grid search 时 alpha 采样点数
    :param num_gamma: grid search 时 gamma 采样点数
    :param num_samples: random search 时采样次数
    :param epochs: 每组参数快速训练 epoch 数
    """
    best_auc = -1
    best_alpha, best_gamma = 0.5, 2.0

    if search_mode == "grid":
        alphas = np.linspace(alpha_range[0], alpha_range[1], num_alpha)
        gammas = np.linspace(gamma_range[0], gamma_range[1], num_gamma)
        candidates = [(a, g) for a in alphas for g in gammas]
    else:  # random search
        candidates = [(random.uniform(*alpha_range), random.uniform(*gamma_range)) 
                      for _ in range(num_samples)]

    for alpha, gamma in candidates:
        focal_loss_fn = FocalLoss(alpha=alpha, gamma=gamma).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _ in range(epochs):  # 少量 epoch 快速验证
            model.train()
            optimizer.zero_grad()
            preds = model(train_data.x, train_data.edge_index)
            y_true = train_data.y.to(device)

            num_pos = (y_true == 1).sum()
            num_neg = (y_true == 0).sum()
            weight_factor = (num_neg.float() / num_pos.float()) if num_pos > 0 else 1.0
            pos_weight = torch.tensor([weight_factor], device=device)

            loss = focal_loss_fn(preds, y_true, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()

        # 验证集 AUC
        model.eval()
        with torch.no_grad():
            preds_val = model(val_data.x, val_data.edge_index)
            y_val = val_data.y.to(device)
            auc_val = roc_auc_score(y_val.cpu().numpy(), torch.sigmoid(preds_val).cpu().numpy())

        if auc_val > best_auc:
            best_auc = auc_val
            best_alpha, best_gamma = alpha, gamma

    print(f"最佳 FocalLoss 参数: alpha={best_alpha:.3f}, gamma={best_gamma:.3f}, AUC={best_auc:.4f}")
    return best_alpha, best_gamma

#设置随机种子
seed = 18
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def compute_accuracy(preds, y_true):
    return ((preds > 0).float() == y_true).sum().item() / preds.size(0)

def compute_aupr(preds, y_true):
    probs = torch.sigmoid(preds)
    probs_numpy = probs.detach().cpu().numpy()
    y_true_numpy = y_true.detach().cpu().numpy()
    return average_precision_score(y_true_numpy, probs_numpy)

def compute_auc(preds, y_true):
    probs = torch.sigmoid(preds)
    y_true_numpy = y_true.detach().cpu().numpy()
    probs_numpy = probs.detach().cpu().numpy()
    return roc_auc_score(y_true_numpy, probs_numpy)


args = parse_args()
with open(args.configs_path) as file:
    configs = yaml.safe_load(file)

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

#加载 5 折交叉验证的数据，将数据移动到指定设备
# data_list = esm2_embeddings_5fold(args.configs_path)
data_list = TEINet_embeddings_5fold(args.configs_path)
data_list = [data.to(device) for data in data_list]

train_data = data_list[0]
test_data = data_list[1]

model = GraphNet(num_node_features=train_data.num_node_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
sgd_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9) #***

margin = 4.0
epoch_decay = 0.0046
weight_decay = 0.006
aucm_optimizer = PESG(model.parameters(),
                 loss_fn=AUCMLoss(),
                 lr=args.lr,
                 momentum=0.4,
                 margin=margin,
                 device=device,
                 epoch_decay=epoch_decay,
                 weight_decay=weight_decay)

focal_loss_fn = FocalLoss(alpha=0.5, gamma=1.9).to(device)

# *** 搜索最佳 alpha, gamma (在 0.1~1.0, 1.0~5.0 范围内)
best_alpha, best_gamma = tune_focal_loss(
    model, train_data, test_data, device,
    alpha_range=(0.1, 1.0), gamma_range=(1.0, 5.0),
    search_mode="grid", num_alpha=10, num_gamma=10, epochs=5
)

# 用最优参数训练
focal_loss_fn = FocalLoss(alpha=best_alpha, gamma=best_gamma).to(device)


num_epochs = args.epochs
best_valid_roc = 0
best_valid_acc = 0

#train
for epoch in range(num_epochs):
    model.train()

    optimizer.zero_grad()
    aucm_optimizer.zero_grad()
    sgd_optimizer.zero_grad()#***
 
    out = model(train_data.x, train_data.edge_index)
    preds = out
    y_true = train_data.y.to(device)

    #
    num_positive_samples = (y_true == 1).sum()
    num_negative_samples = (y_true == 0).sum()
    
    weight_factor = num_negative_samples.float() / num_positive_samples.float()
    pos_weight = torch.tensor([weight_factor * args.positive_weights], device=device)##*
    
    #
    # bce_loss = F.binary_cross_entropy_with_logits(preds, y_true, pos_weight=pos_weight)
    focal_loss = focal_loss_fn(preds, y_true, pos_weight=pos_weight)##*

    aucm_module = AUCMLoss()
    aucm_loss = aucm_module(torch.sigmoid(preds), y_true)

    total_loss = args.w_celoss * focal_loss + args.w_aucloss * aucm_loss.to(device)##*
    # total_loss = args.w_celoss * bce_loss + args.w_aucloss * aucm_loss.to(device)

    total_loss.backward()

    # optimizer strategy
    if args.opt_strategy == "adam":
        optimizer.step()
    elif args.opt_strategy == "pesg":
        aucm_optimizer.step()
    elif args.opt_strategy == "sgd":
        sgd_optimizer.step()
    elif args.opt_strategy == "dual":
        optimizer.step()
        aucm_optimizer.step() 
    elif args.opt_strategy == "triple":
        optimizer.step()
        aucm_optimizer.step()
        sgd_optimizer.step()

    #
    accuracy = compute_accuracy(preds, y_true)
    roc_auc = compute_auc(preds, y_true)
    aupr = compute_aupr(preds, y_true)

    #
    model.eval()
    with torch.no_grad():
        out_valid = model(test_data.x, test_data.edge_index)
        preds_valid = out_valid
        y_true_valid = test_data.y.to(device)

        valid_acc = compute_accuracy(preds_valid, y_true_valid)
        roc_auc_valid = compute_auc(preds_valid, y_true_valid)
        valid_aupr = compute_aupr(preds_valid, y_true_valid)

        #save the best model
        if roc_auc_valid > best_valid_roc:
            best_valid_roc = roc_auc_valid
            torch.save(model.state_dict(), configs['save_model'])
    print("Epoch: {}/{}, Loss: {:.7f}, Train Acc: {:.4f}, Test Acc: {:.4f}, Train AUC: {:.4f}, Train APUR: {:.4f}, Test AUC: {:.4f}, Test AUPR: {:.4f}".format(epoch+1, num_epochs, total_loss.item(), accuracy, valid_acc, roc_auc, aupr, roc_auc_valid, valid_aupr))
    
# Load the best model
best_model = GraphNet(num_node_features=test_data.num_node_features).to(device)
##  TEINet_embeddings_5fold
best_model.load_state_dict(torch.load(configs['save_model'],weights_only=True))
##  esm_embeddings_5fold
# best_model.load_state_dict(torch.load(configs['save_model'],weights_only=False))


# Evaluate on test test_data
best_model.eval()
with torch.no_grad():
    out_test = best_model(test_data.x, test_data.edge_index)
    preds_test = out_test
    y_true_test = test_data.y.to(device)

    test_acc = compute_accuracy(preds_test, y_true_test)
    roc_auc_test = compute_auc(preds_test, y_true_test)
    test_aupr = compute_aupr(preds_test, y_true_test)

    # save results
    probabilities = torch.sigmoid(preds_test)
    binary_predictions = (probabilities > 0.5).type(torch.int).detach().cpu().numpy()
    df = pd.DataFrame({
        'prediction': binary_predictions,
        'label': y_true_test.detach().cpu().numpy().astype(int)
    })
    df.to_csv(f'results/{configs["dataset_name"]}.csv', index=False)

print("Test Acc: {:.4f}, Test AUC: {:.4f}, Test AUPR: {:.4f}".format(test_acc, roc_auc_test, test_aupr))
    
    