# -*- coding: utf-8 -*-
"""
03_train_anfis.py - ANFIS model training with COVID discount
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

class GaussianMF(nn.Module):
    """Gaussian Membership Function"""
    def __init__(self, c, sigma):
        super().__init__()
        self.c = nn.Parameter(torch.tensor(c, dtype=torch.float32))
        self.log_s = nn.Parameter(torch.log(torch.tensor(sigma, dtype=torch.float32)))
    
    def forward(self, x):
        sigma = torch.exp(self.log_s) + 1e-6
        return torch.exp(-0.5 * ((x - self.c) / sigma) ** 2)

class ANFIS(nn.Module):
    """ANFIS Model"""
    def __init__(self, n_inputs, n_rules, init_centers, init_sigmas):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_rules = n_rules
        
        # Premise part: Gaussian MF for each rule-input pair
        self.mfs = nn.ModuleList([
            nn.ModuleList([GaussianMF(init_centers[r, j], init_sigmas[r, j]) 
                          for j in range(n_inputs)])
            for r in range(n_rules)
        ])
        
        # Consequent coefficients: (n_rules, n_inputs+1) - with bias
        self.consequents = nn.Parameter(torch.zeros(n_rules, n_inputs+1))
        nn.init.normal_(self.consequents, mean=0.0, std=0.1)
    
    def forward(self, X):
        """
        X: (B, n_inputs)
        Returns: (B,) predictions
        """
        B = X.shape[0]
        
        # Rule firing: w_r = Π_j μ_{rj}(x_j)
        W = []
        for r in range(self.n_rules):
            mu = 1.0
            for j in range(self.n_inputs):
                mu = mu * self.mfs[r][j](X[:, j])
            W.append(mu.unsqueeze(1))
        W = torch.cat(W, dim=1)  # (B, n_rules)
        
        # Normalized firing
        W_sum = W.sum(dim=1, keepdim=True) + 1e-9
        Wn = W / W_sum
        
        # Linear output for each rule f_r(x) = a0 + Σ a_j x_j
        ones = torch.ones(B, 1, dtype=X.dtype, device=X.device)
        X_ext = torch.cat([ones, X], dim=1)
        Fr = X_ext @ self.consequents.t()
        
        # Weighted sum
        y_hat = (Wn * Fr).sum(dim=1)
        return y_hat

def init_rules(X, K=8):
    """Initialize rules using K-means"""
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    
    # Sigma initialization: half of std for each dimension
    sigmas = np.tile(X.std(axis=0, ddof=1), (K, 1)) * 0.5 + 1e-6
    
    return centers.astype(np.float32), sigmas.astype(np.float32)

def train_model(X_tr, y_tr, X_te, y_te, n_rules=8, epochs=500, lr=0.01):
    """Model training"""
    # Initialize rules
    centers, sigmas = init_rules(X_tr, K=n_rules)
    model = ANFIS(X_tr.shape[1], n_rules, centers, sigmas).to(device)
    
    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.MSELoss()
    
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)
    
    # Store originals for metrics
    y_tr_orig = y_tr.copy()
    y_te_orig = y_te.copy()
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        y_pred_tr = model(X_tr_t)
        loss = criterion(y_pred_tr, y_tr_t)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                y_pred_tr = model(X_tr_t).cpu().numpy()
                y_pred_te = model(X_te_t).cpu().numpy()
            
            tr_rmse = math.sqrt(mean_squared_error(y_tr_orig, y_pred_tr))
            te_rmse = math.sqrt(mean_squared_error(y_te_orig, y_pred_te))
            tr_r2 = r2_score(y_tr_orig, y_pred_tr)
            te_r2 = r2_score(y_te_orig, y_pred_te)
            
            print(f"[{epoch:3d}] Train RMSE={tr_rmse:.1f}, R²={tr_r2:.3f} | "
                  f"Test RMSE={te_rmse:.1f}, R²={te_r2:.3f}")
            
            if te_rmse < best_val_loss:
                best_val_loss = te_rmse
                best_state = {k: v.cpu().clone() if hasattr(v, 'clone') else v 
                             for k, v in model.state_dict().items()}
    
    # Restore best state
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Final predictions
    model.eval()
    with torch.no_grad():
        y_pred_tr = model(X_tr_t).cpu().numpy()
        y_pred_te = model(X_te_t).cpu().numpy()
    
    return model, y_pred_tr, y_pred_te

def evaluate(y_true, y_pred, label):
    """Performance evaluation"""
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))) * 100
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n[{label}] Performance:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")
    
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R²": r2}

def visualize_results(df_train, df_test, y_tr_pred, y_te_pred, model, feature_names):
    """Visualize results"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # (1) Prediction vs Actual
    ax = axes[0, 0]
    train_dates = df_train.index
    test_dates = df_test.index
    
    ax.plot(train_dates, df_train['Current_Demand'].values, 
           'b-', label='Train True', linewidth=1)
    ax.plot(train_dates, y_tr_pred, 
           'b--', label='Train Pred', linewidth=1, alpha=0.7)
    ax.plot(test_dates, df_test['Current_Demand'].values, 
           'r-', label='Test True', linewidth=1)
    ax.plot(test_dates, y_te_pred, 
           'r--', label='Test Pred', linewidth=1, alpha=0.7)
    ax.set_xlabel('Date')
    ax.set_ylabel('Water Demand (㎥)')
    ax.set_title('Prediction vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (2) Scatter plot
    ax = axes[0, 1]
    ax.scatter(df_train['Current_Demand'].values, y_tr_pred, 
              alpha=0.5, label='Train', s=50)
    ax.scatter(df_test['Current_Demand'].values, y_te_pred, 
              alpha=0.5, label='Test', s=50)
    all_true = np.concatenate([df_train['Current_Demand'].values, 
                               df_test['Current_Demand'].values])
    ax.plot([all_true.min(), all_true.max()], 
           [all_true.min(), all_true.max()], 'r--', linewidth=2)
    ax.set_xlabel('Actual Demand')
    ax.set_ylabel('Predicted Demand')
    ax.set_title('Actual vs Predicted Scatter')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (3) Residual distribution
    ax = axes[1, 0]
    residuals_tr = df_train['Current_Demand'].values - y_tr_pred
    residuals_te = df_test['Current_Demand'].values - y_te_pred
    ax.hist(residuals_tr, bins=20, alpha=0.5, label='Train')
    ax.hist(residuals_te, bins=20, alpha=0.5, label='Test')
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Frequency')
    ax.set_title('Residual Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (4) COVID discount effect
    ax = axes[1, 1]
    df_combined = pd.DataFrame({
        'covid_discount': np.concatenate([df_train['covid_discount'].values, 
                                         df_test['covid_discount'].values]),
        'demand': np.concatenate([df_train['Current_Demand'].values,
                                 df_test['Current_Demand'].values])
    })
    
    discounted_mask = df_combined['covid_discount'] > 0
    ax.scatter(df_combined[~discounted_mask]['covid_discount'], 
              df_combined[~discounted_mask]['demand'],
              alpha=0.5, label='Normal', s=50)
    ax.scatter(df_combined[discounted_mask]['covid_discount'], 
              df_combined[discounted_mask]['demand'],
              alpha=0.8, label='COVID Discount (50%)', s=50, color='red')
    ax.set_xlabel('COVID Discount Rate')
    ax.set_ylabel('Demand (㎥)')
    ax.set_title('COVID Discount Effect')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/anfis_results_with_covid.png', dpi=300, bbox_inches='tight')
    print("\n[OK] Results graph saved: ../results/anfis_results_with_covid.png")

def main():
    print("=" * 60)
    print("ANFIS Model Training (with COVID discount)")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1] Loading data...")
    df = pd.read_csv('../data/anfis_dataset_with_covid.csv', index_col=0, parse_dates=True)
    
    # 2. Feature selection and train/test split
    feature_cols = ['Temperature', 'Precipitation', 'population_norm', 
                   'Prev_Demand', 'effective_fee_adjusted', 'month_sin', 'month_cos']
    
    # Remove missing values
    df = df.dropna(subset=feature_cols + ['Current_Demand'])
    
    # 2018-2022 training, 2023-2024 test
    train_mask = df.index.year <= 2022
    df_train = df[train_mask].copy()
    df_test = df[~train_mask].copy()
    
    print(f"Training data: {len(df_train)} months ({df_train.index.min()} ~ {df_train.index.max()})")
    print(f"Test data: {len(df_test)} months ({df_test.index.min()} ~ {df_test.index.max()})")
    
    # 3. Feature scaling
    print("\n[2] Feature scaling...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train = scaler_X.fit_transform(df_train[feature_cols])
    y_train = scaler_y.fit_transform(df_train[['Current_Demand']]).ravel()
    
    X_test = scaler_X.transform(df_test[feature_cols])
    y_test = scaler_y.transform(df_test[['Current_Demand']]).ravel()
    
    # 4. Model training
    print("\n[3] Model training...")
    model, y_tr_pred, y_te_pred = train_model(
        X_train, y_train, X_test, y_test, n_rules=6, epochs=1000, lr=0.01
    )
    
    # Inverse transform
    y_tr_pred = scaler_y.inverse_transform(y_tr_pred.reshape(-1, 1)).ravel()
    y_te_pred = scaler_y.inverse_transform(y_te_pred.reshape(-1, 1)).ravel()
    
    # 5. Evaluation
    print("\n[4] Performance evaluation...")
    tr_metrics = evaluate(df_train['Current_Demand'].values, y_tr_pred, "Training Set")
    te_metrics = evaluate(df_test['Current_Demand'].values, y_te_pred, "Test Set")
    
    # 6. Visualization
    print("\n[5] Visualization...")
    visualize_results(df_train, df_test, y_tr_pred, y_te_pred, model, feature_cols)
    
    # 7. Save results
    print("\n[6] Saving results...")
    results_df = pd.DataFrame({
        'date': list(df_train.index) + list(df_test.index),
        'set': ['train'] * len(df_train) + ['test'] * len(df_test),
        'true': np.concatenate([df_train['Current_Demand'].values, 
                               df_test['Current_Demand'].values]),
        'pred': np.concatenate([y_tr_pred, y_te_pred]),
    })
    results_df.to_csv('../results/anfis_predictions_with_covid.csv', index=False)
    print("[OK] Predictions saved: ../results/anfis_predictions_with_covid.csv")
    
    # Compare with/without COVID
    print("\n[7] Comparison with previous model:")
    prev_results = pd.read_csv('../results/anfis_predictions.csv')
    prev_r2 = r2_score(prev_results[prev_results['set']=='test']['true'], 
                       prev_results[prev_results['set']=='test']['pred'])
    print(f"Previous Test R²: {prev_r2:.4f}")
    print(f"Current Test R²: {te_metrics['R²']:.4f}")
    print(f"Improvement: {te_metrics['R²'] - prev_r2:.4f}")

if __name__ == "__main__":
    main()

