import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


unified_latent_space_path = "/Users/rascalpel/Desktop/Datasets for the final project/latent_space/unified_latent_space.npy"
phenotype_labels_path = "/Users/rascalpel/Desktop/Datasets for the final project/processed/BRCA/phenotype_processed.csv"

unified_latent_space = np.load(unified_latent_space_path)
if unified_latent_space.shape[0] == 1:
    unified_latent_space = unified_latent_space.squeeze(0)
print(f"Unified Latent Space Shape: {unified_latent_space.shape}")


phenotype_data = pd.read_csv(phenotype_labels_path, index_col=0).T
phenotype_data = phenotype_data.iloc[:unified_latent_space.shape[0]]
phenotype_data['overall_survival'] = pd.to_numeric(phenotype_data['overall_survival'], errors='coerce')
phenotype_data = phenotype_data.dropna(subset=['overall_survival'])

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error, r2_score


X = unified_latent_space  
y = phenotype_data['overall_survival'].values  


print(f"Features (X) Shape: {X.shape}, Labels (y) Shape: {y.shape}")

print(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=314)



kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))


gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=900)


print("Training the Gaussian Process Regressor...")
gpr.fit(X_train, y_train)


y_pred, y_std = gpr.predict(X_test, return_std=True)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2): {r2:.4f}")


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.errorbar(range(len(y_test)), y_pred, yerr=y_std, fmt='o', label='Predictions with Uncertainty')
plt.scatter(range(len(y_test)), y_test, color='red', label='True Values')
plt.xlabel("Sample Index")
plt.ylabel("Overall Survival")
plt.title("GPR: Style")
plt.legend()
plt.tight_layout()
plt.show()