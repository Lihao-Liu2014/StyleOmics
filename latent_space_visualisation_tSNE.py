import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import seaborn as sns
import umap
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt


unified_latent_space_path = "/Users/rascalpel/Desktop/Datasets for the final project/latent_space/mofa_unified_latent_space.npy"  
phenotype_labels_path = "/Users/rascalpel/Desktop/Datasets for the final project/processed/BRCA/phenotype_processed.csv"       
num_complete_patients = 65  
variable = 'ER.Status'  


unified_latent_space = np.load(unified_latent_space_path)  
if unified_latent_space.shape[0] == 1:
    unified_latent_space = unified_latent_space.squeeze(0)  
print(f"Unified Latent Space Shape: {unified_latent_space.shape}")


phenotype_data = pd.read_csv(phenotype_labels_path, index_col=0).T
print(f"Phenotype Data Shape: {phenotype_data.shape}")
phenotype_data = phenotype_data.iloc[:num_complete_patients]  
print(f"Phenotype Data Shape: {phenotype_data.shape}")


if variable not in phenotype_data.columns:
    raise ValueError(f"The variable '{variable}' is not found in the phenotype data.")
labels = phenotype_data[variable].fillna("unknown").values.astype(str)
labels_df = phenotype_data.fillna('unknown')
print(f"Labels Shape: {labels.shape}")


if unified_latent_space.shape[0] != len(labels):
    raise ValueError("Mismatch between unified latent space samples and phenotype labels.")





pca = PCA(n_components=2)
pca_results = pca.fit_transform(unified_latent_space)

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=pca_results[:, 0],
    y=pca_results[:, 1],
    hue=labels_df["HER2.Status"],  
    palette="tab10",
    s=80,
    alpha=0.8,
    edgecolor="w"
)
plt.title("ER.status", fontsize=16)
plt.xlabel("PC1", fontsize=14)
plt.ylabel("PC2", fontsize=14)
plt.legend(title=None, fontsize=10)
plt.grid(alpha=0.3)



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def compute_fisher_score(latent_space, labels, feature_index):
    feature_values = latent_space[:, feature_index]
    unique_classes = np.unique(labels)

    numerator = 0
    denominator = 0
    overall_mean = np.mean(feature_values)

    for cls in unique_classes:
        cls_mask = (labels == cls)
        cls_values = feature_values[cls_mask]

        cls_mean = np.mean(cls_values)
        cls_variance = np.var(cls_values)
        cls_size = len(cls_values)

        numerator += cls_size * (cls_mean - overall_mean) ** 2
        denominator += cls_size * cls_variance

    if denominator == 0:
        return 0
    return numerator / denominator


phenotype_features = [
    "pathologic_stage",
    "pathology_T_stage",
    "pathology_N_stage",
    "histological_type",
    "PAM50",
    "ER.Status",
    "PR.Status",
    "HER2.Status"
]


results = {}


for feature in phenotype_features:
    if feature not in phenotype_data.columns:
        print(f"Feature {feature} not found in phenotype data. Skipping...")
        continue

    
    labels = phenotype_data[feature].fillna("unknown").values.astype(str)

    
    fisher_scores = []
    for i in range(unified_latent_space.shape[1]):
        score = compute_fisher_score(unified_latent_space, labels, i)
        fisher_scores.append(score)

    
    sorted_indices = np.argsort(fisher_scores)[::-1]  
    best_factors = sorted_indices[:2]
    results[feature] = {
        "fisher_scores": fisher_scores,
        "best_factors": best_factors,
        "labels": labels
    }
    print(
        f"{feature}: Top 2 Latent Factors: {best_factors} with Scores: "
        f"{fisher_scores[best_factors[0]]}, {fisher_scores[best_factors[1]]}"
    )


num_features = len(phenotype_features)
num_rows = (num_features + 2) // 3  
plt.figure(figsize=(15, 5 * num_rows))

for i, feature in enumerate(phenotype_features):
    if feature not in results:
        continue

    
    best_factors = results[feature]["best_factors"]
    labels = results[feature]["labels"]
    top_latent_factors = unified_latent_space[:, best_factors]

    
    ax = plt.subplot(2, 4, i + 1)
    sns.scatterplot(
        x=top_latent_factors[:, 0],
        y=top_latent_factors[:, 1],
        hue=labels,
        palette="husl",
        s=50,
        alpha=0.8,
        edgecolor="w",
        ax=ax
    )
    ax.set_xlabel(f"Latent Factor {best_factors[0] + 1}", fontsize=10)
    ax.set_ylabel(f"Latent Factor {best_factors[1] + 1}", fontsize=10)
    ax.set_title(feature, fontsize=12)
    ax.legend(title=feature, loc="best", fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()





print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=24, random_state=42)
tsne_results = tsne.fit_transform(unified_latent_space)


num_labels = min(20, len(labels_df.columns))  
labels_to_plot = labels_df.columns[:num_labels]
fig, axes = plt.subplots(2, 4, figsize=(20, 12))  
axes = axes.flatten()


for i, label in enumerate(phenotype_features):
    ax = axes[i]
    sns.scatterplot(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        hue=labels_df[label],
        palette="tab10",
        s=60,
        alpha=0.8,
        edgecolor="w",
        ax=ax
    )
    ax.set_title(label, fontsize=14)
    ax.set_xlabel("t-SNE Component 1", fontsize=12)
    ax.set_ylabel("t-SNE Component 2", fontsize=12)

    
    ax.legend(
        title=label, fontsize=10, loc="upper right", title_fontsize=12,
        frameon=True, fancybox=True, framealpha=0.7
    )
    ax.grid(alpha=0.4)


for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

