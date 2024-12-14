from statistics import variance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch


data_dir = "/Users/rascalpel/Desktop/Datasets for the final project/processed/BRCA"
modalities = ["methylation", "mutation", "transcriptomics", "proteomics"]
num_complete_patients = 65  
datasets = {}
input_dims = {}

for modality in modalities:
    file_path = os.path.join(data_dir, f"{modality}_processed.csv")
    if os.path.exists(file_path):
        data = pd.read_csv(file_path, index_col=0).T.iloc[:num_complete_patients]  
        datasets[modality] = torch.tensor(data.values, dtype=torch.float32)
        input_dims[modality] = datasets[modality].shape[1]
        print(f"{modality}: Loaded with shape {datasets[modality].shape}")
    else:
        print(f"File not found for {modality}")

output_dir = "/Users/rascalpel/Desktop/Datasets for the final project/saved_components"
os.makedirs(output_dir, exist_ok=True)
decoder_weights_path = os.path.join(output_dir, "decoders.pt")

shared_latents = {mod: np.load(f"{output_dir}/{mod}_shared_latents.npy") for mod in modalities}
style_latents = {mod: np.load(f"{output_dir}/{mod}_style_latents.npy") for mod in modalities}


from integrate_omics_diffusion import ModalityDecoder


decoders = {}
latent_dim = 16
for modality, input_dim in input_dims.items():
    decoders[modality] = ModalityDecoder(shared_dim=latent_dim, style_dim=latent_dim, output_dim=input_dim)


decoder_states = torch.load(decoder_weights_path, map_location=torch.device("cpu"))
print(f"Loaded decoder keys: {decoder_states.keys()}")

for modality, decoder in decoders.items():
    if modality in decoder_states:
        decoder.load_state_dict(decoder_states[modality])
        print(f"Loaded weights for {modality}")
    else:
        print(f"Warning: No weights found for modality {modality}")

print(f"Decoders reloaded from: {decoder_weights_path}")


def compute_variance_explained(datasets, shared_latents, style_latents, decoders):
    variance_dict = {modality: [] for modality in datasets.keys()}

    for modality, data in datasets.items():
        decoder = decoders[modality]
        data = data.numpy()  
        z_shared = torch.tensor(shared_latents[modality]).to(next(decoder.parameters()).device)
        z_style = torch.tensor(style_latents[modality]).to(next(decoder.parameters()).device)

        
        recon_data = decoder(z_shared, z_style).detach().cpu().numpy()

        
        total_variance_per_feature = np.var(data, axis=0, ddof=1)
        total_variance = total_variance_per_feature.sum()  

        
        for latent_idx in range(z_shared.shape[1]):
            
            z_shared_single = torch.zeros_like(z_shared)
            z_shared_single[:, latent_idx] = z_shared[:, latent_idx]

            recon_single = decoder(z_shared_single, z_style).detach().cpu().numpy()

            
            recon_variance_per_feature = np.var(recon_single, axis=0, ddof=1)
            recon_variance = recon_variance_per_feature.sum()  

            
            explained_variance = recon_variance / total_variance if total_variance > 0 else 0
            variance_dict[modality].append(explained_variance)

    
    variance_df = pd.DataFrame(variance_dict, index=[f"Factor {i + 1}" for i in range(len(next(iter(variance_dict.values()))))])
    return variance_df


def normalize_variance_across_modalities(variance_df):
    normalized_df = variance_df.div(variance_df.sum(axis=0), axis=1)
    return normalized_df



variance_df = compute_variance_explained(
    datasets=datasets,
    shared_latents=shared_latents,  
    style_latents=style_latents,  
    decoders=decoders  
)

variance_df = normalize_variance_across_modalities(variance_df)



def plot_variance_heatmap(variance_df):
    plt.figure(figsize=(20, 8))
    sns.heatmap(
        variance_df.T,
        annot=True,
        fmt=".4f",
        cmap="Blues",
        cbar_kws={'label': 'Proportion of Variance Explained'},
        xticklabels=[f"Factor {i + 1}" for i in range(variance_df.shape[0])],
        yticklabels=variance_df.columns
    )
    plt.title("Variance Explained by Latent Factors", fontsize=16)
    plt.xlabel("Latent Factors", fontsize=14)
    plt.ylabel("Modalities", fontsize=14)
    plt.tight_layout()
    plt.show()



print(variance_df.shape)
print(variance_df)


plot_variance_heatmap(variance_df)