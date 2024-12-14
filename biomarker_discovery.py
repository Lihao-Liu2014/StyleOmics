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
        data = pd.read_csv(file_path, index_col=0).T  
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


def discover_biomarkers_with_shared_latents(
    decoders, shared_latents, modalities, data_dir, top_n=10, output_dir="shared_biomarkers"
):
    """
    Discover and visualize top biomarkers using shared latent spaces.

    Args:
        decoders (dict): Decoders for each modality.
        shared_latents (dict): Shared latent spaces for each modality.
        modalities (list): List of modality names.
        data_dir (str): Path to the directory containing modality data.
        top_n (int): Number of top biomarkers to discover.
        output_dir (str): Directory to save the biomarker plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    for modality in modalities:
        try:
            
            z_shared = torch.tensor(shared_latents[modality]).to(torch.float32)

            
            decoder = decoders[modality]
            z_style = torch.zeros_like(z_shared)  
            recon_data = decoder(z_shared, z_style).detach().cpu().numpy()  

            
            dataset_path = os.path.join(data_dir, f"{modality}_processed.csv")
            original_data = pd.read_csv(dataset_path, index_col=0).T.iloc[:num_complete_patients].values  

            if recon_data.shape != original_data.shape:
                raise ValueError(f"Shape mismatch: recon_data {recon_data.shape} vs original_data {original_data.shape}")

            
            total_variance = np.var(recon_data, axis=0, ddof=1)  
            original_variance = np.var(original_data, axis=0, ddof=1)  
            explained_variance = total_variance / (original_variance + 1e-8)  

            
            sorted_indices = np.argsort(-explained_variance)[:top_n]
            top_explained_variance = explained_variance[sorted_indices]

            
            feature_names = pd.read_csv(dataset_path, index_col=0).index[sorted_indices]

            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=top_explained_variance, y=feature_names, palette="Blues_r", orient="h")
            plt.title(f"Top {top_n} Biomarkers (Shared Latents) - {modality.capitalize()}", fontsize=16)
            plt.xlabel("Explained Variance", fontsize=14)
            plt.ylabel("Feature", fontsize=14)
            plt.grid(axis="x", linestyle="--", alpha=0.7)
            plt.tight_layout()

            
            plot_path = os.path.join(output_dir, f"{modality}_shared_biomarkers.png")
            plt.savefig(plot_path, dpi=300)
            plt.show()

        except Exception as e:
            print(f"Error processing {modality}: {e}")


output_dir = "/Users/rascalpel/Desktop/Datasets for the final project/shared_biomarkers"
discover_biomarkers_with_shared_latents(decoders, shared_latents, modalities, data_dir, top_n=10, output_dir=output_dir)