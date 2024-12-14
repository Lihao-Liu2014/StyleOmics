import os
import numpy as np
import pandas as pd
from mofapy2.run.entry_point import entry_point
import h5py


data_dir = '/Users/rascalpel/Desktop/Datasets for the final project/processed/BRCA'
output_dir = '/Users/rascalpel/Desktop/Datasets for the final project/latent_space'
latent_space_output_path = os.path.join(output_dir, "mofa_unified_latent_space.npy")
os.makedirs(output_dir, exist_ok=True)

modalities = ["methylation", "mutation", "transcriptomics", "proteomics"]
num_complete_patients = 65  



def load_data_for_mofa(data_dir, modalities, num_patients):
    data_list = {}
    for modality in modalities:
        file_path = os.path.join(data_dir, f"{modality}_processed.csv")
        if os.path.exists(file_path):
            
            data = pd.read_csv(file_path, index_col=0).T.iloc[:num_patients, :]  
            print(f"{modality}: Loaded with shape {data.shape}")

            
            

            
            data_list[modality] = data
        else:
            print(f"File not found for modality: {modality}")

    return data_list


data_list = load_data_for_mofa(data_dir, modalities, num_complete_patients)


for mod, data in data_list.items():
    print(f"{mod} shape: {data.shape}")



def run_mofa(data_list, output_dir):
    
    ent = entry_point()

    
    for modality, data in data_list.items():
        print(f"Adding data for modality: {modality}, shape: {data.shape}")
        ent.set_data_matrix(data.values)

    
    ent.set_model_options(factors=16)
    ent.set_train_options(iter=1000, convergence_mode="fast")

    
    print("Building and training the MOFA model...")
    ent.build()
    ent.run()

    
    model_path = os.path.join(output_dir, "trained_mofa_model.hdf5")
    print(f"Saving trained model to {model_path}")
    ent.save(model_path)

    
    print("Available attributes and methods in 'ent.model':")
    print(dir(ent.model))

    
    latent_factors = ent.model.getExpectations("Z")
    
    unified_latent_space = latent_factors["Z"]  

    
    print(f"Unified latent space shape: {unified_latent_space.shape}")  

    
    np.save(os.path.join(output_dir, "mofa_unified_latent_space.npy"), unified_latent_space)
    print(f"Unified latent space saved to {os.path.join(output_dir, 'mofa_unified_latent_space.npy')}")






run_mofa(data_list, output_dir)
model_path = os.path.join(output_dir, "trained_mofa_model.hdf5")


print("MOFA integration completed successfully.")