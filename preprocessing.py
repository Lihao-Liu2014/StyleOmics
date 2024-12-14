import os
import pandas as pd
import numpy as np
from utils.normalization import log_transform, z_score_normalize


def preprocess_multiomics_data(input_dir, output_dir):
    
    file_extensions = {
        "methylation": "cct",
        "mutation": "cbt",
        "transcriptomics": "cct",
        "proteomics": "cct",
        "phenotype": "tsi"
    }

    modalities = ["methylation", "mutation", "transcriptomics", "proteomics", "phenotype"]
    data_dict = {}

    
    
    all_patient_ids = set()
    patient_ids_per_modality = {}

    for modality in modalities:
        file_extension = file_extensions[modality]
        file_path = os.path.join(input_dir, f"{modality}.{file_extension}")
        try:
            if modality == "phenotype":
                
                data = preprocess_phenotype(file_path)
            else:
                
                sep = "\t" if file_extension in ["cct", "cbt"] else ","
                data = pd.read_csv(file_path, sep=sep)

                print(f"Loaded {modality} data with shape: {data.shape}")

                  
                  
                non_nan_mask = ~data.iloc[:, 1:].isna().any(axis=0)  
                filtered_data = pd.concat([data.iloc[:, 0], data.iloc[:, 1:].loc[:, non_nan_mask]], axis=1)
                print(
                    f"Filtered {modality} data: Removed {data.shape[1] - filtered_data.shape[1]} features with NaN values.")

                if filtered_data.iloc[:, 1:].isna().any().any():
                    print(f"Warning: {modality} still contains NaN values after filtering!")
                else:
                    print(f"{modality}: No NaN values found after filtering.")

            
            patient_ids_per_modality[modality] = set(data.columns[1:])  
            all_patient_ids.update(data.columns[1:])
            data_dict[modality] = data
        except Exception as e:
            print(f"Error loading {modality} data: {e}")
            continue

    print(f"Total unique patient IDs across all modalities: {len(all_patient_ids)}")

    
    common_patient_ids = set.intersection(*patient_ids_per_modality.values())
    print(f"Number of patients with data available for all modalities: {len(common_patient_ids)}")

    
    all_patient_ids = list(common_patient_ids) + [pid for pid in all_patient_ids if pid not in common_patient_ids]

    
    for modality in modalities:
        if modality in data_dict:
            print(f"\nAligning {modality} data to include all patient IDs...")
            data = data_dict[modality]
            attrib_name = data.iloc[:, 0]

            
            print(f"Validating common_patient_ids for {modality}...")
            missing_ids = [pid for pid in common_patient_ids if pid not in data.columns]
            if missing_ids:
                print(f"Warning: {modality} is missing these common_patient_ids: {missing_ids}")
            else:
                print(f"All common_patient_ids are present in {modality}.")

            
            aligned_columns = [
                pd.Series(data[patient_id].values, name=patient_id) if patient_id in data.columns else None
                for patient_id in all_patient_ids
            ]

            
            aligned_columns = [col for col in aligned_columns if col is not None]

            
            aligned_data = pd.concat([pd.DataFrame(attrib_name, columns=["attrib_name"])] + aligned_columns, axis=1)
            aligned_data = aligned_data.fillna(0)
            aligned_data.columns = ["attrib_name"] + [col.name for col in aligned_columns]


            
            data_dict[modality] = aligned_data
            print(f"Aligned {modality} data shape: {aligned_data.shape}")

    for modality in modalities:
        aligned_data = data_dict[modality]
        if aligned_data.iloc[:, 1:].isna().any().any():
            print(f"Warning: NaN values found in aligned {modality} data!")
        else:
            print(f"No NaN values in aligned {modality} data.")

    
    num_complete_patients = len(common_patient_ids)
    print("\nChecking if the first {num_complete_patients} patients have no NaN values across all modalities...")
    all_ok = True

    for modality in modalities:
        if modality in data_dict:
            data = data_dict[modality]
            patient_data = data.iloc[:, 1:num_complete_patients + 1]  
            nan_indices = patient_data.isna()
            if nan_indices.any().any():
                all_ok = False
                print(f"\nNaN values found in {modality}:")
                for patient_idx, col_idx in zip(*nan_indices.to_numpy().nonzero()):
                    patient_id = data.columns[1:][col_idx]
                    attrib_name = data.iloc[patient_idx, 0]
                    print(f"  Patient ID = {patient_id}, Modality = {modality},Attribute = {attrib_name}")

    if all_ok:
        print("\nAll first {num_complete_patients} patients' data are complete across all modalities.")
    else:
        print("\nSome NaN values found in the first {num_complete_patients} patients' data. See above for details.")

    
    for modality in ["methylation", "mutation", "transcriptomics", "proteomics"]:
        if modality in data_dict:
            print(f"\nNormalizing {modality} data...")
            data = data_dict[modality]
            attrib_name = data["attrib_name"]
            feature_data = data.iloc[:, 1:]  

            
            feature_data = log_transform(feature_data)
            feature_data = z_score_normalize(feature_data)
            feature_data = pd.DataFrame(feature_data, columns=data.columns[1:], index=data.index)

            
            normalized_data = pd.concat([attrib_name, feature_data], axis=1)
            data_dict[modality] = normalized_data
            print(f"Normalized {modality} data saved with shape: {normalized_data.shape}")

    
    os.makedirs(output_dir, exist_ok=True)
    for modality, data in data_dict.items():
        output_file = os.path.join(output_dir, f"{modality}_processed.csv")
        data.to_csv(output_file, index=False)
        print(f"Saved processed {modality} data to: {output_file}")


def preprocess_phenotype(file_path):
    try:
        
        phenotype_data = pd.read_csv(file_path, sep="\t", index_col=0)
        print(f"Loaded phenotype data with shape: {phenotype_data.shape}")

        
        phenotype_data.reset_index(inplace=True)  
        phenotype_data.rename(columns={"index": "attrib_name"}, inplace=True)

        print("Phenotype data sample:")
        print(phenotype_data.head())

        return phenotype_data
    except Exception as e:
        print(f"Error processing phenotype data: {e}")
        return None



if __name__ == "__main__":
    base_dir = "/Users/rascalpel/Desktop/Datasets for the final project"
    output_dir = os.path.join(base_dir, "processed")
    datasets = ["BRCA", "OV"]

    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        dataset_input_dir = os.path.join(base_dir, dataset)
        dataset_output_dir = os.path.join(output_dir, dataset)
        preprocess_multiomics_data(dataset_input_dir, dataset_output_dir)