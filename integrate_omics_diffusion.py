import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


data_dir = '/Users/rascalpel/Desktop/Datasets for the final project/processed/BRCA'
output_dir = "/Users/rascalpel/Desktop/Datasets for the final project/latent_space"
modalities = ["methylation", "mutation", "transcriptomics", "proteomics"]
num_complete_patients = 65


def load_data(data_dir, modalities):
    datasets = {}
    input_dims = {}
    for modality in modalities:
        file_path = os.path.join(data_dir, f"{modality}_processed.csv")
        if os.path.exists(file_path):
            data = pd.read_csv(file_path, index_col=0).T  
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(data.values)
            datasets[modality] = torch.tensor(normalized_data, dtype=torch.float32)
            input_dims[modality] = datasets[modality].shape[1]
            print(f"{modality}: Loaded with shape {datasets[modality].shape}")
        else:
            print(f"File not found: {file_path}")
    return datasets, input_dims

datasets, input_dims = load_data(data_dir, modalities)

class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, shared_dim, style_dim):
        super(ModalityEncoder, self).__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, shared_dim)
        )
        self.style_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, style_dim)
        )

    def forward(self, x):
        z_shared = self.shared_encoder(x)
        z_style = self.style_encoder(x)
        return z_shared, z_style

class ModalityDecoder(nn.Module):
    def __init__(self, shared_dim, style_dim, output_dim):
        super(ModalityDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(shared_dim + style_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, z_shared, z_style):
        z_combined = torch.cat([z_shared, z_style], dim=-1)
        return self.decoder(z_combined)


class CrossModalityAttention(nn.Module):
    def __init__(self, shared_dim, num_modalities):
        super(CrossModalityAttention, self).__init__()
        self.query = nn.Linear(shared_dim, shared_dim)
        self.key = nn.Linear(shared_dim, shared_dim)
        self.value = nn.Linear(shared_dim, shared_dim)
        self.num_modalities = num_modalities
        self.layer_norm = nn.LayerNorm(shared_dim)

        
        self.init_weights()

    def init_weights(self):
        for layer in [self.query, self.key, self.value]:
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

    def forward(self, shared_latents):
        """
        Args:
            shared_latents (torch.Tensor): Shape (num_modalities, batch_size, shared_dim)
        Returns:
            unified_latent (torch.Tensor): Shape (batch_size, shared_dim)
            attention_scores (torch.Tensor): Shape (num_modalities, num_modalities, batch_size)
        """
        
        Q = self.query(shared_latents) / (shared_latents.size(-1) ** 0.5)
        K = self.key(shared_latents) / (shared_latents.size(-1) ** 0.5)
        V = self.value(shared_latents)

        
        logits = torch.matmul(Q, K.transpose(-2, -1))
        logits = logits - logits.max(dim=-1, keepdim=True).values  
        attention_scores = torch.softmax(logits, dim=-1)

        
        attended_values = torch.matmul(attention_scores, V)

        
        residual = self.layer_norm(shared_latents.mean(dim=0))  
        unified_latent = self.layer_norm(attended_values.mean(dim=0) + residual)

        return unified_latent, attention_scores

def reconstruction_loss(recon, original):
    return nn.MSELoss()(recon, original)

def cycle_consistency_loss(encoders, decoders, datasets, modalities):
    """
    Compute cycle consistency loss for all modality pairs using both shared and style latents.
    """
    loss = 0
    for mod_i in modalities:
        for mod_j in modalities:
            if mod_i != mod_j:
                
                data_i = datasets[mod_i].to(device)
                z_shared_i, z_style_i = encoders[mod_i](data_i)  

                
                z_shared_j, z_style_j = torch.zeros_like(z_shared_i), torch.zeros_like(z_shared_i)
                if mod_j in datasets:
                    data_j = datasets[mod_j].to(device)
                    z_shared_j, z_style_j = encoders[mod_j](data_j)

                recon_cross = decoders[mod_j](z_shared_i, z_style_j)  

                
                z_shared_j_prime, z_style_j_prime = encoders[mod_j](recon_cross)

                
                recon_cycle = decoders[mod_i](z_shared_j_prime, z_style_i)  

                
                loss += reconstruction_loss(recon_cycle, data_i)  
    return loss


def cross_modality_loss(shared_latents):
    loss = 0
    for i in range(len(shared_latents)):
        for j in range(i + 1, len(shared_latents)):
            loss += torch.mean((shared_latents[i] - shared_latents[j]) ** 2)
    return loss

def attention_sparsity_loss(attention_weights):
    mean_weights = torch.ones_like(attention_weights) / attention_weights.size(-1)
    return torch.mean((attention_weights - mean_weights) ** 2)

def style_disentanglement_loss(shared_latents, style_latents):
    loss = 0
    for z_shared, z_style in zip(shared_latents, style_latents):
        loss += torch.mean((z_shared * z_style) ** 2)
    return loss

def attention_entropy_loss(attention_weights):
    loss = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8)) / attention_weights.size(0)
    return loss

def latent_sparsity_loss(unified_latent):
    return torch.mean(torch.abs(unified_latent))

def cross_modality_loss(shared_latents, margin=1.0):
    loss = 0
    for i in range(len(shared_latents)):
        for j in range(i + 1, len(shared_latents)):
            diff = torch.norm(shared_latents[i] - shared_latents[j], p=2, dim=-1)
            loss += torch.mean((diff - margin).clamp(min=0) ** 2)
    return loss

def style_disentanglement_loss(shared_latents, style_latents):
    loss = 0
    for z_shared, z_style in zip(shared_latents, style_latents):
        loss += torch.mean((z_shared @ z_style.T) ** 2)
    return loss

def latent_sparsity_loss(z_style):
    return torch.mean(torch.abs(z_style))


class ModalityDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(ModalityDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def vib_loss(shared_latents):
    mean_latents = torch.mean(shared_latents, dim=0)
    variance_loss = torch.mean((shared_latents - mean_latents) ** 2)
    return variance_loss


def adversarial_loss(pred, target):
    return nn.BCELoss()(pred, target)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)





def compute_dwa(prev_losses, curr_losses, temp=2.0):
    
    speeds = [curr / max(prev, 1e-8) for curr, prev in zip(curr_losses, prev_losses)]
    weights = torch.softmax(torch.tensor(speeds) / temp, dim=0)  
    return weights.tolist()



if __name__ == "__main__":


    
    shared_dim = 16
    style_dim = 16
    epochs = 3000
    batch_size = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for mod in modalities:
        datasets[mod] = datasets[mod].T[:, :num_complete_patients].T  
        print(f"{mod}: Truncated dataset shape = {datasets[mod].shape}")

    for mod in modalities:
        data = datasets[mod]
        print(f"{mod}: Has NaN: {torch.isnan(data).any().item()}")

    
    for mod in modalities:
        data = datasets[mod]
        mean_all = data.mean().item()  
        std_all = data.std().item()    

        print(f"{mod}: Mean (overall): {mean_all:.4f}, Std (overall): {std_all:.4f}")


    lambda_rec = 1.0
    lambda_cross = 0.01
    lambda_cycle = 0.1
    lambda_disentangle = 0.001
    lambda_latent_sparsity = 0.1
    
    
    lambda_rec = 10.1  
    lambda_cross = 1.1  
    lambda_cycle = 10  
    lambda_disentangle = 0.1  
    lambda_att_sparsity = 0.1  
    lambda_att_entropy = 0.07  
    lambda_latent_sparsity = 2  







    
    encoders = {mod: ModalityEncoder(input_dims[mod], shared_dim, style_dim).to(device) for mod in modalities}
    decoders = {mod: ModalityDecoder(shared_dim, style_dim, input_dims[mod]).to(device) for mod in modalities}
    attention = CrossModalityAttention(shared_dim, len(modalities)).to(device)
    discriminators = {mod: ModalityDiscriminator(input_dims[mod]).to(device) for mod in modalities}

    for mod in modalities:
        encoders[mod].apply(init_weights)
        decoders[mod].apply(init_weights)
        attention.apply(init_weights)


    
    optimizers = {
        mod: torch.optim.Adam(list(encoders[mod].parameters()) + list(decoders[mod].parameters()), lr=3e-6)
        for mod in modalities
    }
    attention_optimizer = torch.optim.Adam(attention.parameters(), lr=3e-6)
    discriminator_optimizers = {
        mod: torch.optim.Adam(discriminators[mod].parameters(), lr=1e-5) for mod in modalities
    }


    torch.autograd.set_detect_anomaly(True)
    
    
    prev_losses = [lambda_rec, lambda_cross, lambda_cycle, lambda_disentangle, lambda_latent_sparsity]  
    print(f"Initial Loss Weights: {prev_losses}")
    all_shared_latents = {mod: [] for mod in modalities}
    all_style_latents = {mod: [] for mod in modalities}
    shared_latents_final = {mod: None for mod in modalities}
    style_latents_final = {mod: None for mod in modalities}
    import matplotlib.pyplot as plt

    
    total_losses = []
    rec_losses = []
    cross_losses = []
    cycle_losses = []
    disentangle_losses = []
    att_sparsity_losses = []
    att_entropy_losses = []
    latent_sparsity_losses = []
    vib_losses = []

    for epoch in range(epochs):
        total_loss = 0
        shared_latents = []
        style_latents = []
        curr_losses = []

        for mod in modalities:
            data = datasets[mod].to(device)

            
            z_shared, z_style = encoders[mod](data)
            style_latents.append(z_style)
            shared_latents.append(z_shared.unsqueeze(0))  

        
        shared_latents = torch.stack(shared_latents)  
        unified_latent, attention_weights = attention(shared_latents)  
        
        unified_latent = unified_latent.squeeze(0)  

        
        for mod in modalities:
            z_style = style_latents[modalities.index(mod)]  

            
            recon = decoders[mod](unified_latent, z_style)  
            data = datasets[mod].to(device)

            
            rec_loss = reconstruction_loss(recon, data)

            
            total_loss += lambda_rec * rec_loss
            curr_losses.append(rec_loss.item())


        
        if epoch > epochs // 2:
            cross_loss = cross_modality_loss(shared_latents)
            total_loss += lambda_cross * cross_loss
            curr_losses.append(cross_loss.item())  
            cycle_loss = cycle_consistency_loss(encoders, decoders, datasets, modalities)
            total_loss += lambda_cycle * cycle_loss
            curr_losses.append(cycle_loss.item())  

        else:
            curr_losses.append(0.0)
            curr_losses.append(0.0)  


        
        vib_regularization = vib_loss(shared_latents)
        total_loss += vib_regularization
        curr_losses.append(vib_regularization.item())  

        
        disentangle_loss = style_disentanglement_loss(shared_latents, style_latents)
        total_loss += lambda_disentangle * disentangle_loss

        
        att_sparsity_loss = attention_sparsity_loss(attention_weights)
        att_entropy_loss = attention_entropy_loss(attention_weights)
        total_loss += lambda_att_sparsity * att_sparsity_loss + lambda_att_entropy * att_entropy_loss

        
        latent_sparsity = latent_sparsity_loss(shared_latents.mean(dim=0))
        total_loss += lambda_latent_sparsity * latent_sparsity

        
        total_loss.backward()

        for mod in modalities:
            for name, param in encoders[mod].named_parameters():
                if param.grad is not None:
                    
                    torch.nn.utils.clip_grad_norm_(param, max_norm=1.0)

        
        for mod in modalities:
            optimizers[mod].step()
        attention_optimizer.step()

        
        loss_weights = compute_dwa(prev_losses[:5], curr_losses[:5])
        print(f"Dynamic Loss Weights: {lambda_rec, lambda_cross, lambda_cycle, lambda_disentangle, lambda_latent_sparsity}")

        total_losses.append(total_loss.item())
        rec_losses.append(rec_loss.item())
        if epoch > epochs // 2:
            cross_losses.append(cross_loss.item())
            cycle_losses.append(cycle_loss.item())
        else:
            cross_losses.append(0.0)
            cycle_losses.append(0.0)
        disentangle_losses.append(disentangle_loss.item())
        att_sparsity_losses.append(att_sparsity_loss.item())
        att_entropy_losses.append(att_entropy_loss.item())
        latent_sparsity_losses.append(latent_sparsity.item())
        vib_losses.append(vib_regularization.item())

        
        
        

        
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Total Loss: {total_loss.item():.4f}, "
              f"Rec Loss: {rec_loss.item():.4f}, "
              f"Cross Loss: {cross_loss.item() if epoch > epochs // 2 else 0:.4f}, "
              f"Cycle Loss: {cycle_loss.item() if epoch > epochs // 2 else 0:.4f}, "
              f"Disentangle Loss: {disentangle_loss.item():.4f}, "
              f"Att Sparsity Loss: {att_sparsity_loss.item():.4f}, "
              f"Att Entropy Loss: {att_entropy_loss.item():.4f}, "
              f"Latent Sparsity Loss: {latent_sparsity.item():.4f}, "
              f"VIB Loss: {vib_regularization.item():.4f}")

        for mod in modalities:
            data = datasets[mod].to(device)

            
            z_shared, z_style = encoders[mod](data)

            
            shared_latents_final[mod] = z_shared.detach().cpu().numpy()
            style_latents_final[mod] = z_style.detach().cpu().numpy()

        
        prev_losses = curr_losses

    
    unified_latent_space = unified_latent.cpu().detach().numpy()
    np.save(os.path.join(output_dir, "unified_latent_space.npy"), unified_latent_space)
    print("Unified latent space saved.")

    import os
    import torch
    import numpy as np

    
    output_dir = "/Users/rascalpel/Desktop/Datasets for the final project/saved_components"
    os.makedirs(output_dir, exist_ok=True)

    
    shared_latents_np = shared_latents.cpu().detach().numpy()
    latent_factors_path = os.path.join(output_dir, "shared_latents.npy")
    np.save(latent_factors_path, shared_latents_np)
    print(f"Shared latent factors saved to: {latent_factors_path}")

    
    decoder_weights_path = os.path.join(output_dir, "decoders.pt")
    torch.save({modality: decoder.state_dict() for modality, decoder in decoders.items()}, decoder_weights_path)
    print(f"Decoders saved to: {decoder_weights_path}")

    for mod, latents in shared_latents_final.items():
        np.save(os.path.join(output_dir, f"{mod}_shared_latents.npy"), latents)

    
    for mod, latents in style_latents_final.items():
        np.save(os.path.join(output_dir, f"{mod}_style_latents.npy"), latents)

    print(f"Latents saved to {output_dir}")

    for mod in modalities:
        shared_latents = np.load(os.path.join(output_dir, f"{mod}_shared_latents.npy"))
        style_latents = np.load(os.path.join(output_dir, f"{mod}_style_latents.npy"))
        print(f"{mod} Shared Latents Shape: {shared_latents.shape}")
        print(f"{mod} Style Latents Shape: {style_latents.shape}")

    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))  

    
    losses = [
        (total_losses, "Total Loss"),
        (rec_losses, "Reconstruction Loss"),
        (cross_losses, "Cross-Modality Loss"),
        (cycle_losses, "Cycle Consistency Loss"),
        (disentangle_losses, "Style Disentanglement Loss"),
        (att_sparsity_losses, "Attention Sparsity Loss"),
        (att_entropy_losses, "Attention Entropy Loss"),
        (latent_sparsity_losses, "Latent Sparsity Loss"),
        (vib_losses, "VIB Loss"),
    ]

    
    for i, (loss, label) in enumerate(losses):
        row, col = divmod(i, 3)
        ax = axes[row, col]
        ax.plot(loss, label=label, color="blue")
        ax.set_title(label)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss Value")
        ax.grid(True)
        ax.legend()

    
    plt.tight_layout()

    
    plt.savefig(os.path.join(output_dir, "loss_trends_grid.png"))
    print(f"Loss trend grid plot saved to {output_dir}/loss_trends_grid.png")

    
    plt.show()