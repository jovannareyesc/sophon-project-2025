import os
import sys
import torch
import uproot
import numpy as np
from tqdm import tqdm
import csv

sys.path.append(".")
from networks.example_ParticleTransformer_sophon import get_model

# Feature keys
particle_keys = [
    'part_px', 'part_py', 'part_pz', 'part_energy',
    'part_deta', 'part_dphi', 'part_d0val', 'part_d0err',
    'part_dzval', 'part_dzerr', 'part_charge',
    'part_isChargedHadron', 'part_isNeutralHadron',
    'part_isPhoton', 'part_isElectron', 'part_isMuon'
]

scalar_keys = [
    'label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg',
    'label_H4q', 'label_Hqql', 'label_Zqq', 'label_Wqq',
    'label_Tbqq', 'label_Tbl', 'jet_pt', 'jet_eta', 'jet_phi',
    'jet_energy', 'jet_nparticles', 'jet_sdmass', 'jet_tau1',
    'jet_tau2', 'jet_tau3', 'jet_tau4', 'aux_genpart_eta',
    'aux_genpart_phi', 'aux_genpart_pid', 'aux_genpart_pt',
    'aux_truth_match'
]

pf_keys = particle_keys + scalar_keys

root_dir = "../data/JetClass/val_5M"
root_files = ["ZJetsToNuNu_120.root", "ZJetsToNuNu_121.root", "ZJetsToNuNu_122.root",
              "ZJetsToNuNu_123.root", "ZJetsToNuNu_124.root"]

# Dummy config for model
class DummyDataConfig:
    input_dicts = {"pf_features": list(range(37))}
    input_names = ["pf_points"]
    input_shapes = {"pf_points": (128, 37)}
    label_names = ["label"]
    num_classes = 10

data_config = DummyDataConfig()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _ = get_model(data_config, num_classes=data_config.num_classes)
model.eval().to(device)

# Output CSV path
output_csv_path = "ZToNuNu_INFERENCE_08312025csv"

with open(output_csv_path, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    header = [f"P_{i}" for i in range(10)] + ["truth"]
    writer.writerow(header)

    for file_name in root_files:
        print(f"\nRunning inference on: {file_name}")
        file_path = os.path.join(root_dir, file_name)
        with uproot.open(file_path) as f:
            tree = f["tree"]
            arrays = tree.arrays(pf_keys, library="np")

        max_part = 128
        total_events = len(arrays["part_px"])

        for i in tqdm(range(total_events), desc=f"{file_name}"):
            try:
                n_part = arrays["part_px"][i].shape[0]
                if n_part > max_part:
                    continue

                # Build features
                particle_feats = [arrays[k][i] for k in particle_keys]
                scalar_feats = [np.full(n_part, arrays[k][i]) for k in scalar_keys]
                all_feats = particle_feats + scalar_feats
                pf_features = np.stack(all_feats, axis=1)

                # Pad
                padded = np.zeros((max_part, pf_features.shape[1]), dtype=np.float32)
                padded[:n_part, :] = pf_features

                jet_tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(device)
                lorentz_vectors = jet_tensor[:, :, 0:4].transpose(1, 2)
                features = jet_tensor[:, :, 4:].transpose(1, 2)
                mask = (jet_tensor.sum(dim=2) != 0).unsqueeze(1)
                points = None

                with torch.no_grad():
                    logits = model(points, features, lorentz_vectors, mask).squeeze(0)[:10].cpu()
                    probs = torch.softmax(logits, dim=0).numpy()

                # Determine truth label from one-hot
                label_array = np.array([arrays[k][i] for k in [
                    'label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg',
                    'label_H4q', 'label_Hqql', 'label_Zqq', 'label_Wqq',
                    'label_Tbqq', 'label_Tbl'
                ]])
                truth_label = int(np.argmax(label_array))

                # Write only probabilities + truth
                writer.writerow(list(probs) + [truth_label])

            except Exception as e:
                print(f"Error in event {i}: {e}")
                continue

print(f"saved csv data to {output_csv_path}")