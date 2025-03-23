import os
import json
import torch
from Bio.PDB import MMCIFParser, Polypeptide
from transformers import BertTokenizer, BertModel
import numpy as np

class ProteinEmbedding:
    def __init__(self, model_name, tokenizer_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)

    def get_embedding(self, sequence):
        # Add spaces between amino acids and disable special tokens
        sequence = ' '.join(list(sequence))
        encoded_input = self.tokenizer(
            sequence,
            return_tensors='pt',
            padding=False,
            truncation=True,
            max_length=len(sequence),
            add_special_tokens=False
        )
        encoded_input = encoded_input.to(self.device)
        with torch.no_grad():
            output = self.model(**encoded_input)
        return output  # Return the sequence of hidden states

def load_data_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def calculate_distances(embeddings, idx1, idx2):
    emb1 = embeddings[idx1]
    emb2 = embeddings[idx2]
    euclidean_distance = torch.norm(emb1 - emb2).item()
    cosine_similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()
    return euclidean_distance, cosine_similarity

def process_pairs(protein_id, pairs_list, mmcif_dir, embedding_model):
    mmcif_path = os.path.join(mmcif_dir, f"{protein_id}.cif")
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(protein_id, mmcif_path)
    except Exception as e:
        print(f"Error parsing MMCIF file for protein {protein_id}: {e}")
        return []

    first_chain = next(structure.get_chains())

    # Extract sequence using Polypeptide builder
    ppb = Polypeptide.PPBuilder()
    polypeptides = ppb.build_peptides(first_chain)
    if not polypeptides:
        print(f"No polypeptides found in chain for protein {protein_id}")
        return []
    sequence = str(polypeptides[0].get_sequence())

    # Get embeddings
    try:
        output = embedding_model.get_embedding(sequence)
        embeddings = output['last_hidden_state'][0]  # Shape: [sequence_length, hidden_size]
    except Exception as e:
        print(f"Error generating embeddings for protein {protein_id}: {e}")
        return []

    # Ensure embeddings length matches sequence length
    if embeddings.shape[0] != len(sequence):
        print(f"Mismatch in sequence length and embeddings length for protein {protein_id}")
        return []

    results = []
    for pair in pairs_list:
        idx1, idx2 = pair['i'] - 1, pair['j'] - 1  # Adjust for 0-indexing

        # Validate indices
        if idx1 < 0 or idx1 >= len(sequence) or idx2 < 0 or idx2 >= len(sequence):
            print(f"Invalid indices for protein {protein_id}: idx1={idx1+1}, idx2={idx2+1}")
            continue

        # Validate residue types
        residue1_in_sequence = sequence[idx1]
        residue2_in_sequence = sequence[idx2]
        if residue1_in_sequence != pair['residue1'] or residue2_in_sequence != pair['residue2']:
            print(f"Residue mismatch at indices {idx1+1} and {idx2+1} in protein {protein_id}")
            continue

        # Calculate distances
        euclidean_dist, cosine_sim = calculate_distances(embeddings, idx1, idx2)
        results.append({
            'protein_id': protein_id,
            'residue1_index': pair['i'],
            'residue2_index': pair['j'],
            'residue1_type': pair['residue1'],
            'residue2_type': pair['residue2'],
            'euclidean_distance': euclidean_dist,
            'cosine_similarity': cosine_sim
        })
    return results

def main():
    json_file_path = 'protein_residue_distances.json'  # Path to JSON file containing the pairs
    mmcif_directory = 'proteins'  # Directory containing MMCIF files
    data = load_data_from_json(json_file_path)
import os
import json
import torch
from Bio.PDB import MMCIFParser, Polypeptide
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ProteinEmbedding:
    def __init__(self, model_name, tokenizer_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)

    def get_embedding(self, sequence):
        # Add spaces between amino acids and disable special tokens
        sequence = ' '.join(list(sequence))
        encoded_input = self.tokenizer(
            sequence,
            return_tensors='pt',
            padding=False,
            truncation=True,
            max_length=len(sequence),
            add_special_tokens=False
        )
        encoded_input = encoded_input.to(self.device)
        with torch.no_grad():
            output = self.model(**encoded_input)
        return output  # Return the sequence of hidden states

def load_data_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def calculate_distances(embeddings, idx1, idx2):
    emb1 = embeddings[idx1]
    emb2 = embeddings[idx2]
    euclidean_distance = torch.norm(emb1 - emb2).item()
    cosine_similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()
    return euclidean_distance, cosine_similarity

def process_pairs(protein_id, pairs_list, mmcif_dir, embedding_model):
    mmcif_path = os.path.join(mmcif_dir, f"{protein_id}.cif")
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(protein_id, mmcif_path)
    except Exception as e:
        print(f"Error parsing MMCIF file for protein {protein_id}: {e}")
        return []

    first_chain = next(structure.get_chains())

    # Extract sequence using Polypeptide builder
    ppb = Polypeptide.PPBuilder()
    polypeptides = ppb.build_peptides(first_chain)
    if not polypeptides:
        print(f"No polypeptides found in chain for protein {protein_id}")
        return []
    sequence = str(polypeptides[0].get_sequence())

    # Get embeddings
    try:
        output = embedding_model.get_embedding(sequence)
        embeddings = output['last_hidden_state'][0]  # Shape: [sequence_length, hidden_size]
    except Exception as e:
        print(f"Error generating embeddings for protein {protein_id}: {e}")
        return []

    # Ensure embeddings length matches sequence length
    if embeddings.shape[0] != len(sequence):
        print(f"Mismatch in sequence length and embeddings length for protein {protein_id}")
        return []

    results = []
    for pair in pairs_list:
        idx1, idx2 = pair['i'] - 1, pair['j'] - 1  # Adjust for 0-indexing

        # Validate indices
        if idx1 < 0 or idx1 >= len(sequence) or idx2 < 0 or idx2 >= len(sequence):
            print(f"Invalid indices for protein {protein_id}: idx1={idx1+1}, idx2={idx2+1}")
            continue

        # Validate residue types
        residue1_in_sequence = sequence[idx1]
        residue2_in_sequence = sequence[idx2]
        if residue1_in_sequence != pair['residue1'] or residue2_in_sequence != pair['residue2']:
            print(f"Residue mismatch at indices {idx1+1} and {idx2+1} in protein {protein_id}")
            continue

        # Calculate distances
        euclidean_dist, cosine_sim = calculate_distances(embeddings, idx1, idx2)

        # Get sequence distance and 3D distance from pair data
        sequence_distance = abs(pair['i'] - pair['j'])
        distance_3d = pair.get('3d_distance', None)

        # Get the existing category from the pair data
        category = pair.get('category', 'unknown')

        results.append({
            'protein_id': protein_id,
            'residue1_index': pair['i'],
            'residue2_index': pair['j'],
            'residue1_type': pair['residue1'],
            'residue2_type': pair['residue2'],
            'sequence_distance': sequence_distance,
            '3d_distance': distance_3d,
            'euclidean_distance': euclidean_dist,
            'cosine_similarity': cosine_sim,
            'category': category
        })
    return results

def main():
    json_file_path = 'protein_residue_distances.json'  # Path to JSON file containing the pairs
    mmcif_directory = 'proteins'  # Directory containing MMCIF files
    data = load_data_from_json(json_file_path)

    embedding_model = ProteinEmbedding('Rostlab/prot_bert', 'Rostlab/prot_bert')

    all_results = []
    for protein_id, details in data.items():
        print(f"Processing protein {protein_id}")
        pairs_list = details.get('pairs', [])
        results = process_pairs(protein_id, pairs_list, mmcif_directory, embedding_model)
        all_results.extend(results)

    # Save the results to a JSON file
    with open('embedding_distances_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)

    # Optional: Analyze and plot the results
    analyze_results(all_results)

def analyze_results(all_results):
    # Convert the results to a pandas DataFrame
    df = pd.DataFrame(all_results)

    # Drop entries without 3D distance
    df = df.dropna(subset=['euclidean_dist'])

    # Ensure 'category' column exists
    if 'category' not in df.columns:
        print("No 'category' column found in data.")
        return

    # Plot embedding Euclidean distance vs. 3D distance, colored by existing category
    plt.figure(figsize=(10, 6))
    categories = df['category'].unique()
    for category in categories:
        subset = df[df['category'] == category]
        plt.scatter(subset['3d_distance'], subset['euclidean_distance'], label=category, alpha=0.5)
    plt.xlabel('3D Distance (Å)')
    plt.ylabel('Embedding Euclidean Distance')
    plt.title('Embedding Distance vs. 3D Distance')
    plt.legend()
    plt.savefig('embedding_vs_3d_distance.png')
    plt.show()

    # Additional analysis can be performed here

if __name__ == "__main__":
    main()

    embedding_model = ProteinEmbedding('Rostlab/prot_bert', 'Rostlab/prot_bert')

    all_results = []
    for protein_id, details in data.items():
        print(f"Processing protein {protein_id}")
        far_results = process_pairs(protein_id, details.get('far_in_1d_far_in_3d', []), "Projects\isp\proteins", embedding_model)
        far1d_close3d_results = process_pairs(protein_id, details.get('far_in_1d_close_in_3d', []), "Projects\isp\proteins", embedding_model)
        all_results.append({
            'protein_id': protein_id,
            'far_in_1d_far_in_3d': far_results,
            'far_in_1d_close_in_3d': far1d_close3d_results
        })

    # Output the results to a file
    with open('pairwise_distances_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    main()
