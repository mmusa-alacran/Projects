import os
import torch
import argparse
import json  # Added import for json module
from Bio import PDB
from transformers import BertTokenizer, BertModel
from Bio.PDB.Polypeptide import PPBuilder

# Initialize the tokenizer and model outside of main to avoid reloading
tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)
model = BertModel.from_pretrained('Rostlab/prot_bert')
model.eval()  # Set model to evaluation mode

def get_protein_sequence(mmcif_path):
    parser = PDB.MMCIFParser(QUIET=True)
    structure = parser.get_structure('protein', mmcif_path)
    ppb = PPBuilder()
    sequences = []
    for pp in ppb.build_peptides(structure):
        seq = pp.get_sequence()
        sequences.append(str(seq))
    if sequences:
        # Assuming we're interested in the first polypeptide chain
        return sequences[0]
    else:
        return ''

def get_embeddings(sequence):
    # Add spaces between amino acids for ProtBERT
    sequence_spaced = ' '.join(list(sequence))
    # Remove truncation or specify a max_length to avoid the warning
    tokens = tokenizer(sequence_spaced, return_tensors="pt", padding=True, add_special_tokens=False)
    with torch.no_grad():
        outputs = model(**tokens)
    embeddings = outputs.last_hidden_state.squeeze(0)  # Shape: [sequence_length, hidden_size]
    protein_embedding = torch.mean(embeddings, dim=0)  # Mean pooling
    return embeddings, protein_embedding

def mutate_sequence(sequence, position, mutation):
    sequence = list(sequence)
    sequence[position] = mutation
    return ''.join(sequence)

def calculate_distances(embedding1, embedding2):
    euclidean_distance = torch.norm(embedding1 - embedding2).item()
    cosine_similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0).item()
    return euclidean_distance, cosine_similarity

def main():
    parser = argparse.ArgumentParser(description="Process protein data and compute embeddings.")
    parser.add_argument("--protein_id", type=str, required=True, help="Protein ID for the MMCIF file.")
    parser.add_argument("--pos1", type=int, required=True, help="Position of the first amino acid in the pair (1-based index).")
    parser.add_argument("--pos2", type=int, required=True, help="Position of the second amino acid in the pair (1-based index).")
    parser.add_argument("--mut_pos", type=int, required=True, help="Position to mutate (1-based index).")
    parser.add_argument("--mut_res", type=str, required=True, help="Residue to mutate to (single-letter code).")
    args = parser.parse_args()

    mmcif_dir = 'proteins'
    mmcif_path = os.path.join(mmcif_dir, f"{args.protein_id}.cif")

    if not os.path.exists(mmcif_path):
        print("MMCIF file does not exist.")
        return

    sequence = get_protein_sequence(mmcif_path)
    if not sequence:
        print("Failed to extract protein sequence.")
        return

    sequence = sequence.upper()
    args.mut_res = args.mut_res.upper()

    # Validate positions
    sequence_length = len(sequence)
    if args.pos1 < 1 or args.pos1 > sequence_length or args.pos2 < 1 or args.pos2 > sequence_length:
        print("Position indices are out of range.")
        return
    if args.mut_pos < 1 or args.mut_pos > sequence_length:
        print("Mutation position is out of range.")
        return

    # Get embeddings before mutation
    embeddings_before, protein_embedding_before = get_embeddings(sequence)

    # Mutation
    mutated_sequence = mutate_sequence(sequence, args.mut_pos - 1, args.mut_res)
    embeddings_after, protein_embedding_after = get_embeddings(mutated_sequence)

    # Calculate distances between per-residue embeddings
    emb1_before = embeddings_before[args.pos1 - 1]
    emb2_before = embeddings_before[args.pos2 - 1]
    emb1_after = embeddings_after[args.pos1 - 1]
    emb2_after = embeddings_after[args.pos2 - 1]

    dist_residue_before = calculate_distances(emb1_before, emb2_before)
    dist_residue_after = calculate_distances(emb1_after, emb2_after)

    # Calculate distances between per-residue embeddings before and after mutation
    if args.pos1 == args.mut_pos:
        emb_mutated = emb1_after
        emb_original = emb1_before
        non_mutated_residue_index = args.pos2 - 1
    elif args.pos2 == args.mut_pos:
        emb_mutated = emb2_after
        emb_original = emb2_before
        non_mutated_residue_index = args.pos1 - 1
    else:
        emb_mutated = embeddings_after[args.mut_pos - 1]
        emb_original = embeddings_before[args.mut_pos - 1]
        non_mutated_residue_index = None  # Neither residue in the pair was mutated

    dist_residue_mutation = calculate_distances(emb_original, emb_mutated)

    # Calculate distances between the non-mutated residue embeddings before and after mutation (if applicable)
    if non_mutated_residue_index is not None:
        emb_non_mutated_before = embeddings_before[non_mutated_residue_index]
        emb_non_mutated_after = embeddings_after[non_mutated_residue_index]
        dist_non_mutated_residue = calculate_distances(emb_non_mutated_before, emb_non_mutated_after)
    else:
        dist_non_mutated_residue = None

    # Calculate distances between protein embeddings before and after mutation
    dist_protein = calculate_distances(protein_embedding_before, protein_embedding_after)

    # Output results
    output = {
        'protein_id': args.protein_id,
        'amino_acid_pair': {
            'residue1': sequence[args.pos1 - 1],
            'position1': args.pos1,
            'residue2': sequence[args.pos2 - 1],
            'position2': args.pos2
        },
        'mutation': {
            'original_residue': sequence[args.mut_pos - 1],
            'position': args.mut_pos,
            'mutated_residue': args.mut_res
        },
        'distances': {
            'residue_embeddings_before': {
                'euclidean_distance': dist_residue_before[0],
                'cosine_similarity': dist_residue_before[1]
            },
            'residue_embeddings_after': {
                'euclidean_distance': dist_residue_after[0],
                'cosine_similarity': dist_residue_after[1]
            },
            'residue_embedding_mutation_change': {
                'euclidean_distance': dist_residue_mutation[0],
                'cosine_similarity': dist_residue_mutation[1]
            },
            'protein_embedding_change': {
                'euclidean_distance': dist_protein[0],
                'cosine_similarity': dist_protein[1]
            }
        }
    }

    if dist_non_mutated_residue is not None:
        output['distances']['non_mutated_residue_change'] = {
            'euclidean_distance': dist_non_mutated_residue[0],
            'cosine_similarity': dist_non_mutated_residue[1]
        }

    # Print output
    print(json.dumps(output, indent=4))

if __name__ == "__main__":
    main()
