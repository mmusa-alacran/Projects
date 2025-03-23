import os
import random
import argparse
import json
from sequence_embedding import SequenceEmbedding

def select_unique_random_mutation(current_sequence, used_positions, used_amino_acids):
    """ Selects a random mutation not previously used at a new position. """
    amino_acids = 'ARNDCEQGHILKMFPSTWYVBUZ'  # Including some ambiguous or rare amino acids like B, Z, U
    possible_positions = [i for i, aa in enumerate(current_sequence) if i not in used_positions]
    
    if not possible_positions:
        raise ValueError("No more positions available for mutation.")
    
    mutation_position = random.choice(possible_positions)
    current_aa = current_sequence[mutation_position]
    possible_amino_acids = [aa for aa in amino_acids if aa != current_aa and aa not in used_amino_acids[mutation_position]]
    
    if not possible_amino_acids:
        raise ValueError(f"No more unique amino acids available for mutation at position {mutation_position}.")
    
    new_aa = random.choice(possible_amino_acids)
    return new_aa, mutation_position

def process_file(file_path, transformer, iterations):
    sequence_embedder = SequenceEmbedding(transformer)
    original_sequence = sequence_embedder.load_sequence(file_path)
    current_sequence = original_sequence
    original_embedding = sequence_embedder.get_embedding(original_sequence)['last_hidden_state'].squeeze().detach()

    results = []
    used_positions = set()
    used_amino_acids = {i: [aa] for i, aa in enumerate(current_sequence)}  # Tracking amino acids used at each position

    for i in range(iterations):
        new_aa, mutation_position = select_unique_random_mutation(current_sequence, used_positions, used_amino_acids)
        current_sequence = current_sequence[:mutation_position] + new_aa + current_sequence[mutation_position + 1:]
        used_positions.add(mutation_position)
        used_amino_acids[mutation_position].append(new_aa)

        mutated_embedding = sequence_embedder.get_embedding(current_sequence)['last_hidden_state'].squeeze().detach()
        euclidean_distance = sequence_embedder.calculate_distance(original_embedding, mutated_embedding, 'euclidean')
        cosine_distance = sequence_embedder.calculate_distance(original_embedding, mutated_embedding, 'cosine')

        results.append({
            'iteration': i + 1,
            'original_sequence': original_sequence,
            'mutated_sequence': current_sequence,
            'position': mutation_position,
            'new_amino_acid': new_aa,
            'euclidean_distance': euclidean_distance,
            'cosine_distance': cosine_distance
        })

    return results

def main():
    parser = argparse.ArgumentParser(description='Iterative protein sequence mutation and analysis for multiple files.')
    parser.add_argument('--dir', type=str, required=True, help='Directory of protein files')
    parser.add_argument('--transformer', type=str, required=True, choices=['bert', 'esm2'], help='Transformer model to use')
    parser.add_argument('--iterations', type=int, default=15, help='Number of mutation iterations')
    args = parser.parse_args()

    all_results = {}

    for filename in os.listdir(args.dir):
        if filename.endswith('.cif'):
            file_path = os.path.join(args.dir, filename)
            file_results = process_file(file_path, args.transformer, args.iterations)
            all_results[filename] = file_results

    with open('mutation_results_all.json', 'w') as outfile:
        json.dump(all_results, outfile, indent=4)

if __name__ == "__main__":
    main()
