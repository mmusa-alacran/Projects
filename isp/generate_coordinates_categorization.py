import os
import json
import numpy as np
from Bio.PDB import MMCIFParser, PDBExceptions
from Bio.SeqUtils import IUPACData

def calculate_distance(coord1, coord2):
    """Calculate the Euclidean distance between two 3D points, rounded to two decimal places."""
    coord1 = np.array(coord1)
    coord2 = np.array(coord2)
    return round(np.sqrt(np.sum((coord1 - coord2) ** 2)), 2)

def three_to_one(residue_name):
    """Convert three-letter amino acid code to one-letter code."""
    return IUPACData.protein_letters_3to1.get(residue_name.capitalize(), 'X')

def classify_pairs(coordinates, distance_matrix, max_pairs):
    np_matrix = np.array(distance_matrix)
    mean_distance = round(np.mean(np_matrix[np.triu_indices_from(np_matrix, 1)]), 2)
    sequence_length = len(coordinates)
    sequence_threshold = int(0.25 * sequence_length)  # Threshold for what constitutes "close" or "far" in 1D

    categories = {
        'close_in_1d': [],
        'far_in_1d': [],
        'close_in_3d': [],
        'far_in_3d': [],
        'close_in_1d_close_in_3d': [],
        'close_in_1d_far_in_3d': [],
        'far_in_1d_close_in_3d': [],
        'far_in_1d_far_in_3d': []
    }

    # Collect all potential pairs
    for i in range(sequence_length):
        for j in range(i + 1, sequence_length):
            euclidean_dist = np_matrix[i][j]
            sequence_dist = abs(i - j)
            pair_info = {
                'i': i + 1,
                'j': j + 1,
                'euclidean_dist': euclidean_dist,
                'sequence_dist': sequence_dist,
                'residue1': coordinates[i + 1][0],
                'coordinates1': coordinates[i + 1][1],
                'residue2': coordinates[j + 1][0],
                'coordinates2': coordinates[j + 1][1]
            }

            categories['close_in_1d'].append(pair_info)
            categories['far_in_1d'].append(pair_info)
            categories['close_in_3d'].append(pair_info)
            categories['far_in_3d'].append(pair_info)

            if sequence_dist <= sequence_threshold and euclidean_dist <= mean_distance:
                categories['close_in_1d_close_in_3d'].append(pair_info)
            if sequence_dist <= sequence_threshold and euclidean_dist > mean_distance:
                categories['close_in_1d_far_in_3d'].append(pair_info)
            if sequence_dist > sequence_threshold and euclidean_dist <= mean_distance:
                categories['far_in_1d_close_in_3d'].append(pair_info)
            if sequence_dist > sequence_threshold and euclidean_dist > mean_distance:
                categories['far_in_1d_far_in_3d'].append(pair_info)

    # Define sorting criteria for each category
    sort_criteria = {
        'close_in_1d': lambda x: x['sequence_dist'],
        'far_in_1d': lambda x: -x['sequence_dist'],
        'close_in_3d': lambda x: x['euclidean_dist'],
        'far_in_3d': lambda x: -x['euclidean_dist'],
        'close_in_1d_close_in_3d': lambda x: (x['sequence_dist'], x['euclidean_dist']),
        'close_in_1d_far_in_3d': lambda x: (x['sequence_dist'], -x['euclidean_dist']),
        'far_in_1d_close_in_3d': lambda x: (-x['sequence_dist'], x['euclidean_dist']),
        'far_in_1d_far_in_3d': lambda x: (-x['sequence_dist'], -x['euclidean_dist'])
    }

    # Sort and limit each category to max_pairs
    for key in categories:
        categories[key].sort(key=sort_criteria[key])
        categories[key] = categories[key][:max_pairs]

    return categories, mean_distance


def process_mmcif_files(directory, max_pairs):
    parser = MMCIFParser(QUIET=True)
    results = {}

    for filename in os.listdir(directory):
        if filename.endswith(".cif"):
            filepath = os.path.join(directory, filename)
            structure_id = filename[:-4]
            try:
                structure = parser.get_structure(structure_id, filepath)
            except (PDBExceptions.PDBConstructionException, IOError) as e:
                print(f"Error parsing {filename}: {e}")
                continue

            ca_coords = {}
            model = structure[0]
            chain = next(model.get_chains())

            for residue in chain:
                if 'CA' in residue:
                    ca = residue['CA']
                    res_id = residue.get_id()[1]
                    res_name = three_to_one(residue.get_resname())
                    # Storing coordinates as a list directly here
                    ca_coords[res_id] = [res_name, np.round(ca.get_coord(), 2).tolist()]

            residues = list(ca_coords.keys())
            n_residues = len(residues)
            distance_matrix = np.zeros((n_residues, n_residues))

            for i, res_i in enumerate(residues):
                for j, res_j in enumerate(residues):
                    if i > j:
                        dist = calculate_distance(ca_coords[res_i][1], ca_coords[res_j][1])
                        distance_matrix[i][j] = dist
                        distance_matrix[j][i] = dist

            categories, mean_distance = classify_pairs(ca_coords, distance_matrix, max_pairs)

            results[structure_id] = {
                'mean_distance': mean_distance,
                **categories,
                'coordinates': ca_coords,
                'distance_matrix': distance_matrix.tolist()  # Convert to list for JSON serialization
            }

    return results

def main():
    directory = 'proteins'
    max_pairs = 5  # This can be adjusted as needed
    results = process_mmcif_files(directory, max_pairs)
    output_file = 'protein_residue_distances.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results written to {output_file}")

if __name__ == "__main__":
    main()