import warnings
import torch
from torch.nn.functional import cosine_similarity
from transformers import BertModel, BertTokenizer, EsmModel, EsmTokenizer
from Bio.PDB import MMCIFParser
from Bio.SeqUtils import seq1
import argparse

class SequenceEmbedding:
    def __init__(self, transformer, metric):
        # Initialize the device to GPU if available, otherwise use CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the appropriate model and tokenizer based on the input argument
        if transformer == 'bert':
            # ProtBERT is a BERT model trained on protein sequences
            self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
            self.model = BertModel.from_pretrained("Rostlab/prot_bert")
        elif transformer == 'esm2':
            # ESM-2 is a model designed specifically for protein sequences
            self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
            self.model = EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D")
        else:
            # Raise an error if an unsupported transformer model is specified
            raise ValueError("Unsupported transformer. Choose 'bert' or 'esm2'.")

        # Move the model to the device (GPU or CPU)
        self.model.to(self.device)
        self.metric = metric

    def load_sequence(self, file_path):
        # Suppresses warnings about discontinuous chains when parsing protein structures
        warnings.filterwarnings("ignore", category=UserWarning, message="Chain [A-Z] is discontinuous.*")
        parser = MMCIFParser(QUIET=True)
        # Load the protein structure from a CIF file
        structure = parser.get_structure('Hemoglobin', file_path)
        # Extract the first chain from the structure
        first_chain = next(structure.get_chains())
        # Convert the sequence of residues to a one-letter code, excluding unknown types ('X')
        sequence = "".join(residue.get_resname() for residue in first_chain.get_residues())
        return seq1(sequence).replace('X', '')
    
    def mutate_sequence(self, sequence, position, new_amino_acid):
        # Modify a specific position in the protein sequence with a new amino acid
        if position < 1 or position > len(sequence):
            raise ValueError("Position out of the sequence range.")
        return sequence[:position - 1] + new_amino_acid + sequence[position:]

    def get_embedding(self, sequence):
        # Tokenize the sequence and encode it into a tensor
        sequence = ' '.join(sequence) if self.tokenizer.__class__.__name__ == "BertTokenizer" else sequence
        encoded_input = self.tokenizer(sequence, return_tensors='pt', padding="max_length", truncation=True, max_length=1022)
        encoded_input = encoded_input.to(self.device)
        # Generate embeddings for the sequence without updating model weights
        with torch.no_grad():
            output = self.model(**encoded_input)
        return output  # Returns a tensor of embeddings

    def calculate_distance(self, embedding1, embedding2):
        # Compute the distance between two embeddings based on the specified metric
        if self.metric == 'euclidean':
            # Compute Euclidean distance
            return torch.sqrt(torch.sum((embedding1 - embedding2) ** 2)).item()
        elif self.metric == 'cosine':
            # Compute cosine similarity using two different dimensions and return their means
            cos_sim0 = torch.nn.CosineSimilarity(dim=0)
            cos_sim1 = torch.nn.CosineSimilarity(dim=1)
            similarity_scores0 = cos_sim0(embedding1, embedding2)
            similarity_scores1 = cos_sim1(embedding1, embedding2)
            return (similarity_scores0.mean().item(), similarity_scores1.mean().item())
        else:
            raise ValueError("Unsupported metric. Choose 'euclidean' or 'cosine'.")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process protein sequences.')
    parser.add_argument('--transformer', type=str, required=True, help='Transformer model to use (protbert or esm2)')
    parser.add_argument('--mutation', type=str, required=True, help='Amino acid to mutate to')
    parser.add_argument('--position', type=int, required=True, help='Position of the mutation')
    parser.add_argument('--metric', type=str, required=True, help='Distance metric to use (euclidean or cosine)')
    args = parser.parse_args()

    # Initialize the SequenceEmbedding object
    sequence_embedder = SequenceEmbedding(args.transformer, args.metric)
    # Load and display the original sequence
    sequence = sequence_embedder.load_sequence('1a3n.cif')
    print("Original Sequence:", sequence)
    
    # Mutate the sequence and display the mutated sequence
    mutated_sequence = sequence_embedder.mutate_sequence(sequence, args.position, args.mutation)
    print("Mutated Sequence:", mutated_sequence)

    # Get and print embeddings for the original and mutated sequences
    embedding_original = sequence_embedder.get_embedding(sequence)
    embedding_mutated = sequence_embedder.get_embedding(mutated_sequence)
    print(f"{'Original:'} {embedding_original.last_hidden_state.squeeze().detach()}")
    print(embedding_original['last_hidden_state'].shape)
    print(f"{'Mutated:'} {embedding_mutated.last_hidden_state.squeeze().detach()}")
    print(embedding_mutated['last_hidden_state'].shape)
    
    # Calculate and print the distance between the original and mutated sequence embeddings
    distance = sequence_embedder.calculate_distance(embedding_original.last_hidden_state.squeeze().detach(),
                                                    embedding_mutated.last_hidden_state.squeeze().detach())
    print(f"Distance between original and mutated sequence embeddings ({args.metric.capitalize()}):", distance)

if __name__ == "__main__":
    main()
