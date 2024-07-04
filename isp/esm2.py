import torch
import numpy as np
import os
from transformers import EsmTokenizer, EsmModel
import gc
from Bio import SeqIO

THRESHOLD = 1022  # Max sequence length for the model

def main(file_path):
    # Load ESM-2 model
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
    output_dir = 'embeddings'
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        i = 0
        for record in SeqIO.parse(file_path, "fasta"):
            i+=1
            sequence = str(record.seq)
            vectors = []
            print(f"{i}: {record.id} ...")
            while sequence:
                subseq = sequence[:THRESHOLD]
                sequence = sequence[THRESHOLD:]
                inputs = tokenizer(subseq, return_tensors="pt", padding="max_length", truncation=True, max_length=THRESHOLD)
                inputs = inputs.to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    embeddings = outputs.last_hidden_state.squeeze().detach().cpu().numpy()
                    print(embeddings)
                    print(outputs['last_hidden_state'].shape)
    except Exception as e:
        print(f"Error processing file: {e}")

    gc.collect()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', metavar='path', required=True, help='the path to the fasta files')
    args = parser.parse_args()
    main(args.input)
