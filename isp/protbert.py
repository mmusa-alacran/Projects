import torch
from transformers import BertModel, BertTokenizer
from Bio import SeqIO

device = torch.device('cuda')
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
model = BertModel.from_pretrained("Rostlab/prot_bert")

try:
   # Open the FASTA file
   with open("test.fa", "r") as fasta_file:
      # Iterate through each record in the FASTA file
      for record in SeqIO.parse(fasta_file, "fasta"):
            sequence = record.seq
            print(len(sequence))
            sequence = ' '.join(list(sequence)) # The tokenizer of prot_bert only accept white space splitted sequence
            encoded_input = tokenizer(sequence, return_tensors='pt').to(device)
            output = model(**encoded_input)
            print(f"{record.id}: {output.last_hidden_state.squeeze().detach().cpu().numpy()}")
            print(output['last_hidden_state'].shape)
except FileNotFoundError:
   print("The file was not found. Please check the file path.")
except Exception as e:
   print(f"An error occurred: {e}")


