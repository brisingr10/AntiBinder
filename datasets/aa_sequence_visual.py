"""
This script is designed to analyze and visualize amino acid sequence data.
"""
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import logomaker
import os

def read_sequences_from_csv(df):
    sequences = []
    for index, row in df.iterrows():
        sequence = SeqRecord(Seq(row['vh']), id=str(index))
        sequences.append(sequence)
    return sequences

def perform_alignment(sequences):
    with open("temp.fasta", "w") as temp_file:
        for sequence in sequences:
            temp_file.write(f">{sequence.id}\n")
            temp_file.write(str(sequence.seq) + "\n")
    
    align_file_cmd = "muscle5.1.linux_intel64 -align temp.fasta -output aln.fasta"
    os.system(align_file_cmd)
    alignment = AlignIO.read("aln.fasta", "fasta")
    return alignment

def create_df(alignment):
    sequences = []
    for record in alignment:
        sequences.append(str(record.seq))
    return sequences

def create_logo(sequence_list, output_filename):
    counts_df = logomaker.alignment_to_matrix(sequence_list, to_type='counts')
    
    # It's up to you to decide.
    colors = {
        'A': 'green',
        'C': 'blue',
        'D': 'red',
        'E': 'red',
        'F': 'cyan',
        'G': 'orange',
        'H': 'yellow',
        'I': 'cyan',
        'K': 'magenta',
        'L': 'cyan',
        'M': 'cyan',
        'N': 'purple',
        'P': 'purple',
        'Q': 'purple',
        'R': 'magenta',
        'S': 'green',
        'T': 'green',
        'V': 'cyan',
        'W': 'cyan',
        'Y': 'cyan',
    }
    
    plt.figure(figsize=(90, 30))
    logo_fig = logomaker.Logo(counts_df, color_scheme=colors)
    plt.savefig(output_filename)
    plt.show()

if __name__ == "__main__":
    # demo
    random.seed(42)
    
    # Load data
    df = pd.read_csv("data.csv")
    
    # Randomly sample 250 sequences
    data1 = df.sample(n=250)
    data1 = read_sequences_from_csv(data1)
    aligned_data = perform_alignment(data1)
    aligned_df = create_df(aligned_data)
    create_logo(aligned_df, "output1.png")
    print("1 Finished!")
    

