import os
import pandas as pd
from abnumber import Chain
from joblib import Parallel, delayed
from tqdm import tqdm

# Define Project Root
PROJECT_ROOT = os.path.dirname(__file__) 

def process_sequence(seq_index, sequence, scheme):
    output = {'Seq_Index': seq_index}
    try:
        ab_chain = Chain(sequence, scheme=scheme)
        if ab_chain.chain_type == 'H':
            output.update({
                'Seq_Index': seq_index,
                'Framework_1': ab_chain.fr1_seq or 'None',
                'Complementarity_1': ab_chain.cdr1_seq or 'None',
                'Framework_2': ab_chain.fr2_seq or 'None',
                'Complementarity_2': ab_chain.cdr2_seq or 'None',
                'Framework_3': ab_chain.fr3_seq or 'None',
                'Complementarity_3': ab_chain.cdr3_seq or 'None',
                'Framework_4': ab_chain.fr4_seq or 'None'
            })
            return output
    except Exception as error:
        output.update({
            'Framework_1': '', 'Complementarity_1': '',
            'Framework_2': '', 'Complementarity_2': '',
            'Framework_3': '', 'Complementarity_3': '',
            'Framework_4': ''
        })
    
    return output

def process_file(input_filepath, output_filepath, scheme_type):
    dataframe = pd.read_csv(input_filepath)
    parallel_processor = Parallel(n_jobs=-1, backend="loky")
    processed_list = parallel_processor(
        delayed(process_sequence)(idx, sequence, scheme_type) 
        for idx, sequence in tqdm(dataframe['vh'].dropna().iteritems(), desc="Processing Antibody Sequences")
    )
    processed_dataframe = pd.DataFrame(processed_list).set_index('Seq_Index')

    for column in processed_dataframe.columns:
        dataframe[column] = processed_dataframe[column]

    dataframe.to_csv(output_filepath, index=False)

def run_processing(input_path, output_path, scheme):
    process_file(input_path, output_path, scheme)

if __name__ == "__main__":
    # Construct relative input path
    input_file = os.path.join(PROJECT_ROOT, 'datasets', 'combined_training_data.csv') 
    
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Apply the function to the 'vh' column and create new columns
    df[['H-FR1', 'H-CDR1', 'H-FR2', 'H-CDR2', 'H-FR3', 'H-CDR3', 'H-FR4']] = df['vh'].apply(lambda x: pd.Series(split_heavy_chain(x)))

    # Construct relative output path
    output_file = os.path.join(PROJECT_ROOT, 'datasets', 'combined_training_data_split.csv') 
    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

    print(f"Processed data saved to {output_file}")


