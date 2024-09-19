import os
import pandas as pd
from abnumber import Chain
from joblib import Parallel, delayed 
from tqdm import tqdm 


def prepare_seq(seq_idx, seq, main_sp) :
    result = {'Index': seq_idx}
    try:
        chain = Chain(seq, scheme=main_sp)
        if chain.chain_type == 'H':
            result = {
                'Index':seq_idx,
                'FR1':chain.fr1_seq or 'None',
                'CDR1':chain.cdr1_seq or 'None',
                'FR2':chain.fr2_seq or 'None',
                'CDR2':chain.cdr2_seq or 'None',
                'FR3':chain.fr3_seq or 'None',
                'CDR3':chain.cdr3_seq or 'None',
                'FR4':chain.fr4_seq or 'None'
            }
            return result
        
    except Exception as e:
        result.update({'FR1': '', 'CDR1': '', 'FR2' : '', 'CDR2': '', 'FR3': '', 'CDR3': '','FR4': ''})
    
    return result


def prepare_file(file_path, output_path, main_sp):
    # multi-process
    df = pd.read_csv(file_path)
    n_jobs = -1
    parallel_pre = Parallel(n_jobs=n_jobs, backend="loky")
    processed_data = parallel_pre(delayed(prepare_seq)(idx, seq, main_sp) for idx, seq in tqdm(df['vh'].dropna().iteritems(), desc="Processing Sequences"))
    processed_df = pd.DataFrame(processed_data).set_index('Index')
    for col in processed_df.columns:
        df[col] = processed_df[col]
    df.to_csv(output_path, index=False)


def final_pdata(input_file, output_file, main_sp):
    prepare_file(input_file,
                 output_file,
                 main_sp
                 )
    
if __name__ == "__main__":
    work_dir = "/AntiBinder/"
    input_file = ""
    output_file_prefix = input_file.split('.')[2]
    output_file = ""

    os.chdir(work_dir)

    main_sp = 'chothia'
    second_sp = 'Heavy_fv_oas_train_filtered'

    final_pdata(input_file, output_file, main_sp)
    print("Finish!")