import os

import numpy as np
import yaml
import pandas as pd

IN_DIR = r'../data/tensile'
OUT_DIR = r'../data/02 prepared data'
INFO_PATH = r'../info/01 raw info.xlsx'


def extract_info(in_dir, info_path):
    info_df = pd.DataFrame(columns=['filename', 'test type', 'temperature', 'material'])
    for filename in os.listdir(in_dir):
        info_row = pd.Series()
        info_row['filename'] = filename
        name_list = filename.split('_')
        if name_list[0] == 'P':
            info_row['test type'] = 'PST'
        else:
            info_row['test type'] = 'UT'
        info_row['temperature'] = float(name_list[1])
        info_row['material'] = 'AA6061-T651_' + name_list[2]
        info_df = info_df.append(info_row, ignore_index=True)
    info_df.to_excel(info_path, index=False)


def prepare_data(in_dir, out_dir, info_path):
    info_df = pd.read_excel(info_path)
    for filename in os.listdir(in_dir):
        out_data_df = pd.DataFrame()
        in_data_df = pd.read_csv(f'{in_dir}/{filename}')
        eng_strain = in_data_df['Strain']
        eng_stress = in_data_df['Stress_MPa']
        true_strain = np.log(1 + eng_strain)
        true_stress = eng_stress * (1 + eng_strain)
        out_data_df['eng strain'] = eng_strain
        out_data_df['eng stress'] = eng_stress
        out_data_df['Strain'] = true_strain
        out_data_df['Stress(MPa)'] = true_stress
        test_id = info_df.loc[info_df['filename']==filename]['test id'].values[0]
        out_data_df.to_csv(f'{out_dir}/{test_id}.csv', index=False)

if __name__ == '__main__':
    # extract_info(IN_DIR, INFO_PATH)
    prepare_data(IN_DIR, OUT_DIR, INFO_PATH)
