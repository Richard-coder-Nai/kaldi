#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import pandas as pd
import numpy as np
import argparse
import pdb

def trans_overlap_txt(meta_csv, overlap_txt):
    '''convert vggface1id to voxceleb1id format'''
    assert os.path.exists(meta_csv)
    assert os.path.exists(overlap_txt)

    id2name_df = pd.read_csv(meta_csv, sep='\t', usecols=[0,1], index_col=[1]) 
    overlap_names = np.loadtxt(overlap_txt, dtype=str)
        
    with open(overlap_txt, 'w') as scp:
        for name in overlap_names:
            idex = id2name_df.loc[str(name),'VoxCeleb1 ID']
            scp.write(idex + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta-csv-file', help='csv file  containing infomation between vggface1id and voxceleb1id.')
    parser.add_argument('--overlap-txt-file', help='txt file  containing the list of speakers that also exist in SITW. ')
    args = parser.parse_args()
    trans_overlap_txt(args.meta_csv_file, args.overlap_txt_file)