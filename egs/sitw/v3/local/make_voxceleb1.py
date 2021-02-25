#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
import argparse
import requests
import pdb


def trans_overlap_txt(args):
    '''convert vggface1id to voxceleb1id format'''
    meta_csv = os.join(args.voxceleb1_dir, 'vox1_meta.csv') 
    assert os.path.exists(meta_csv)

    overlap_txt_address = 'http://www.openslr.org/resources/49/voxceleb1_sitw_overlap.txt'
    overlap_txt = requests.get(overlap_txt_address)
    id2name_df = pd.read_csv(meta_csv, sep='\t', usecols=[0, 1], index_col=[1])
    overlap_names = str(overlap_txt.content,encoding='utf8').split('\n')[:-1]
    overlap_idx = [id2name_df.loc[str(name),'VoxCeleb1 ID'] for name in overlap_names]
    
    pdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxceleb1-dir', help='voxceleb1 database dir, e.g. /export/voxceleb1',
                        default='/work102/lilt/database/VoxCeleb/voxceleb1')
    # parser.add_argument('--meta-csv-file', help='csv file  containing infomation between vggface1id and voxceleb1id.')
    parser.add_argument('--data-dir', help='e.g. data/', default='data/train')
    parser.add_argument('--overlap-txt-file', help='voxceleb1_sitw_overlap.txt', default=None)
    args = parser.parse_args()
    trans_overlap_txt(os.path.join(args.voxceleb1_dir, 'vox1_meta.csv'))
