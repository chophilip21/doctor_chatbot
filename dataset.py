 import tqdm
import spacy
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from simpletransformers.seq2seq import Seq2SeqModel
from sklearn.model_selection import train_test_split
import os
import os.path
import tensorflow as tf
import os
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.nist_score import sentence_nist


 sentence_nist

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('good to go')
print(f'Device: {DEVICE}')

csv_link = 'https://raw.githubusercontent.com/chophilip21/covid_dialogue/main/dialogue.csv' #original
augmented_link = 'https://raw.githubusercontent.com/chophilip21/covid_dialogue/main/augmented.csv' #augmented version
txt_link = 'https://raw.githubusercontent.com/chophilip21/covid_dialogue/main/covid_additional.txt' # raw text version

dataset = pd.read_csv(augmented_link, names = ['input_text', 'target_text'], header=0)
train_df, test_df = train_test_split(dataset, test_size=0.2)
valid_df, test_df = train_test_split(test_df, test_size=0.5)

print('The length of train_df is: ', len(train_df))
print('The length of valid is: ', len(valid_df))
print('The length of test_df is: ', len(test_df))


def txt_to_dict(txt_path, save_path):

    patient = []
    doctor = []

    with open(txt_path, 'r') as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            if line.startswith('Patient:'): 
                patient.append(' '.join(lines[i+1:i+2]))
            
            elif line.startswith('Doctor:'):
                doctor.append(' '.join(lines[i+1: i+2]))

    data = {'src': patient, 'trg': doctor}

    return data