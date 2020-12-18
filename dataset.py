import tqdm
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from collections import defaultdict
import os
import os.path
import os
import csv
from itertools import zip_longest


def txt_to_csv(txt_path, save_path):

    patient = []
    doctor = []

    print('reading the file....')

    with open(txt_path, 'r') as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            
            if line.startswith('Patient:'):

                if not lines[i+2].startswith('Doctor:'):
                    patient.append(' '.join(lines[i+1:i+2]))
                else:
                    patient.append(lines[i+1])

            elif line.startswith('Doctor:'):
                
                doctor.append(lines[i+1])


    print('length of patient query', len(patient))
    print('length of doctor response', len(doctor))
    
    if len(patient) == len(doctor):
        data = {'src': patient, 'trg': doctor}
    
        data = pd.DataFrame.from_dict(data)

        data.to_csv(save_path, index=False)
    
    else: 

        zipped = zip_longest(patient, doctor, fillvalue='Hello, how are you doing?')

        with open(save_path, mode='w', newline='') as csvfile:
            
            fieldnames = ['src', 'trg']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)


            for query, response in zipped:

                writer.writerow({'src': query, 'trg': response})


# Initial data cleaning
def clean_up_txt(txt_path, cleaned_path):

    line_count = 0

    with open(txt_path, 'r') as reader, open(cleaned_path, 'w') as writer:

        line = reader.readline()

        ignore_1 = 'Dr.'
        ignore_2 = 'Regards'
        ignore_3 = "Dr "
        ignore_4 = "Ask A Doctor"
        ignore_5 = 'Dialogue'
        ignore_6 = 'http'
        ignore_7 = 'id='
        ignore_8 = 'Hello and Welcome to'
        ignore_9 = 'Hello and welcome to ‘Ask A Doctor’ service. I have reviewed your query and here is my'
        ignore_10 = 'I have reviewed your query and here is my advice'

        # a sentence gets very useless when it starts with any of the following character
        ignore_list = [ignore_1, ignore_2, ignore_3, ignore_4, ignore_5, ignore_6, ignore_7, ignore_8, ignore_9, ignore_10]

        while line:

            line_count += 1
            line = line.strip()

            if any(map(line.startswith, ignore_list)):
                print('removing useless words')


            elif len(line) <= 6:
                print('removing shorter sentences')
            
            else:
                writer.writelines(line + '\n')

            if line_count % 100:
                print('processing {}th line'.format(line_count))

            line = reader.readline()

    print('process finished')


if __name__ == "__main__":


    covid_txt = 'dataset/covid.txt'
    txt_path = 'dataset/non_covid_2.txt'
    cleaned_path = 'dataset/non_covid_2_cleaned.txt'

    # clean_up_txt(txt_path, cleaned_path)

    non_covid_csv = 'dataset/non_covid_2.csv'
    txt_to_csv(cleaned_path, non_covid_csv)

    original_csv = 'dataset/covid.csv'
    txt_to_csv(covid_txt, original_csv)

    original = pd.read_csv(original_csv, names=['src', 'trg'])
    non_covid = pd.read_csv(non_covid_csv, names=['src', 'trg'])
    merge = original.append(non_covid)

    merge.to_csv('dataset/combined.csv', index=False)



