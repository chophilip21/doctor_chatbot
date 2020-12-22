import tqdm
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
import os
import csv
from itertools import zip_longest
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
from nlpaug.util.file.download import DownloadUtil


def txt_to_csv(txt_path, save_path, style='GPT2'):

    """
    Take raw txt file and organize it into basic csv format
    """

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
    
    if not style == 'GPT2' :
        data = {'src': patient, 'trg': doctor}
    
        data = pd.DataFrame.from_dict(data)

        data.to_csv(save_path, index=False)
    
    else: 

        zipped = zip_longest(patient, doctor, fillvalue='Hello, how are you doing?')

        with open(save_path, mode='w', newline='') as csvfile:
            
            fieldnames = ['Character', 'Line']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            for query, response in zipped:
                writer.writerow({'Character': 'Patient', 'Line': query})
                writer.writerow({'Character': 'Doctor', 'Line': response})


def augment_dataset(input_csv, output_csv):

    """
    Augmenting the dataset based on NLP aug library. If the dataset is small, this is a great way to boost things up.
    But you do not want to apply augmentation on the Doctor's response. These should not have any spelling mistakes. 
    - The augmentation that will be done here is character level augmentations and word level augmentations:
    - OCR error augmentation (character level)
    - Keyboard augmentation (character level)
    - Synonym augmenter (word level)
    """

    print('Augmenting the dataset based on Synonyms...')
    original = pd.read_csv(input_csv, names=['character', 'line'])

    ocr = nac.OcrAug()
    c_OCR = []
    l_OCR = []
    
    keyboard = nac.KeyboardAug()
    c_keyboard = []
    l_keyboard = []

    synonym = naw.SynonymAug(aug_src='wordnet')
    c_synonym = []
    l_synonym = []

    for i in original.index:

        if i % 10 == 0:
            print('processing {}th line'.format(i))

        character = original['character'][i]
        line = original['line'][i]


        if character == 'Patient':
            #OCR augmentation
            ocr_augmented_line = ocr.augment(line, n=3)
            c_OCR.append(character)
            l_OCR.append(ocr_augmented_line)

            #keyboard augmentation
            keyboard_augmented_line = keyboard.augment(line)
            c_keyboard.append(character)
            l_keyboard.append(keyboard_augmented_line)

            #synonym augmentation
            synonym_augmented_line = synonym.augment(line)
            c_synonym.append(character)
            l_synonym.append(synonym_augmented_line)

        else:
            c_OCR.append(character)
            l_OCR.append(line)
            c_keyboard.append(character)
            l_keyboard.append(line)
            c_synonym.append(character)
            l_synonym.append(line)
        
    ocr_augmented_data = {'character': c_OCR, 'line': l_OCR}
    ocr_df = pd.DataFrame.from_dict(ocr_augmented_data)

    keyboard_augmented_data = {'character': c_keyboard, 'line': l_keyboard}
    keyboard_df = pd.DataFrame.from_dict(keyboard_augmented_data)

    synonym_augmented_data = {'character': c_synonym, 'line': l_synonym}
    synonym_df = pd.DataFrame.from_dict(synonym_augmented_data)

    augmented_1 = original.append(ocr_df, ignore_index=True)
    augmented_2 = augmented_1.append(keyboard_df, ignore_index=True)
    augmented_3 = augmented_2.append(synonym_df, ignore_index=True)
    augmented_3.to_csv(output_csv, index=False)

    print('original dataset length: {}'.format(len(original)))
    print('Augmented dataset length: {}'.format(len(augmented_2)))


def add_context_to_csv(input_csv, output_csv, n=6):

    contexted = []

    df = pd.read_csv(input_csv, names=['Character',"line"])

    for i in range(n, len(df['line'])):

        if df['Character'][i] == "Doctor":
            row = []
            prev = i - 1 - n # we additionally subtract 1, so row will contain current response and 7 previous responses  
            for j in range(i, prev, -1):
                row.append(df['line'][j])
            contexted.append(row)
        else:
            continue 

    columns = ['response', 'context'] 
    columns = columns + ['context/'+str(i) for i in range(n-1)]
    df1 = pd.DataFrame.from_records(contexted, columns=columns)
    df1.to_csv(output_csv, index=False)

    print('Modified csv to context based format')



if __name__ == "__main__":

    raw_txt = 'dataset/raw.txt'
    raw_csv = "dataset/1.raw.csv"

    #first convert txt to csv file
    txt_to_csv(raw_txt, raw_csv)

    #augment the dataset
    aug_csv = 'dataset/2.aug.csv'
    augment_dataset(raw_csv, aug_csv)

    context_csv = 'dataset/3.data.csv'

    add_context_to_csv(aug_csv, context_csv)








