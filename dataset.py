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
import json
import csv


def make_dataset(path_to_train_data, path_to_validation_data):
  f_train = open(path_to_train_data)
  train_data = json.load(f_train)
  f_train.close()
  # print(len(train_data))
  
  f_validate = open(path_to_validation_data)
  validate_data = json.load(f_validate)
  f_validate.close()
  # print(len(validate_data))

  train_contexted = []
  train_data = train_data

  for i in range(len(train_data)):
    row = []
    row.append(train_data[i][1])
    row.append(train_data[i][0])

    train_contexted.append(row)  

  validate_contexted = []

  for i in range(len(validate_data)):
    row = []
    row.append(validate_data[i][1])
    row.append(validate_data[i][0])
    validate_contexted.append(row)

  columns = ['response', 'context'] 

  trn_df = pd.DataFrame.from_records(train_contexted, columns=columns)
  
  trn_df = augment_dataset(trn_df)
  
  val_df = pd.DataFrame.from_records(validate_contexted, columns=columns)

  return trn_df,val_df


def augment_dataset(input_df):

    """
    Augmenting the dataset based on NLP aug library. If the dataset is small, this is a great way to boost things up.
    But you do not want to apply augmentation on the Doctor's response. These should not have any spelling mistakes. 
    - The augmentation that will be done here is character level augmentations and word level augmentations:
    - OCR error augmentation (character level)
    - Keyboard augmentation (character level)
    - Synonym augmenter (word level)
    """

    print('Augmenting the dataset based on Synonyms...')

    ocr = nac.OcrAug()
    response_OCR = []
    context_OCR = []
    
    keyboard = nac.KeyboardAug()
    response_keyboard = []
    context_keyboard = []

    synonym = naw.SynonymAug(aug_src='wordnet')
    response_synonym = []
    context_synonym = []


    for i in input_df.index:

        if i % 10 == 0:
            print('processing {}th line'.format(i))

        response = input_df['response'][i]
        context = input_df['context'][i]
        
        #augmentation
        ocr_augmented_line = ocr.augment(context, n=3)
        response_OCR.append(response)
        context_OCR.append(ocr_augmented_line)

        #keyboard augmentation
        keyboard_augmented_line = keyboard.augment(context)
        response_keyboard.append(response)
        context_keyboard.append(keyboard_augmented_line)

        #synonym augmentation
        synonym_augmented_line = synonym.augment(context)
        response_synonym.append(response)
        context_synonym.append(synonym_augmented_line)

        
    ocr_augmented_data = {'response': response_OCR, 'context': context_OCR}
    ocr_df = pd.DataFrame.from_dict(ocr_augmented_data)

    keyboard_augmented_data = {'response': response_keyboard, 'context': context_keyboard}
    keyboard_df = pd.DataFrame.from_dict(keyboard_augmented_data)

    synonym_augmented_data = {'response': response_synonym, 'context': context_synonym}
    synonym_df = pd.DataFrame.from_dict(synonym_augmented_data)

    augmented_1 = input_df.append(ocr_df, ignore_index=True)
    augmented_2 = augmented_1.append(keyboard_df, ignore_index=True)
    augmented_3 = augmented_2.append(synonym_df, ignore_index=True)

    print('original dataset length: {}'.format(len(input_df)))
    print('Augmented dataset length: {}'.format(len(augmented_2)))

    return augmented_3


def reformat_df(input_df, save_dir, n=6, train=True):

    print('reformatting the df to make space for context....')

    if train:
      save_path = os.path.join(save_dir, 'train.csv')
    else:
      save_path = os.path.join(save_dir, 'valid.csv')

    with open(save_path, mode='w+', newline='') as csvfile:
      fieldnames = ['Character', 'line']
      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

      #THE FORMAT NEEDS TO BE DIFFERENT. 
      for i in input_df.index:
        response = input_df['response'][i]
        query = input_df['context'][i]
        writer.writerow({'Character': 'Patient', 'line': query})
        writer.writerow({'Character': 'Doctor', 'line': response})

    csvfile.close()

    reformatted_df = pd.read_csv(save_path, names=fieldnames)
    contexted = []

    print('Inserting context... ')

    for i in range(n, len(reformatted_df['line'])):
      row = []
      prev = i - 1 - n # we additionally subtract 1, so row will contain current response and 7 previous responses  
      for j in range(i, prev, -1):
        row.append(reformatted_df['line'][j])
      contexted.append(row)

    columns = ['response', 'context'] 
    columns = columns + ['context/'+str(i) for i in range(n-1)]

    df = pd.DataFrame.from_records(contexted, columns=columns)

    if train:
      final_path = os.path.join(save_dir, 'contexted_train.csv')
    else:
      final_path = os.path.join(save_dir, 'contexted_valid.csv')

    df.to_csv(final_path, index=False)

    # return df


if __name__ == "__main__":

    train_json = 'dataset/train_duplicates.json'
    validate_json = "dataset/validate_data.json"
    
    train, valid = make_dataset(train_json, validate_json)

    reformat_df(train, 'dataset', train=True)
    reformat_df(valid, 'dataset', train=False)



    # train.to_csv('dataset/train.csv', index=False)
    # valid.to_csv('dataset/valid.csv', index=False)








