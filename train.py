import tqdm
import warnings
import numpy as np
import pandas as pd
from simpletransformers.seq2seq import (
    Seq2SeqModel,
    Seq2SeqArgs,
)
from sklearn.model_selection import train_test_split
import os.path
# import torch
from static.macro import MODEL_ARGS 


if __name__ == "__main__":
    # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('good to go')
    # print(f'Device: {DEVICE}')

    large_destination = 'dataset/aug_reduced.csv'

    # dataset = pd.read_csv(large_destination, names = ['input_text', 'target_text'], header=0)[0:50000]
    dataset = pd.read_csv(large_destination, names = ['input_text', 'target_text'], header=0)[1:]

    train_df, valid_df = train_test_split(dataset, test_size=0.05)

    print('The length of train_df is: ', len(train_df))
    print('The length of valid is: ', len(valid_df))


   
    model = Seq2SeqModel(
        "distilbert",
        "distilbert-base-cased-distilled-squad",
        "bert-base-uncased",
    )

  
    print('start training....')
    model.train_model(train_df, eval_data=valid_df, args=MODEL_ARGS)

    print('Done training. Testing a sample...')
    test = "an coronavirus symptoms be mild for some people versus severe ?"
    inference = model.predict([test])
    print(inference)
