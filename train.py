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
import torch




if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('good to go')
    print(f'Device: {DEVICE}')

    large_destination = 'dataset/combined.csv'

    # dataset = pd.read_csv(large_destination, names = ['input_text', 'target_text'], header=0)[0:50000]
    dataset = pd.read_csv(large_destination, names = ['input_text', 'target_text'], header=0)[1:]

    train_df, test_df = train_test_split(dataset, test_size=0.1)
    valid_df, test_df = train_test_split(test_df, test_size=0.5)


    print('The length of train_df is: ', len(train_df))
    print('The length of valid is: ', len(valid_df))
    print('The length of test_df is: ', len(test_df))


    # model_args = Seq2SeqArgs()
    # model_args.num_train_epochs = 10
    # model_args.evaluate_generated_text = True
    # model_args.evaluate_during_training = True
    # model_args.evaluate_during_training_verbose = True
    # model_args.max_length = 30
    # model_args.overwrite_output_dir = True


    model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 50,
    "train_batch_size": 2,
    "num_train_epochs": 5,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    "evaluate_generated_text": True,
    "evaluate_during_training": True,
    "evaluate_during_training_verbose": True,
    "save_best_model": True,
    "max_length": 50,
    'eval_batch_size': 2,
    "save_steps": 8000,
    "evaluate_during_training_steps": 8000,
    'use_multiprocessing': False,
    'fp16': True,
}


    model = Seq2SeqModel(
        "distilbert",
        "distilbert-base-cased-distilled-squad",
        "bert-base-cased",
    )

      # model = Seq2SeqModel(
    #     "roberta",
    #     "roberta-base",
    #     "bert-base-cased",
    # )

    # model = Seq2SeqModel(
    #     encoder_decoder_type="bart",
    #     encoder_decoder_name="facebook/bart-base",
    #     args=model_args,
    # )

    print('start training....')
    model.train_model(train_df, eval_data=valid_df, args=model_args)

    print('Done training. Testing a sample...')
    test = "an coronavirus symptoms be mild for some people versus severe ? for example, could it just involve being very fatigued, low grade fever for a few days and not the extreme symptoms? or is it always a full blown cold and struggle to breathe?"
    inference = model.predict([test])
    print(inference)