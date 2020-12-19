from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.nist_score import sentence_nist
from nltk.translate.meteor_score import meteor_score
from simpletransformers.seq2seq import (
    Seq2SeqModel,
    Seq2SeqArgs,
)
import torch
from sklearn.model_selection import train_test_split
import pandas as pd



def print_prediction(labels, preds):
  
    for i, (p, g, q) in enumerate(zip(preds, labels, test_df.input_text)):
      print('------------example {}----------'.format(i))
      print('patient query:', q.strip())
      print('ground truth:', g.strip())
      print('prediction: ', p.strip())



def nist_2(labels, preds):

    label = ' '.join([str(elem) for elem in labels])
    prediction = ' '.join([str(elem) for elem in preds])
  
    if len(prediction) < 2 or len(label) < 2:
        return 0
    return sentence_nist([label], prediction, 2)

def nist_4(labels, preds):

    label = ' '.join([str(elem) for elem in labels])
    prediction = ' '.join([str(elem) for elem in preds])

    if len(prediction) < 4 or len(label) < 4:
        return 0

    return sentence_nist([label], prediction, 4)

def calculate_m_score(target, predictions, length):

  score = 0

  for t, p in zip(target, predictions):
    score += meteor_score(t, p)

  return score / length



if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('good to go')
    print(f'Device: {DEVICE}')


    # model = Seq2SeqModel("bart", "facebook/bart-base", "bart", config="outputs/best_model/config.json")   
    model = Seq2SeqModel(encoder_decoder_type='bart', encoder_decoder_name="facebook/bart-base", config="outputs/best_model/config.json")   


    test = "I have been having fever last few days. Any thoughts on this?"
    inference = model.predict([test])
    print(inference)