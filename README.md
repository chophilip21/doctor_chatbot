# covid_dialgue

- This repository is created for SFU CMPT 825 Final Project, and the following is our report: https://drive.google.com/file/d/16trZ2V6sWcwlxhVWZyeJ1Y-YaIVAs2ds/view?usp=sharing  

## DialoGPT
- This repository uses DialoGPT as the backbone: https://arxiv.org/abs/1911.00536. 
- DialoGPT's architecture is based on GPT-2, an `autoregressive` language model that uses output of the previous timestep as an input to predict the next token. This is different from other Transformer born language models like BERT. 

![gpt-2](https://github.com/ncoop57/i-am-a-nerd/blob/master/images/autoregressive.gif?raw=1)

- The idea here is that we use DialoGPT pretrained on huge conversational corpus as the base, and fine-tune the model using the COVID-19 dataset. 