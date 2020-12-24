# covid_dialgue

- This repository is created based on my SFU CMPT 825 Final Project, and the following is our report: https://drive.google.com/file/d/16trZ2V6sWcwlxhVWZyeJ1Y-YaIVAs2ds/view?usp=sharing  
- The implementation here uses GPT-2 structure pretrained with DialoGPT weights provided from Microsoft.
- If you would like access to the custom dataset I have used for training, please email pycho@sfu.ca.
- This dataset is custom made and it's very small. GPT-2 model needs a lot of data to function properly and NLP-aug library was used to increase the size of the dataset. Refer to dataset.py for more information. 


## Instrunctions for running the app
- Install Virtual environemnt, and simply pip install requirements.txt
- If you just want to test the app, just run python app.py
- If you want to train from scratch, run train.py.

## Demo
Below is the live demo of the app.py. The app uses Javascript for the front and simple Python Flask as backend. It was originally launched on Heroku sever, but later pulled down as the files are too large to meet Heroku limit. I plan to explore other options like AWS EC2 in the future.  
![demo](demo.gif)

