from flask import Flask, request, render_template, redirect, url_for
import requests
import json
import os
import emoji
import re
from static.macro import PROFANITY, GREETINGS, FAREWELL
import COVID19Py
import random
from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True


# The first index page
@app.route("/", methods=['GET', 'POST'])
def chat():
    return render_template('chatroom.html')


"""
TODO:
- When the bot responds Take care and bye! The session is done. You should no longer be able to type
- Replace the type message to start another session, or logout(refresh..? clear output..?), whatever.
- Add grammar checker. There will be a lot of grammar problems for sure.
"""


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')

    response = "I am unable to answer your question at the moment. Please contact emergency if this is an urgent matter."

    if len(userText) <= 1:
        response = 'Hmm your question is too short. Maybe you mistyped your question?'

    else:

        # default Macro
        if userText.startswith('stats:covid'):
            covid19 = COVID19Py.COVID19(data_source="csbs")

            stats = covid19.getLatest()

            confirmed = stats['confirmed']
            deaths = stats['deaths']

            response = f'Here are the quick stats. \n  Total confirmed: {confirmed}, \n Total deaths: {deaths}'

        elif re.search("("+")|(".join(PROFANITY)+")", userText):
            response = 'I am sorry but I really do not want to hear any hate speach from you. Please ask a valid question, or leave'

        elif len(userText) < 10 and any(map(userText.startswith, GREETINGS)):
            response = ["Hi there, how can I help you today?",
                        'Hey, how can I help?', 'How may I help you today?', "Hello there!", "Hi, feel free to ask any questions", "How can I help you today?"]
            response = random.choice(response)

        elif len(userText) < 10 and any(map(userText.startswith, FAREWELL)):
            response = ["Take care and bye!", "Okay bye!",
                        "Thanks for visiting", "Okay stay safe!", "Goodbye!", "Farewell!"]
            response = random.choice(response)

        else:

            tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
            model = AutoModelWithLMHead.from_pretrained('output-small')

            new_user_input_ids = tokenizer.encode(userText + tokenizer.eos_token, return_tensors='pt')

            chat_history_ids = model.generate(
            new_user_input_ids, max_length=200,
            pad_token_id=tokenizer.eos_token_id,  
            no_repeat_ngram_size=3,       
            do_sample=True, 
            top_k=100, 
            top_p=0.7,
            temperature = 0.8
            )

            response = tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)


    response = emoji.emojize(':pill: {}'.format(response))

    return response


if __name__ == "__main__":
    #* For simple testing, use below:
    app.run(host="localhost", port=5000, debug=True) #only for testing
    
    #TODO: For real server deployment, use below
    # app.run(threaded=True, port=5000)
