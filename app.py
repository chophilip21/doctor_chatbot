from flask import Flask, request, render_template, redirect, url_for
import requests
import json
import os
import emoji

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True


#The first index page
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
  
    #TODO: You will instantiate the model here
    # bot = Seq2SeqModel()
    # response = bot.predict(userText)
    # response = ' '.join(response) 
    
    greetings_1 = "hello"
    greetings_2 = "Hi there"
    greetings_3 = "Hi"
    greetings_4 = "Hello"
    greetings_5 = "hi"
    greetings = [greetings_1, greetings_2, greetings_3, greetings_4, greetings_5]

    farewell_1 = 'bye'
    farewell_2 = "okay bye"
    farewell_3 = 'see you'
    farewell_4 = 'thanks bye'
    farewell_5 = "Bye"
    farewell = [farewell_1, farewell_2, farewell_3, farewell_4, farewell_5]

    response = "Sorry Me completely untrained right now. Too dumb to answer your qestion"
    
    if len(userText) <10 and any(map(userText.startswith, greetings)):
        response = "Hi there, how can I help you today?"
      
    if len(userText) <10 and any(map(userText.startswith, farewell)):
        response = "Take care and bye!"
    
    response = emoji.emojize(':pill: {}'.format(response))

    return response


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)