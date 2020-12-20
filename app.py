from flask import Flask, request, render_template, redirect, url_for
import requests
import json
from login_fields import *
from database import *
from passlib.hash import pbkdf2_sha256
from flask_login import LoginManager, login_user, current_user, login_required, logout_user
import os
import emoji


app = Flask(__name__)
app.config['SECRET_KEY'] = 'replace_later'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Configure Recaptcha
app.config['RECAPTCHA_USE_SSL']= False
app.config['RECAPTCHA_PUBLIC_KEY']='6LcTAAwaAAAAAA1Kfmk0gyCMWzvNMCJb-VRtnCgD'
app.config['RECAPTCHA_PRIVATE_KEY']='6LcTAAwaAAAAAD-kInsgXf9C0147lTD4wQZj3hl7' #! When debugging, Recaptcha will not work.
app.config['RECAPTCHA_OPTIONS']= {'theme':'white'}

#configure database
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgres://smnmkjoqawrehl:ca81f4997efc52260d52ffa90924a5d380e5dbb93e1d557b4a432676a53251d3@ec2-52-5-176-53.compute-1.amazonaws.com:5432/deand921k6jgvb'
db = SQLAlchemy(app)

#Configure flask login
login = LoginManager(app)
login.init_app(app)


@login.user_loader
def load_user(id):
    return User.query.get(int(id))



#The first index page
@app.route("/", methods=['GET', 'POST'])
def index():

    #instatiate form
    reg_form = RegistrationForm()


    if reg_form.validate_on_submit():
        username = reg_form.username.data
        password = reg_form.password.data

        #random 16 bit salt and iteration for password security
        hashed_pswd = pbkdf2_sha256.hash(password)

        # add to db
        user = User(username=username, password=hashed_pswd)
        db.session.add(user)
        db.session.commit()

        return redirect(url_for('login'))


    return render_template('index.html', form=reg_form)


@app.route("/login", methods=['GET', 'POST'])
def login():

    login_form = LoginForm()

    if login_form.validate_on_submit():

        user_object= User.query.filter_by(username=login_form.username.data).first()
        login_user(user_object)

        if current_user.is_authenticated:
            #double check authentication just in case
            return redirect(url_for('chat'))
        else:
            return "login authentication failed some reason. Please open a thread or email chophilip21@gmail.com about this"

    return render_template('login.html', form=login_form)

def dir_last_updated(folder):
    return str(max(os.path.getmtime(os.path.join(root_path, f))
                   for root_path, dirs, files in os.walk(folder)
                   for f in files))


# @login_required #you should not see this without logging in
@app.route("/chat", methods=['GET', 'POST'])
def chat():

    # if not current_user.is_authenticated:
    #     return "Please login first to access chat"

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
    login_form = LoginForm()
    username = login_form.username.data

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

        if username is None: 
            response = "Hi there, how can I help you today?"
        else: 
            response = "Hi {}, how can I help you today?".format(username)

    if len(userText) <10 and any(map(userText.startswith, farewell)):
        response = "Take care and bye!"
    
    response = emoji.emojize(':pill: {}'.format(response))

    return response



@app.route("/logout", methods=['GET'])
def logout():

    logout_user()

    return "Logged out. Thanks for visiting"



if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)