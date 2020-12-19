from flask import Flask, request, render_template, redirect, url_for
import requests
import json
from login_fields import *
from database import *
from passlib.hash import pbkdf2_sha256
from flask_login import LoginManager, login_user, current_user, login_required, logout_user

app = Flask(__name__)
app.config['SECRET_KEY'] = 'replace_later'

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

# @login_required #you should not see this without logging in
@app.route("/chat", methods=['GET', 'POST'])
def chat():

    if not current_user.is_authenticated:
        return "Please login first to access chat"

    return "Chating room will be here"

@app.route("/logout", methods=['GET'])
def logout():

    logout_user()

    return "Logged out. Thanks for visiting"



if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)