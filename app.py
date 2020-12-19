from flask import Flask, request, render_template
import requests
import json
from login_fields import *
from database import *


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


@app.route("/", methods=['GET', 'POST'])
def index():

    #instatiate form
    reg_form = RegistrationForm()

    if reg_form.validate_on_submit():
        username = reg_form.username.data
        password = reg_form.password.data

        # add to db
        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()

        return "finished signing up"


    return render_template('index.html', form=reg_form)


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)