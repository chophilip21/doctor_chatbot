from flask import Flask, request, render_template
import requests
import json
from login_fields import *


app = Flask(__name__)
app.secret_key = 'replace later'

@app.route("/", methods=['GET', 'POST'])
def index():

    #instatiate form
    reg_form = RegistrationForm()
    if reg_form.validate_on_submit():
        return "Validation is working well"


    return render_template('index.html', form=reg_form)


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)