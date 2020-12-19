from flask_wtf import FlaskForm, RecaptchaField
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, EqualTo, Regexp, ValidationError
from database import *
from passlib.hash import pbkdf2_sha256


def invalid_credentials(form, field):
    """
    Flask will automatically detect the paramters depending on where we are calling it
    username and password checker. It should not specifically tell what's wrong for security reason.
    """

    username_entered = form.username.data
    password_entered = field.data

    user_object = User.query.filter_by(username=username_entered).first()
    
    if user_object is None:
        raise ValidationError('Username or password is incorrect')

    #unhash to plain text version and compare
    elif not pbkdf2_sha256.verify(password_entered, user_object.password):
        raise ValidationError('Username or password is incorrect')



# if you have not registered yet, register
class RegistrationForm(FlaskForm):
    """
    Registration form 
    """

    #regex for checking password. 
    regex = "^(?=.*[0-9])(?=.*[a-z])(?=.*[A-Z])(?=.*[*.!@$%^&(){}[]:;<>,.?/~_+-=|\]).{8,25}$"

    username = StringField('username_label',
                           validators=[InputRequired(message="Username is required"), 
                           Length(min=4, max=25, message='Username must be between 4 to 25 characters'), ])

    password = PasswordField('password_label', validators=[InputRequired(message="Password is required"), 
                           Length(min=8, max=25, message='Password must be between 8 to 25 characters'),
                           ])

    confirm_pswd = PasswordField('confirm_pswd_label', validators=[InputRequired(message="Retype is required"), 
                           EqualTo('password', message='Passwords must match')])

    recaptcha = RecaptchaField()

    submit_button = SubmitField('Create')

    def validate_username(self, username):
        user_object = User.query.filter_by(username=username.data).first() #whatever user entered. This will be none if non matching

        if user_object:
            raise ValidationError('user name "{}" already exists. Select different name'.format(username.data))

# if you have already registered, login
class LoginForm(FlaskForm):

    """Login form"""

    username = StringField('username_label', validators=[InputRequired('user name required')])
    password = PasswordField('password_label', validators=[InputRequired(message='Password required'), invalid_credentials])

    submit_button = SubmitField('Login')


