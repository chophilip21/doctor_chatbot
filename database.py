from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

#Initialize db for inheritance
db = SQLAlchemy()

class User(UserMixin, db.Model):
    """
    ? Here, we define what user is. 
    UserMixin inheritance for managing login session
    To connect from terminal:
    psql 'heroku credential url'
    use \dt to check current tables
    use table users; to see currently registered users.
    """ 

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(25), unique=True, nullable=False) #nullable = can it be empty?
    password = db.Column(db.String(), nullable=False)


if __name__ == "__main__":
    print('fail?')
