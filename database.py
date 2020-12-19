from flask_sqlalchemy import SQLAlchemy

#Initialize db for inheritance
db = SQLAlchemy()

class User(db.Model):
    """
    User model. To connect from terminal:
    psql 'heroku credential url'
    CREATE TABLE users(id, SERIAL.... VARCHAR )
    use \dt to check current tables
    \d users to see specific entries the table
    """ 

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(25), unique=True, nullable=False) #nullable = can it be empty?
    password = db.Column(db.String(), nullable=False)


if __name__ == "__main__":
    print('fail?')
