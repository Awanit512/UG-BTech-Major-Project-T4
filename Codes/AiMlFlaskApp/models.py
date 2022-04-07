from flask_login import UserMixin
from __init__ import db

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True) # primary keys are required by SQLAlchemy
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(1000))



class InappropriateContentTweet(db.Model):
    id = db.Column(db.Integer, primary_key=True) # primary keys are required by SQLAlchemy
    tweet = db.Column(db.String(250))
    subjectP = db.Column(db.String(100))
    ObjectQ = db.Column(db.String(100))
    ConnectorR = db.Column(db.String(50))
    classToxic = db.Column(db.Integer)
    classSevereToxic = db.Column(db.Integer)
    classObscene = db.Column(db.Integer)
    classThreat = db.Column(db.Integer)
    classInsult = db.Column(db.Integer)
    classIdentityHate =  db.Column(db.Integer)



    