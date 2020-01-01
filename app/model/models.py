from flask import Flask
from flask_sqlalchemy import SQLAlchemy
app = Flask(__name__)
app.config.from_object('config')
db = SQLAlchemy(app)


class XinlangNews(db.Model):
    __tablename__ = 'xinlang_news'

    id = db.Column(db.Integer, primary_key=True)
    comment = db.Column(db.String(10000))
    title = db.Column(db.String(256))


class SpeechNews(db.Model):
    __tablename__ = 'speech_news'

    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(5000))


class ChouJiang(db.Model):
    __tablename__ = 'choujiang'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    got_name = db.Column(db.String(100))
    got_date = db.Column(db.Date())


db.create_all()