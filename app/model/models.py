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


db.create_all()