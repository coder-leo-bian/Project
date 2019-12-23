from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bootstrap import Bootstrap
app = Flask(__name__)
app.config.from_object('config')
db = SQLAlchemy(app)
bootstrap = Bootstrap(app)
from flask import render_template
from flask import request, redirect, jsonify
import logging, re
from .GenerationSummarize import MMR_summarize
logger = logging.getLogger()


@app.route('/', methods=['GET', 'POST'])
def index_bak():
    return render_template('ProjectHome.html')


@app.route('/ProjectHome', methods=['GET', 'POST'])
def index():
    return render_template('ProjectHome.html')


# @app.route('/SpeechExtraction/')
# def speech_extraction():
#     return render_template('SpeechExtraction.html')
#
#
# @app.route('/SpeechExtraction/solve', methods=['GET', 'POST'])
# def speech_extraction_solve():
#     input_text = request.form.get('data')
#     print(input_text)
#     return render_template('SpeechExtraction.html')

@app.route('/AbastractGeneration/')
def generate_summarize():
    return render_template('GenerateSummarize.html')


@app.route('/AbastractGeneration/solve', methods=['GET', 'POST'])
def generate_summarize_solve():
    data = request.json
    text = data['text']
    title = data['title']
    num = data['num']
    mmr = MMR_summarize.MMRSummarization()
    res = mmr.MMR(sentence=text, max_size=int(num))
    # Topic = MMR_summarize.Topic(text)
    # topic = Topic.topic()
    return jsonify({'summarization': res})



