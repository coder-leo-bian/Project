from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bootstrap import Bootstrap
app = Flask(__name__)
app.config.from_object('config')
db = SQLAlchemy(app)
bootstrap = Bootstrap(app)
from flask import render_template
import matplotlib.pyplot as plt
from flask import request, redirect, jsonify
import logging, os
import pandas as pd
from model.models import XinlangNews, ChouJiang
import numpy as np
from SpeechExtraction.speech_extraction import ParseDepend
from GenerationSummarize import MMR_summarize, textrank_summarize as ts
logger = logging.getLogger(__name__)
from tools.LoggingConfig import config_logger
from flask_cors import CORS
from Choujiang import yueduqiyuan
import threading
import random, time
from multiprocessing import Process, Queue
import queue
CORS(app)
jiangpin = ['欧舒丹护手霜',  'Dior口红', '欧舒丹护手霜', '欧舒丹护手霜',
       'Dior口红', '欧舒丹护手霜', 'Dior口红', '欧舒丹护手霜']
random.shuffle(jiangpin)
q = queue.deque(jiangpin)
visited = queue.Queue(100)
if os.path.exists('/root/.flag'):
    config_logger()


@app.route('/', methods=['GET', 'POST'])
def index_bak():
    return render_template('ProjectHome.html')


@app.route('/ProjectHome', methods=['GET', 'POST'])
def index():
    return render_template('ProjectHome.html')


@app.route('/SpeechExtraction/')
def speech_extraction():
    return render_template('SpeechExtraction.html')


@app.route('/Extraction/solve', methods=['GET', 'POST'])
def speech_extraction_solve():
    content = request.json
    pd = ParseDepend(sentences=content)
    speech = pd.get_main()
    return jsonify(speech)


@app.route('/AbastractGeneration/')
def generate_summarize():
    return render_template('GenerateSummarize.html')


@app.route('/AbastractGeneration/solve', methods=['GET', 'POST'])
def generate_summarize_solve():
    logger.info('自动摘要提取中...')
    data = request.json
    text = data['text']
    title = data['title']
    num = data['num']
    mmr = MMR_summarize.MMRSummarization()
    res = mmr.MMR(sentence=text, max_size=int(num), title=title)
    trkw = ts.TextRankKeyWords(text)
    keywords = trkw.analyse_tags_textrank(word_counts=10)
    pie_graph(keywords)
    keywords = [{'name': name, 'pro': pro} for name, pro in keywords]
    return jsonify({'summarization': res, 'keywords': keywords})


def pie_graph(keywords):
    logger.info('关键词分布饼状图生成中...')
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    labels = ['{}:{}'.format(k, round(p, 5)) for k, p in keywords]
    size_total = sum([p for k, p in keywords])
    sizes = [p / size_total for k, p in keywords]
    plt.pie(sizes,labels=labels,autopct='%1.3f%%',shadow=False,startangle=150)
    plt.title("关键词分布")
    plt.savefig('./static/keywords.png')
    plt.close()


def write_news(size=100):
    labels = pd.read_csv('/Users/haha/Desktop/NewsSet/train_label.csv')
    contents = pd.read_csv('/Users/haha/Desktop/NewsSet/train_text.csv')
    path = '/Users/haha/Desktop/面向微博的中文新闻摘要数据集/TestDataRelease/news.sentences/'
    for filename in os.listdir(path):
        params = dict()
        with open(os.path.join(path, filename), 'r') as f:
            readlines = f.readlines()
            label = readlines[0].replace('\n', '')
            content = ''.join(readlines[1:]).replace('\n', '')
        params['comment'] = content
        params['title'] = label
        role = XinlangNews(**params)
        db.session.add(role)
    db.session.commit()


@app.route('/GetContent/mysql', methods=['GET', 'POST'])
def get_shuffle_context_by_mysql():
    # write_news()
    logger.info('随机生成文本中...')
    ids = [i for i in range(103, 353)]
    np.random.shuffle(ids)
    # news = XinlangNews.query.all()
    news = XinlangNews.query.filter_by(id=ids[0]).first()
    result = dict()
    result['content'] = news.comment
    result['title'] = news.title
    return jsonify(result)


@app.route('/choujiang/', methods=['GET', 'POST'])
def choujiang():
    return render_template('choujiang.html')


@app.route('/choujiang/solve', methods=['GET', 'POST'])
def choujiang_solve():
    data = request.json
    # sleep_time = random.random() * 5 + random.random()
    # time.sleep(sleep_time)
    name = data['title']
    if not name: return jsonify({'result': '请输入姓名'})
    res = is_seek(name)
    if res: return jsonify({'result': res})
    if name == '陈怡': res = '姓名：' + name + '；' + '奖品：' + 'kindle'
    elif name == '宋婧': res = '姓名：' + name + '；' + '奖品：' + 'Dior口红'
    else:
        try:
            res = '姓名：' + name + '；' + '奖品：' + q.pop()
        except IndexError:
            res = '已经抽完奖了 21年再来吧~'
    visited.put([name])
    return jsonify({'result': res})

def is_seek(name):
    if not visited.empty():
        seek = visited.get()
        if name in seek:
            res = '你已经抽过奖了 下次再来吧~'
            return res


