# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 11:58:21 2020
@author: nnig9
"""

from flask import Flask, jsonify
import predictbnp as pb
import importlib
from markupsafe import escape




bpapp = Flask(__name__)

# API ================================================================
@bpapp.route('/')
def predict():
    importlib.reload(pb)
    predict1=str(pb.predict)
    actual_t=str(pb.actual.index[0])
    actual_v=str(pb.actual.vwap[0])
    #pred = "Предсказание -"+predict1+"\n Время -"+actual_t+"__Актуальное -"+actual_v
    return f"""
    <h1>Предсказание - {escape(predict1)}!</h1>
    <h2>Время - {escape(actual_t)}!</h2>
    <h2>Актуальное значение -  {escape(actual_v)}!</h2>

    """


@bpapp.route('/bot', methods=['GET'])

def get_list(): 
    importlib.reload(pb)
    predict1=str(pb.predict)
    return jsonify(predict1)



if __name__ == '__main__': 
    bpapp.run(host='0.0.0.0', port=5000)