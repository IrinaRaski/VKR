import numpy as np
from flask import Flask, render_template, request
import pickle

app_3 = Flask(__name__, template_folder='templates')
path = r'model/model.pkl'
url = r'D:\PROJECTS\VKR\html\background.jpg'
model = pickle.load(open(path, 'rb'))


@app_3.route('/', methods=['get', 'post'])
def main():
    if request.method == 'GET':
        return render_template('main.html')

    if request.method == 'POST':
        prediction_text = [float(x) for x in request.form.values()]
        x_new = [np.array(prediction_text)]
        prediction = model.predict(x_new).flatten()
        result = f'{str(prediction)[1:-1]}'
        return render_template('main.html', result=result)
