import pickle
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
app = Flask(__name__)
CORS(app)

@app.route('/prediction', methods = ['POST'])
def heart_pred():
    #print(request.json)
    x_test = request.json["a"]
    # x_test.reshape(-1,1)
    print(request.json)#form = InputForm(request.form)
    clfr = pickle.load(open("finalmodel.sav", 'rb'))
    y_pred = clfr.predict(x_test)
    
    return jsonify({"out":str(y_pred)})

if __name__ == '__main__':  
    app.run(debug = True) 