import numpy as np
import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename, redirect
from gevent.pywsgi import WSGIServer
from flask import send_from_directory


model = load_model("crude_oil.h5")

app = Flask (__name__,template_folder='templates')
@app.route('/')
def home():
    return render_template("index.html")
@app.route('/about')
def home1():
    return render_template("index.html")
@app.route('/predict')
def home2():
    return render_template("web.html")



@app.route('/predict', methods=['POST'] )
def login() :
     x_input=str(request.form['year'])
     x_input=x_input.split(',')
     print(x_input)
      
     for i in range(0, len(x_input)):
         x_input[i] = float(x_input[i])
     print(x_input)
     x_input=np.array(x_input).reshape(1,-1)
     temp_input=list(x_input)
     temp_input=temp_input[0].tolist()
     lst_output=[]
     n_steps=10
     i=0
     while(i<1):
        if(len(temp_input)>10):
            x_input=np.array(temp_input[1:])
            print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1,n_steps,1))
            yhat = model.predict(x_input,verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
        print(lst_output)
        return render_template("web.html", showcase = 'The next day predicted value is:'+str(lst_output))

port = os.getenv('VCAP_APP_PORT','8080')

if __name__ == '__main__' :
    app.secret_key = os.urandom(12)
    app.run(debug = True,port=port,host='0.0.0.0')
