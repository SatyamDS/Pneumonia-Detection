import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import cv2
#from tensorflow.keras.models import load_model
app = Flask(__name__)

from commons import get_tensor,get_model
#from inference import get_flower_name

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        return render_template('index.html', value='hi')
    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            print('File not Uploaded')
            return
        file = request.files['file']
        #image = cv2.imread(file).astype('uint8')
        image = file.read()
        #image =np.asarray(file)
        
        model = get_model()
        
        image = get_tensor(image)
        
        output = model.predict(image)
        
        if(output == [[0.]]):
            name = "NORMAL/ PNUEMONIA NEGATIVE"
        else:
            name = "PNUEMONIA POSITIVE"
            
        return render_template('result.html', result = name)
        

if __name__ == '__main__':
    app.run(debug=True)