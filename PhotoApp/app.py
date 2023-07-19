from flask import Flask, render_template, request
import cv2
import numpy as np
import re
import sys 
import os
import base64
sys.path.append(os.path.abspath("./model"))
from load import load_trained_model
from torchvision import transforms
import matplotlib.pyplot as plt 

UPLOAD_FOLDER = os.path.join('static', 'uploaded')

app = Flask(__name__,template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index_view():
    return render_template('home.html')

def convertImage(imgData1):
	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
	with open('output.png','wb') as output:
	    output.write(base64.b64decode(imgstr))
	    

	    
@app.route('/load',methods=['GET','POST'])
def load_image():
	if request.method == 'POST':
		if 'file1' not in request.files:
			return 'there is no file1 in form!'
		file1 = request.files['file1']
		path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
		file1.save(path)
		return render_template('image.html',user_image = path)
	else :
		return '''
		<body style="background-color:#000033 ;" >
		<div align="center";  >
		<h1 style="color:#CCCCD6; font-family: Courier;" >Upload new File</h1>
		<form method="post" enctype="multipart/form-data">
		<input type="file" name="file1">
		<input type="submit">
		</div>
		</form>
		</body>
    '''

    

@app.route('/predict',methods=['GET','POST'])
def predict():
	imgpath = request.form["path"]
	print("HERERERREE",imgpath)
	gt_image = cv2.imread(imgpath).astype(np.float32) / 255.
	#gt_image =  cv2.resize(gt_image, (int(gt_image.shape[0]/8),int(gt_image.shape[1]/8)), cv2.INTER_CUBIC)
	gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
	netG = load_trained_model()
	print("model loaded !!")
	gt_tensor = transforms.ToTensor()(gt_image).unsqueeze(0)
	print("image transformed !!!")
	sr_img = netG(gt_tensor)
	print("image upscaled !!!")
	sr_image = transforms.ToPILImage()(sr_img.squeeze().detach().cpu())
	plt.imshow(sr_image)
	plt.axis('off')
	plt.savefig(imgpath.replace(".png","_upscaled.png"), bbox_inches='tight')
	return render_template('upscaled_image.html' ,user_image = imgpath[:-4])

if __name__ == '__main__':
    app.run()