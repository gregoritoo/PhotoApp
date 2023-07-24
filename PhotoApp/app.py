from flask import Flask, render_template, request, jsonify
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
from load import DEVICE
UPLOAD_FOLDER = os.path.join('static', 'uploaded')
from PIL import Image
import io
import requests
from utils import base64_to_pil,np_to_base64

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
		print("file1",request.files)
		if 'file1'  in request.files:
			file1 = request.files['file1']
			print("file1",file1)
			path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
			print("file1",file1)
			file1.save(path)
			return render_template('image.html',user_image = path)
		else :
			return render_template('homepage.html') 
	else :
		return render_template('homepage.html')

    

@app.route('/predict',methods=['GET','POST'])
def predict():
	imgpath = request.form["path"]
	print("HERERERREE",imgpath)
	gt_image = cv2.imread(imgpath).astype(np.float32) / 255.
	gt_image =  cv2.resize(gt_image, (int(gt_image.shape[0]/2),int(gt_image.shape[1]/2)), cv2.INTER_CUBIC)
	gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
	netG = load_trained_model()
	print("model loaded !!")
	gt_tensor = transforms.ToTensor()(gt_image).unsqueeze(0)
	print("image transformed !!!")
	sr_img = netG(gt_tensor.to(DEVICE))
	print("image upscaled !!!")
	sr_image = transforms.ToPILImage()(sr_img.squeeze().detach().cpu())
	plt.imshow(sr_image)
	plt.axis('off')
	plt.savefig(imgpath.replace(".png","_upscaled.png"), bbox_inches='tight')
	return render_template('upscaled_image.html' ,user_image = imgpath[:-4])


@app.route("/predictapi",methods=["POST"])
def predict_binary():
	binary_img = request.files["image"]
	gt_image = Image.open(binary_img.stream)
	gt_image = cv2.cvtColor(np.array(gt_image), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
	gt_image =  cv2.resize(gt_image, (int(gt_image.shape[0]/2),int(gt_image.shape[1]/2)), cv2.INTER_CUBIC)
	gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
	netG = load_trained_model()
	gt_tensor = transforms.ToTensor()(gt_image).unsqueeze(0)
	sr_img = netG(gt_tensor.to(DEVICE))
	sr_image = np.array(transforms.ToPILImage()(sr_img.squeeze().detach().cpu()))
	sr_image = np_to_base64(sr_image)
	return jsonify({'msg': 'success', 'image': sr_image})

if __name__ == '__main__':
	app.run()
