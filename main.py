from Generator import Generator
import torch
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import os
from flask import Flask, request
from flask_restful import Resource, Api
from json import dumps
from flask import jsonify
from flask_cors import CORS, cross_origin
from flask import send_file
from tempfile import NamedTemporaryFile
from shutil import copyfileobj
from io import BytesIO

device = torch.device('cpu')
nz = 100

generator = Generator(ngpu = 0)
generator.load_state_dict(torch.load('generator.pth', map_location=device))
generator = generator.to(device)

def generate_image():
	fixed_noise = torch.randn(64, nz, 1, 1, device=device)
	with torch.no_grad():
	    fake = generator(fixed_noise).detach().cpu()
	    fake = fake[0]
	save_image(fake, 'temp.png')
	img = Image.open('temp.png')
	p = transforms.Compose([transforms.Scale((300,300))])
	img = p(img)
	os.remove('temp.png')
	return img

def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'PNG', quality=70)
    img_io.seek(0)
    pil_img.close()
    return send_file(img_io, mimetype='image/png')

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
def serve_img():
    img = generate_image()
    return serve_pil_image(img)

if __name__ == "__main__":
	app.run()