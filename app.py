# from flask import Flask, render_template, request, flash, redirect, url_for
# import pickle
# import numpy as np
# from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.preprocessing.image import img_to_array
# from torchvision.io import read_image
# import torch
# import torchvision
# from torchvision.utils import draw_bounding_boxes
# import matplotlib.pyplot as plt
# from PIL import Image
# from tensorflow.keras.models import load_model
# import PIL
# import cv2
# import joblib
# import os
# import urllib.request
# from werkzeug.utils import secure_filename
# #from flask import Flask, render_template, request, flash, redirect, url_for
# # import pickle
# # import numpy as np
# import keras
# import keras_utils
# # from werkzeug.utils import secure_filename
# # #from keras.preprocessing.image import load_img
# # from tensorflow.keras.preprocessing.image import load_img
# # from tensorflow.keras.models import load_model
# # # from keras.preprocessing.image import img_to_array
# # from tensorflow.keras.preprocessing.image import img_to_array
# # from torchvision.io import read_image
# # import torch
# # import torchvision
# # from torchvision.utils import draw_bounding_boxes
# # import matplotlib.pyplot as plt
# # from PIL import Image
# # import PIL
# # import cv2
# # import joblib
# # import os
# # import urllib.request


# app = Flask(__name__,template_folder='/Users/SUHANI JAIN/Downloads/CB-WEBSITE-main/CB-WEBSITE-main/CB WEBSITE/templates',static_folder='/Users/SUHANI JAIN/Downloads/CB-WEBSITE-main/CB-WEBSITE-main/CB WEBSITE/static')
# #app = Flask(__name_)


# UPLOAD_FOLDER='static/images/'
# app.secret_key="secret key"
# app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER


# model = load_model(r"C:\Users\SUHANI JAIN\Downloads\detector.h5")
# #model=pickle.load(open('detector3.pkl','rb'))


# @app.route('/')
# def hello_world():
#     return render_template('index.html')

# @app.route('/predict',methods=['POST','GET'])
# def predict():
#     imagefile=request.files['imagefile']
#     h = int(request.form.get("height"))
#     w = int(request.form.get("width"))
    
    
#     filename=secure_filename(imagefile.filename)
#     imagefile.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
#     image_path=os.path.join(app.config['UPLOAD_FOLDER'],filename)
    

#     image=load_img(image_path,target_size=(224,224))
#     image=img_to_array(image)/255.0
#     image=np.expand_dims(image,axis=0)
#     preds = model.predict(image)[0]
    
#     (startX, startY, endX, endY) = preds
    
#     image = cv2.imread(image_path)
#     (ho,wo) = image.shape[:2]
#     startX = int(startX * wo)
#     startY = int(startY * ho)
#     endX = int(endX * wo)
#     endY = int(endY * ho)
#     cropped_image = image[int(startY):int(endY), int(startX):int(endX)]
    
#     resized_image = cv2.resize(cropped_image, (w,h))
#     resized_image = resized_image.astype('uint8')
#     img = Image.fromarray(resized_image, "RGB")
    
#     img.save(filename)
#     return render_template('index.html',filename=filename)


# @app.route('/display/<filename>')
# def display_image(filename):
#     return redirect(url_for('static',filename='images/'+filename),code=301)



# if __name__ =='_main_':
#     app.run(port=3000,debug=True)


from flask import Flask, render_template, request, flash, redirect, url_for
import pickle 
import numpy as np
import keras
import keras_utils
from werkzeug.utils import secure_filename
# from keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
# from keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import img_to_array
from torchvision.io import read_image
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import cv2
import joblib
import os
import urllib.request

#app = Flask(__name__,template_folder='CB-WEBSITE-main/CB WEBSITE/templates',static_folder='CB-WEBSITE-main/CB WEBSITE/static')
app = Flask(__name__,template_folder='templates',static_folder='static')
#model=pickle.load(open('detector3.pkl','rb'))
UPLOAD_FOLDER='static/images/'
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
model = load_model("detector.h5")


# f = open('detector.pkl', 'rb')
# model = pickle.load(f)
ALLOWED_EXTENSIONS= set(['png','jpg','jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
#     if 'imagefile' not in request.files:
#         flash('No file part')
#         return redirect(request.url)
    # imagefile=request.files['imagefile']
    # h = int(request.form.get("height"))
    # w = int(request.form.get("width"))
    
    # if imagefile.filename == '':
    #     flash("No image selected")
    #     return redirect(request.url)
    
    # if imagefile and allowed_file(imagefile.filename): 
    #   filename=secure_filename(imagefile.filename)
    #   imagefile.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
    #   image_path=os.path.join(app.config['UPLOAD_FOLDER'],filename)
 imagefile=request.files['imagefile']
 h = int(request.form.get("height"))
 w = int(request.form.get("width"))
    # filename=secure_filename(imagefile.filename)
    # image_path='static/images/' + filename
    # imagefile.save(filename)
 filename=secure_filename(imagefile.filename)
 imagefile.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
 image_path=os.path.join(app.config['UPLOAD_FOLDER'],filename)

 image=load_img(image_path,target_size=(224,224))
 image=img_to_array(image)/255.0
 image=np.expand_dims(image,axis=0)
 preds = model.predict(image)[0]
    
 (startX, startY, endX, endY) = preds
    
 image = cv2.imread(image_path)
 (ho,wo) = image.shape[:2]
 startX = int(startX * wo)
 startY = int(startY * ho)
 endX = int(endX * wo)
 endY = int(endY * ho)
 #cropped_image = image[int(startY):int(endY), int(startX):int(endX)]
    
 #resized_image = cv2.resize(cropped_image, (w,h))
 #resized_image = resized_image.astype('uint8')
 #img = Image.fromarray(resized_image, "RGB")
 #img.save(filename)
 im = Image.open(image_path)
 im1 = im.crop((startX, startY, endX, endY))
 im1 = im1.resize((w,h)) 
 im1.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
 image_path_new=os.path.join(app.config['UPLOAD_FOLDER'],filename)
     # flash("Image thumbnail is generated")
 return render_template('index.html',filename=filename)

    # else:
    #     flash("Allowed image types are- png, jpg, jpeg")
    #     return redirect(request.url)



   #return render_template('index.html',filename=filename)
    #return render_template('index.html',filename=image_path)

@app.route('/display/<filename>')
def display_image(filename):
     return redirect(url_for('static',filename='images/'+filename,code=301))    

if __name__ =='__main__':
    app.run(port=3000,debug=True)
    
