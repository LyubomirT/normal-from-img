from flask import Flask, render_template, request, flash
import cv2
import numpy as np
import os

import uuid  # For generating unique filenames

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/', methods=['GET', 'POST'])
def index():
    normal_map_path = None
    height_map_path = None

    if request.method == 'POST':
        image = request.files['image']
        if image.filename != '':
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            normal_map = create_normal_map(img)
            height_map = create_height_map(img)
            
            # Generate unique filenames for the normal_map and height_map
            unique_id = str(uuid.uuid4().hex)
            normal_map_path = os.path.join('static', 'uploads', f'normal_map_{unique_id}.png')
            height_map_path = os.path.join('static', 'uploads', f'height_map_{unique_id}.png')

            # Clear the uploads folder
            for file in os.listdir(os.path.join('static', 'uploads')):
                os.remove(os.path.join('static', 'uploads', file))
            
            cv2.imwrite(normal_map_path, normal_map)
            cv2.imwrite(height_map_path, height_map)
            os.remove(image_path)  # Delete the uploaded image
        else:
            flash('Please upload an image')
    
    # remove the static folder from the path
    normal_map_path = normal_map_path[7:] if normal_map_path else None

    # remove the static folder from the path
    height_map_path = height_map_path[7:] if height_map_path else None

    return render_template('index.html', normal_map=normal_map_path, height_map=height_map_path)

def create_normal_map(img):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    normal_map = np.dstack((sobelx, sobely, np.ones_like(img)))
    normal_map = cv2.normalize(normal_map,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    return normal_map

def create_height_map(img):
    height_map = cv2.Laplacian(img,cv2.CV_64F)
    height_map = cv2.normalize(height_map,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    return height_map

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
