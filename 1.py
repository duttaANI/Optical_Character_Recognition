from cv2 import  COLOR_BGR2GRAY
import json


import os
curpath = os.path.abspath(os.getcwd())
print("curpath :",curpath)


input_images_path = os.path.join(curpath,"inputImages")

import cv2
from pdf2image import convert_from_path

# Read the pdf
pdfFile = "Sanskrit_Text.pdf"

pages = convert_from_path(pdfFile, 500)

# seperate pages of pdf to images
for i, image in enumerate(pages):
    fname = 'image'+str(i)+'.png'
    image.save(os.path.join(input_images_path , fname ), "PNG")

# choose any image of the pdf
image = cv2.imread( os.path.join(input_images_path , "image1.png" ) )
copy_image = image.copy()



gray = cv2.cvtColor(image,COLOR_BGR2GRAY )



# blur1 = cv2.GaussianBlur(gray,(5,5),0)

blur = cv2.blur(gray,(200,1),0) # (79,1) onwards works well

cv2.imwrite("index_blur.png",blur)

thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

cv2.imwrite("index_thresh.png",thresh)


kernal = cv2.getStructuringElement(cv2.MORPH_RECT,(3,13))

cv2.imwrite("index_kernal.png",kernal)

dilate = cv2.dilate(thresh,kernal, iterations=1)

cv2.imwrite("index_dilate.png",dilate)

cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = cnts[0] if len(cnts) == 2 else cnts[1]

cnts = sorted(cnts, key = lambda x: cv2.boundingRect(x)[0])

output_path = os.path.join(curpath,"finalOutput")

with open('coordinates_of_the_bounding_box.json', 'w') as f:
    json.dump({}, f)

var =0
height_adjustment_for_matrae_lower = 20
height_adjustment_for_matrae_upper = 20
for c in cnts:
    x , y, w, h = cv2.boundingRect(c)
    if w>100 and h>50: # removing small boxes
        var = var + 1
        new_data={
            "top_left": [x, y-height_adjustment_for_matrae_upper],
            "top_right": [x, y+h+height_adjustment_for_matrae_lower],
            "bottom_left": [x+w, y-height_adjustment_for_matrae_upper],
            "bottom_right": [x+w, y+h+height_adjustment_for_matrae_lower]
        }
        
        with open('coordinates_of_the_bounding_box.json', 'r+') as file:
            
            # First we load existing data into a dict.
            file_data = json.load(file)
            # Join new_data with file_data inside emp_details
            file_data["box"+str(var)]=new_data
            # Sets file's current position at offset.
            file.seek(0)
            # convert back to json.
            json.dump(file_data, file, indent = 4)



        cv2.rectangle(image,(x,y-height_adjustment_for_matrae_upper),(x+w,y+h+height_adjustment_for_matrae_lower),(36,255,12),2)
        crop = copy_image[y-height_adjustment_for_matrae_upper:y+h+height_adjustment_for_matrae_lower, x:x+w]
        fname = 'output_image'+str(var)+'.jpg'
        cv2.imwrite(os.path.join(output_path , fname ) ,crop)

cv2.imwrite("index_bbox.png",image)


