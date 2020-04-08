'''

DISLAIMER: RUN THE FOLLOWING COMMANDS BEFORE RUNNING THIS CODE

# > pip install opencv-python==3.4.2.16
# > pip install opencv-contrib-python==3.4.2.16
# > pip install numpy scipy matplotlib imgurpython Image

'''

######################################## IMPORTS ########################################

import cv2 # OpenCV
import numpy as np # Numpy
from PIL import Image, ImageDraw, ImageFont # Python Image Library
from sklearn.metrics.pairwise import cosine_similarity
import hist_correlation as histo_correlation
from tkinter.filedialog import askopenfilename # For opening custom files
from imgurpython import ImgurClient # Imgur Client for uploading images and retrieving URL to be used in Azura OCR functionality
import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from io import BytesIO
from scipy.ndimage import filters

######################################## EXPLANATIONS ########################################

# Short explanation of our methods/functions.
# There are two ways of extracting features from an image: Image Descriptor (used in this project) and Neural Nets.

# In this project SIFT (Scale-invariant feature transform) is used...
# The scale-invariant feature transform (SIFT) is a feature detection algorithm in computer vision to detect and...
# describe local features in images. Local features refer to a pattern or distinct structure found in an image, such...
# as a point, edge, or small image patch.

# For any object in an image, interesting points on the object can be extracted to provide a...
# "feature description" of the object. This description, extracted from a training image, can...
# then be used to identify the object when attempting to locate the object in a test image...
# containing many other objects. To perform reliable recognition, it is important that the...
# features extracted from the training image be detectable even under changes in image scale,...
# noise and illumination. Such points usually lie on high-contrast regions of the image, such as object edges.

# SIFT can robustly identify objects even among clutter and under partial occlusion, because the SIFT feature...
# descriptor is invariant to uniform scaling, orientation, illumination changes, and partially invariant to affine distortion.

######################################## MAIN PROGRAM ########################################


client_id = "ddac09735b20dd7"
client_secret = "0ae11f510ea90e162f412faac0bc92718bbf04eb"

client = ImgurClient(client_id, client_secret)

# This function bascially utilizes the imgur python api to upload the user selected picture onto the imgur server and return the image URL from server response
def upload(image_path):
    print('Uploading image...')
    image = client.upload_from_path(image_path, config=None, anon=True)

    link = image['link']
    return link

def check_auth(image_path):
    link = upload(image_path)
    print("Imgur image link: %s\n" %(link))

    subscription_key = "f1d8f893807c4b4e89b5493a7b5d1497"
    assert subscription_key

    vision_base_url = "https://ia-project.cognitiveservices.azure.com/vision/v2.1/"

    ocr_url = vision_base_url + "ocr"

    image_url = link

    headers = {'Ocp-Apim-Subscription-Key': subscription_key}
    params  = {'language': 'en', 'detectOrientation': 'true'}
    data    = {'url': image_url}
    response = requests.post(ocr_url, headers=headers, params=params, json=data)
    response.raise_for_status()

    analysis = response.json()

    line_infos = [region["lines"] for region in analysis["regions"]]
    word_infos = []
    for line in line_infos:
        for word_metadata in line:
            for word_info in word_metadata["words"]:
                word_infos.append(word_info)

    flag = 0
    print("Detected Words: \n")
    for word in word_infos:
        text = word["text"]
        if(text.find("child") != -1):
            flag = 1
        print(text+" ")
    return flag

def feature_extraction(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Other algos such as SURF, KAZE, ORB etc can also be used
    algo = cv2.xfeatures2d.SIFT_create()

    # Detect Keypoints (or SIFT points)
    # Images have different properties. Hence it is possible that number of keypoints will be different for different images.
    # The idea of SIFT is to to find keypoints (they are nothing but interest points that stands out in an image) in an image that are identifiable under many conditions.
    # In fact these keypoints should be identifiable despite scaling, rotation, and lighting differences.
    keypoints = algo.detect(image)

    vector_size = 512

    # Getting 512 biggest keypoints using keypoint response value
    keypoints = sorted(keypoints, key = lambda x : -x.response)
    keypoints = keypoints[:vector_size]

    # computing descriptors vector
    # When you wish to find similarities between images you do a pairwise comparison of the keypoint descriptors. The descriptor is a 128x1 vector assigned to every keypoint.
    keypoints, descriptors = algo.compute(image, keypoints)

    # Flatten all descriptor vector into one big vector which is our feature vector
    descriptors = descriptors.flatten()

    return descriptors

# Main function
def main():
    Test_Image_Path = askopenfilename()

    status_code = check_auth(Test_Image_Path)

    if(status_code == 1):
        print("The note is a fake one!!!")
        return 0;

    # Descriptor vectors
    sample_note1 = feature_extraction(Test_Image_Path)
    note_50_euro = feature_extraction('./database/50_Euro.jpg')
    note_100_euro = feature_extraction('./database/100_Euro.jpg')
    note_200_euro = feature_extraction('./database/200_Euro.jpg')
    note_100_rupee = feature_extraction('./database/100_Rupees.jpg')
    note_2000_rupee = feature_extraction('./database/2000_Rupees.jpg')
    note_500_rupee = feature_extraction('./database/500_Rupees.jpg')

    # Cosine Similarities
    A1 = cosine_similarity(sample_note1.reshape(1,-1), note_100_euro.reshape(1,-1))
    B1 = cosine_similarity(sample_note1.reshape(1,-1), note_100_rupee.reshape(1,-1))
    C1 = cosine_similarity(sample_note1.reshape(1,-1), note_2000_rupee.reshape(1,-1))
    D1 = cosine_similarity(sample_note1.reshape(1,-1), note_500_rupee.reshape(1,-1))

    # Correlation using histograms
    A2 = histo_correlation.calculate(Test_Image_Path, './database/100_Euro.jpg')
    B2 = histo_correlation.calculate(Test_Image_Path, './database/100_Rupees.jpg')
    C2 = histo_correlation.calculate(Test_Image_Path, './database/2000_Rupees.jpg')
    D2 = histo_correlation.calculate(Test_Image_Path, './database/500_Rupees.jpg')

    print('Test for 100 Euro Note | ' + 'Cosine_Similarity - ' + str(A1[0][0]) + ', Histogram_Correlation - ' + str(A2) + ', Product - ' + str(A1[0][0] * A2))
    print('Test for 100 Rupee Note | ' + 'Cosine_Similarity - ' + str(B1[0][0]) + ', Histogram_Correlation - ' + str(B2) + ', Product - ' + str(B1[0][0] * B2))
    print('Test for 2000 Rupee Note | ' + 'Cosine_Similarity - ' + str(C1[0][0]) + ', Histogram_Correlation - ' + str(C2) + ', Product - ' + str(C1[0][0] * C2))
    print('Test for 500 Rupee Note | ' + 'Cosine_Similarity - ' + str(D1[0][0]) + ', Histogram_Correlation - ' + str(D2) + ', Product - ' + str(D1[0][0] * D2))
    print('\n')

    image = Image.open(Test_Image_Path)
    font = ImageFont.truetype("arial.ttf", size=35)
    draw = ImageDraw.Draw(image)
    (x, y) = ((image.size[0] / 8), image.size[1] / 2)
    color = 'rgb(255,0,0)'

    if((A1[0][0] * A2) >= (B1[0][0] * B2) and (A1[0][0] * A2) >= (C1[0][0] * C2)):
        message = 'This is a 100 Euro Note'
        draw.text((x, y), message, fill=color, font=font)

        print('Note is a 100 Euro Note')
    elif((B1[0][0] * B2) >= (A1[0][0] * A2) and (B1[0][0] * B2) >= (C1[0][0] * C2)):
        message = 'This is a 100 Rupee Note'
        draw.text((x, y), message, fill=color, font=font)

        print('Note is a 100 Rupee Note')
    elif((C1[0][0] * C2) >= (B1[0][0] * B2) and (C1[0][0] * C2) >= (A1[0][0] * A2)):
        message = 'This is a 2000 Rupee Note'
        draw.text((x, y), message, fill=color, font=font)

        print('Note is a 2000 Rupee Note')

    image.save('./output/output.jpg')
    image = Image.open('./output/output.jpg')
    image.show()


if __name__ == "__main__":
    main()
