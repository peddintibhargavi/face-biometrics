import os
import urllib.request as urllib
import imageio
import cv2
import numpy as np
import onnx
import sys
import onnxruntime

model_url = "https://github.com/Kazuhito00/Skin-Clothes-Hair-Segmentation-using-SMP/raw/043fc08bf0ed845b36485a115085104c6459cbc4/02.model/DeepLabV3Plus(timm-mobilenetv3_small_100)_1366_2.16M_0.8297/best_model.onnx"
model_path = "hair_segmentation.onnx"

def download_github_model(model_url, model_path):
    try:
        if not os.path.exists(model_path):
            print(f"Downloading model from {model_url}")
            urllib.urlretrieve(model_url, model_path)
            print(f"Model downloaded to {model_path}")
            file_size = os.path.getsize(model_path)
            print(f"Model file size: {file_size} bytes")
        else:
            print(f"Model already exists at {model_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please download the model manually from the GitHub repository.")
        sys.exit(1)

class HairSegmentation():

    def __init__(self, webcam_width, webcam_height):
        self.model = self.initialize_model()
        self.fire_img_num = 0

    def __call__(self, image):
        return self.segment_hair(image)

    def initialize_model(self):
        download_github_model(model_url, model_path)

        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            sys.exit(1)

        try:
            self.session = onnxruntime.InferenceSession(model_path)
        except Exception as e:
            print(f"Error creating InferenceSession: {e}")
            sys.exit(1)

        self.getModel_input_details()
        self.getModel_output_details()

    def segment_hair(self, image):
        input_tensor = self.prepare_input(image)
        outputs = self.inference(input_tensor)
        hair_mask = self.process_output(outputs)
        return hair_mask

    def prepare_input(self, image):
        self.img_height, self.img_width, self.img_channels = image.shape
        
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(input_image, (self.input_width, self.input_height))

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        input_image = (input_image / 255 - mean) / std

        input_image = input_image.transpose(2, 0, 1)
        input_tensor = input_image[np.newaxis,:,:,:]   

        return input_tensor.astype(np.float32)

    def inference(self, input_tensor):
        return self.session.run(self.output_names, {self.input_name: input_tensor})

    def process_output(self, outputs):  
        hair_mask = np.squeeze(outputs[0])
        hair_mask = hair_mask.transpose(1, 2, 0)
        hair_mask = hair_mask[:,:,2]
        hair_mask = cv2.resize(hair_mask, (self.img_width, self.img_height))
        return np.round(hair_mask).astype(np.uint8)

    def getModel_input_details(self):
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def getModel_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[0].name]
        self.output_shape = model_outputs[0].shape
        self.output_height = self.output_shape[2]
        self.output_width = self.output_shape[3]

def extract_masked_regions(image, mask):
    masked_image = np.copy(image)
    masked_image[mask > 0] = image[mask > 0]
    masked_image[mask <= 0] = 255
    return masked_image

def extract_masked_regions_overlay(image, mask):
    masked_image = np.copy(image)
    masked_image[mask > 0] = 255
    return masked_image

def find_contours_rectangle(mask):
    contours, hierarchy = cv2.findContours(mask*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        min_left = mask.shape[1]
        min_top = mask.shape[0]
        max_right = 0 
        max_bottom = 0 
        for contour in contours:
            left, top, rect_width, rect_height = cv2.boundingRect(contour)
            bottom = top + rect_height
            right = left + rect_width

            min_left = min([min_left, left])
            min_top = min([min_top, top])
            max_right = max([max_right, right])
            max_bottom = max([max_bottom, bottom])

        contour_rectangle = [min_left, min_top, max_right, max_bottom]
    else:
        contour_rectangle = [0, 0, mask.shape[1], mask.shape[0]]

    return contour_rectangle

if __name__ == '__main__':
    image = cv2.imread(sys.argv[1])
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hair_segmentation = HairSegmentation(rgb_image.shape[1], rgb_image.shape[0])

    hair_mask = hair_segmentation(rgb_image)
    cv2.imwrite('mediapipe_hairmask.png', extract_masked_regions(image, hair_mask))
