import os

import cv2 as cv


class ImagePreProcessor:
    _input_dir = 'input'
    _output_dir = 'output'
    _pwd = os.getcwd()

    def __init__(self, input_dir=None, output_dir=None):
        if input_dir:
            self._input_dir = input_dir
        if output_dir:
            self._output_dir = output_dir

    def process_image(self, imagefile):
        img = cv.imread(self._input_dir + '/' + imagefile, 0)
        edges = cv.Canny(img, 150, 100)  # 200,100  150,100
        output_dir_path = os.path.join(self._pwd + '/' + self._output_dir)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        output_file_path = os.path.join(output_dir_path, 'image_' + str(imagefile.split('.')[0]) + '.png')
        cv.imwrite(output_file_path, edges)

    def start(self):
        images = os.listdir(self._input_dir)
        for image in images:
            self.process_image(image)
