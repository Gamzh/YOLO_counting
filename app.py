from flask import Flask, Response,render_template
import cv2
import numpy as np

import numpy as np

import tracker
from detector import Detector
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('1.html')


# @app.route('/video_feed')
# # def video_feed():
# #     return render_template("1.html",video = Response(genWeb(VideoCamera()),
# #                                                mimetype='multipart/x-mixed-replace; boundary=frame'))
# def video_feed():
#     return Response(genWeb(VideoCamera()),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    down_count = 0
    # 离开数量
    up_count = 0
    app.run(host='0.0.0.0', threaded=True)


