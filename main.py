import os
import sys
import subprocess

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, 'PaddleOCR')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import io
import cv2
import copy
import base64
import numpy as np
import json
import time
import logging
import pika
from PIL import Image
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.utils.logging import get_logger
from tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image, get_minarea_rect_crop
logger = get_logger()


class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

        self.args = args
        self.crop_image_res_index = 0

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(output_dir,
                             f"mg_crop_{bno+self.crop_image_res_index}.jpg"),
                img_crop_list[bno])
            logger.debug(f"{bno}, {rec_res[bno]}")
        self.crop_image_res_index += bbox_num

    def __call__(self, img, cls=True):
        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}

        if img is None:
            logger.debug("no valid image provided")
            return None, None, time_dict

        start = time.time()
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        time_dict['det'] = elapse

        if dt_boxes is None:
            logger.debug("no dt_boxes found, elapsed : {}".format(elapse))
            end = time.time()
            time_dict['all'] = end - start
            return None, None, time_dict
        else:
            logger.debug("dt_boxes num : {}, elapsed : {}".format(
                len(dt_boxes), elapse))
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.args.det_box_type == "quad":
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = get_minarea_rect_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls and cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            time_dict['cls'] = elapse
            logger.debug("cls num  : {}, elapsed : {}".format(
                len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        time_dict['rec'] = elapse
        logger.debug("rec_res num  : {}, elapsed : {}".format(
            len(rec_res), elapse))
        if self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list,
                                   rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        end = time.time()
        time_dict['all'] = end - start
        return filter_boxes, filter_rec_res, time_dict

def sorted_boxes(dt_boxes):
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes

DET_MODEL = "./PaddleOCR/inference/det/en_PP-OCRv3_det_infer/"
CLS_MODEL = "./PaddleOCR/inference/cls/ch_ppocr_mobile_v2.0_cls_infer/"
REC_MODEL = "./PaddleOCR/inference/reg/en_PP-OCRv3_rec_infer/"
REC_CHAR_DICT = "./PaddleOCR/ppocr/utils/en_dict.txt"

def detect_plate(img, det_model=DET_MODEL, cls_model=CLS_MODEL, rec_model=REC_MODEL, rec_char_dict=REC_CHAR_DICT):
    sys.argv = [
        sys.argv[0],
        f'--det_model_dir={det_model}',
        f'--cls_model_dir={cls_model}',
        f'--rec_model_dir={rec_model}',
        f'--rec_char_dict_path={rec_char_dict}'
    ]
    args = utility.parse_args()
    text_sys = TextSystem(args)
    img = cv2.cvtColor(np.array(Image.open(io.BytesIO(base64.b64decode(img)))), cv2.COLOR_BGR2RGB)
    dt_boxes, rec_res, time_dict = text_sys(img)

    # res = [{
    #     "transcription": rec_res[i][0],
    #     "points": np.array(dt_boxes[i]).astype(np.int32).tolist(),
    # } for i in range(len(dt_boxes))]

    return [(text, score) for text, score in rec_res]

credentials = pika.PlainCredentials('admin', 'admin')
connection = pika.BlockingConnection(pika.ConnectionParameters(host='192.168.94.1', credentials=credentials))
channel = connection.channel()
channel.exchange_declare(exchange='images', exchange_type='fanout')
result = channel.queue_declare(queue='', exclusive=True)
queue_name = result.method.queue
channel.queue_bind(exchange='images', queue=queue_name)

def on_new_image(ch, method, props, body):
    results = detect_plate(body)
    if not results or len(results[0][0]) != 7:
        return
    plate = results[0][0]
    channel.exchange_declare(exchange="plate_detected", exchange_type="fanout")
    channel.basic_publish(exchange="plate_detected", routing_key="plate_detected", body=plate)

channel.basic_consume(queue_name, on_new_image, True)
channel.start_consuming()
