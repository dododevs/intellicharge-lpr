FROM ubuntu:latest
COPY . /app
RUN apt -y update
RUN apt -y upgrade

RUN apt install -y git build-essential python3 python3-pip wget libgl1

RUN git clone https://github.com/dododevs/intellicharge-lpr /app/intellicharge-lpr
WORKDIR /app/intellicharge-lpr
RUN git submodule init
RUN git submodule update
RUN python3 -m pip install -r PaddleOCR/requirements.txt
RUN python3 -m pip install -r requirements.txt

WORKDIR /app/intellicharge-lpr/PaddleOCR
RUN mkdir -p inference/cls inference/det inference/reg
WORKDIR /app/intellicharge-lpr/PaddleOCR/inference/det
RUN wget https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar
RUN tar xvf en_PP-OCRv3_det_infer.tar && rm *.tar
WORKDIR /app/intellicharge-lpr/PaddleOCR/inference/cls
RUN wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
RUN tar xvf ch_ppocr_mobile_v2.0_cls_infer.tar && rm *.tar
WORKDIR /app/intellicharge-lpr/PaddleOCR/inference/reg
RUN wget https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar
RUN tar xvf en_PP-OCRv3_rec_infer.tar && rm *.tar

WORKDIR /app/intellicharge-lpr
