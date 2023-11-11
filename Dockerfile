FROM ubuntu:latest
COPY . /app
RUN apt update
RUN apt upgrade

RUN apt install python3 python3-pip wget

RUN git clone https://github.com/dododevs/intellicharge-lpr /app
WORKDIR /app/intellicharge-lpr
RUN python3 -m pip install -r requirements.txt

WORKDIR /app
RUN wget https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar -O 