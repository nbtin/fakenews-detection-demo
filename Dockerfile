FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
SHELL ["/bin/bash", "-c"]

# requirements for TruFor running
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y apt-utils wget unzip git build-essential gcc python3-dev libglib2.0-0 libsm6 libxext6 libxrender-dev && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip

# RUN python -m venv venv
# RUN source venv/bin/activate

RUN pip install tqdm yacs>=0.1.8 timm>=0.5.4 numpy==1.21.5
RUN pip install --upgrade gdown

# Install pycocotools from GitHub
# RUN pip install --force-reinstall pycocotools

# add source code of the whole demo
ADD ./ /thesis-demo/
WORKDIR /thesis-demo/

# requirements for cheapfakes running
RUN git clone https://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@github.com/nbtin/cheapfakes_detection_SCID2024.git
# RUN cd /thesis-demo/cheapfakes_detection_SCID2024
# RUN sed '1d' /thesis-demo/cheapfakes_detection_SCID2024/requirements.txt | xargs -I {} pip install {}
# RUN sed '1d' /thesis-demo/cheapfakes_detection_SCID2024/requirements.txt | xargs -I {} pip install --ignore-installed --no-cache-dir {}
# RUN sed '1d' /thesis-demo/cheapfakes_detection_SCID2024/requirements.txt > /tmp/requirements.txt \
#     && pip install -r /tmp/requirements.txt \
#     && rm /tmp/requirements.txt

RUN pip install -r /thesis-demo/cheapfakes_detection_SCID2024/requirements.txt

# Install pycocotools separately
RUN pip install pycocotools

# # Verify installation
# RUN pip check

# RUN pip install tqdm yacs>=/0.1.8 timm>=0.5.4 numpy==1.21.5
# RUN pip install --upgrade gdown

RUN pip install --upgrade --no-cache-dir gdown
RUN gdown 1JqbqNtwz2-gxeJxmeQF-xzNCasO3EfTQ
RUN mv /thesis-demo/checkpoint.best_snli_score_0.8290.pt /thesis-demo/cheapfakes_detection_SCID2024/checkpoint.best_snli_score_0.8290.pt
RUN gdown 1rmFUkxdZSFJxCx6coCeN9SRTljfpOY2v
RUN mv /thesis-demo/fairseq.zip /thesis-demo/cheapfakes_detection_SCID2024/fairseq.zip
RUN rm -rf /thesis-demo/cheapfakes_detection_SCID2024/fairseq
RUN unzip /thesis-demo/cheapfakes_detection_SCID2024/fairseq.zip -d /thesis-demo/cheapfakes_detection_SCID2024/ && rm /thesis-demo/cheapfakes_detection_SCID2024/fairseq.zip
RUN cd /thesis-demo/cheapfakes_detection_SCID2024/fairseq
RUN pip install /thesis-demo/cheapfakes_detection_SCID2024/fairseq/.
RUN pip install tensorboard
RUN pip install protobuf==3.20.3

# requirements for ROC running
# RUN cd /thesis-demo/hybrid/
# RUN pip install -r /thesis-demo/hybrid/requirements_selenium_updated.txt
RUN pip install streamlit pyperclip selenium google-api-python-client google-auth

# download TruFor weights
# RUN wget -q -c https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip
# RUN cd /thesis-demo/trufor-clone/
RUN gdown 1IWvNt6_bvHbYfXpN53XdxLmTPhg7hds0
RUN mv /thesis-demo/TruFor_weights.zip /thesis-demo/trufor-clone/test_docker/src/TruFor_weights.zip
RUN unzip -q -n /thesis-demo/trufor-clone/test_docker/src/TruFor_weights.zip -d /thesis-demo/trufor-clone/test_docker/src/ && rm /thesis-demo/trufor-clone/test_docker/src/TruFor_weights.zip

# ENTRYPOINT [ "python", "trufor_test.py" ]
# ENTRYPOINT [ "python", "demo_docker.py" ]
# CMD [ "python", "demo_docker.py" ]
