FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
SHELL ["/bin/bash", "-c"]

# os requirements
RUN apt update
RUN apt upgrade -y
RUN apt install -y apt-utils wget unzip git build-essential gcc python3-dev libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx libnss3 libgconf-2-4 libfontconfig1 xclip chromium-browser && rm -rf /var/lib/apt/lists/*
# RUN wget https://chromedriver.storage.googleapis.com/114.0.5735.90/chromedriver_linux64.zip
# RUN unzip chromedriver_linux64.zip
# RUN mv chromedriver /usr/bin/
# RUN chmod +x /usr/bin/chromedriver


# RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
# # RUN dpkg -i google-chrome-stable_current_amd64.deb || true
# RUN apt -f install -y
# RUN apt install -y ./google-chrome-stable_current_amd64.deb

RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \ 
    && echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list
RUN apt update && apt -y install google-chrome-stable



RUN pip install --upgrade pip

# RUN python -m venv venv
# RUN source venv/bin/activate

RUN pip install tqdm yacs>=0.1.8 timm>=0.5.4 numpy==1.21.5

# add source code of the whole demo
ADD ./ /thesis-demo/
WORKDIR /thesis-demo/

# RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
# # RUN dpkg -i google-chrome-stable_current_amd64.deb || true
# RUN apt -f install -y
# RUN dpkg -i google-chrome-stable_current_amd64.deb
# RUN rm google-chrome-stable_current_amd64.deb

# requirements for cheapfakes running
# RUN git clone https://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@github.com/nbtin/cheapfakes_detection_SCID2024.git
# RUN cd /thesis-demo/cheapfakes_detection_SCID2024
# RUN sed '1d' /thesis-demo/cheapfakes_detection_SCID2024/requirements.txt | xargs -I {} pip install {}
# RUN sed '1d' /thesis-demo/cheapfakes_detection_SCID2024/requirements.txt | xargs -I {} pip install --ignore-installed --no-cache-dir {}
# RUN sed '1d' /thesis-demo/cheapfakes_detection_SCID2024/requirements.txt > /tmp/requirements.txt \
#     && pip install -r /tmp/requirements.txt \
#     && rm /tmp/requirements.txt

# RUN pip install -r /thesis-demo/cheapfakes_detection_SCID2024/requirements.txt
RUN pip install -r /thesis-demo/hybrid/requirements.txt


# Install pycocotools separately (to avoid errors)
RUN pip install pycocotools
RUN pip install --upgrade --no-cache-dir gdown

# download cheapfakes weights and move it to the right place
# RUN gdown 1JqbqNtwz2-gxeJxmeQF-xzNCasO3EfTQ # 8290
# RUN gdown 18IA86NDqR5T8u6zS5RNPSfV7rCmA0pBi # 9090
# RUN mv /thesis-demo/checkpoint.best_snli_score_0.8290.pt /thesis-demo/cheapfakes_detection_SCID2024/checkpoint.best_snli_score_0.8290.pt
# RUN mv /thesis-demo/checkpoint.best_snli_score_0.8290.pt /thesis-demo/hybrid/checkpoint.best_snli_score_0.8290.pt
# RUN mv /thesis-demo/checkpoint.best_snli_score_0.9090.pt /thesis-demo/hybrid/checkpoint.best_snli_score_0.9090.pt

# download fairseq and move it to the right place, then install it
# RUN gdown 1rmFUkxdZSFJxCx6coCeN9SRTljfpOY2v
# RUN mv /thesis-demo/fairseq.zip /thesis-demo/cheapfakes_detection_SCID2024/fairseq.zip
# RUN rm -rf /thesis-demo/cheapfakes_detection_SCID2024/fairseq
# RUN unzip /thesis-demo/cheapfakes_detection_SCID2024/fairseq.zip -d /thesis-demo/cheapfakes_detection_SCID2024/ && rm /thesis-demo/cheapfakes_detection_SCID2024/fairseq.zip
# RUN cd /thesis-demo/cheapfakes_detection_SCID2024/fairseq
# RUN pip install /thesis-demo/cheapfakes_detection_SCID2024/fairseq/.
# RUN pip install tensorboard

RUN gdown 1rmFUkxdZSFJxCx6coCeN9SRTljfpOY2v
RUN mv /thesis-demo/fairseq.zip /thesis-demo/hybrid/fairseq.zip
RUN rm -rf /thesis-demo/hybrid/fairseq
RUN unzip /thesis-demo/hybrid/fairseq.zip -d /thesis-demo/hybrid/ && rm /thesis-demo/hybrid/fairseq.zip
RUN cd /thesis-demo/hybrid/fairseq
RUN pip install /thesis-demo/hybrid/fairseq/.
RUN pip install tensorboard

# requirements for ROC running
# RUN cd /thesis-demo/hybrid/
# RUN pip install -r /thesis-demo/hybrid/requirements_selenium_updated.txt
RUN pip install streamlit pyperclip selenium google-api-python-client google-auth firebase-admin google-cloud-storage webdriver-manager

RUN pip install protobuf==3.20.3

# download TruFor weights and move it to the right place
# RUN wget -q -c https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip
# RUN gdown 1IWvNt6_bvHbYfXpN53XdxLmTPhg7hds0
# RUN mv /thesis-demo/TruFor_weights.zip /thesis-demo/trufor-clone/test_docker/TruFor_weights.zip
# RUN unzip -q -n /thesis-demo/trufor-clone/test_docker/TruFor_weights.zip -d /thesis-demo/trufor-clone/test_docker/ && rm /thesis-demo/trufor-clone/test_docker/TruFor_weights.zip

CMD ["streamlit", "run", "demo_docker.py"]
