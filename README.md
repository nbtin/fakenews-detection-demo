# Fake News Detection Demo

Disclaimer: This Readme is generated from Copilot :smile:

This repository contains a demo for detecting fake news using machine learning techniques. The purpose of this README is to provide instructions on how to run the code and understand the project structure.

## Installation

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/nbtin/fakenews-detection-demo.git
    ```

2. Navigate to the project directory:

    ```bash
    cd fakenews-detection-demo
    ```

3. Install the required dependencies:

    ```bash
    conda create --name fakenews python=3.11 -y
    conda activate fakenews
    pip install -r requirements.txt
    ```

4. Build docker image:
    
    ```bash
    cd trufor-clone/test_docker/
    bash docker_build.sh
    cd ../../
    ```

## Usage

Just run

```bash
streamlit run demo.py
```

And enjoys the demo!