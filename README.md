Use PYTHON 2.7, UBUNTU 16.04

### download opencv-3.2 and opencv_contrib-3.2:

    mkdir ~/Downloads
    cd ~/Downloads
    wget -O opencv.zip https://github.com/opencv/opencv/archive/3.2.0.zip
    unzip opencv.zip
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.2.0.zip
    unzip opencv_contrib.zip
### Install dependencies for opencv

    sudo apt-get update
    sudo apt-get install build-essential cmake pkg-config
    sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
    sudo apt-get install libgtk-3-dev
    sudo apt-get install libatlas-base-dev gfortran
    sudo apt-get install python2.7-dev python3.5-dev

### Build and install opencv

    cd ~/Downloads/opencv-3.2.0/
    mkdir build
    cd build
    pip install numpy
    cmake -D CMAKE_BUILD_TYPE=RELEASE -D OPENCV_EXTRA_MODULES_PATH=~/Downloads/opencv_contrib-3.2.0/modules/ -D WITH_CUDA=OFF -D ENABLE_PRECOMPILED_HEADERS=OFF ..
(Verify features to build: text, features_2d)

(If build for python3, if build for java ?)

    make -j4
    sudo make install
    sudo ldconfig
    ln lib/cv2.so <PYTHON2 SITE PACKAGE>/cv2.so
(Verify cv2: import cv2)


### Install Ocropus library:

    cd ~/Downloads
    wget -O ocropus.zip https://github.com/tmbdev/ocropy/archive/v1.3.3.zip
    unzip ocropus.zip
    cd ocropy-1.3.3/
    pip install -r requirements.txt
    python setup.py install


### Install App's dependencies and and configuration:

    mkdir ~/workspace
    cd ~/workspace
    git clone https://github.com/tgialoimtr/cnn_lstm_ctc_ocr.git ocr-app
    cd ocr-app/src
    cp ../resources/common.py ./


### Edit configuration file src/common.py:

1. args.imgsdir to folder containing receipt images 
2. args.model_path to folder containing neural network model 
3. args.java_path to ./resources
4. args.mode to "process-local"
5. args.logsdir and args.download_dir to some temporary folder

### Run App:

    cd ~/workspace/ocr-app/src
    python ocr-app.py

Check ./result.csv for result
