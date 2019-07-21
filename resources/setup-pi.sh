#!/bin/bash

#############
# Libraries #
#############

# Dependencies for tensorflow
sudo apt-get install libatlas-base-dev
# and to read tf models
sudo apt install libhdf5-100

# Dependencies to install OpenCV for Python
# Before pip install opencv-python[==3.4.4.19] --no-cache-dir
# https://qiita.com/atuyosi/items/5f73baa08c3408f248e8
#sudo apt install libhdf5-100
#sudo apt install libharfbuzz0b
#sudo apt install libwebp6
sudo apt install libjasper1
sudo apt install libilmbase12
sudo apt install libopenexr22
sudo apt install libgstreamer1.0-0
sudo apt install libavcodec-extra57
sudo apt install libavformat57
sudo apt install libswscale4
#sudo apt install libgtk-3
#sudo apt install libgtk-3-0
sudo apt install libqtgui4
sudo apt install libqt4-test

# It is probably needed to execute this everytime we want to use the camera
# https://community.autopi.io/t/configure-usb-power-for-a-webcam/1372
sudo su
echo 1 > /sys/bus/usb/devices/1-1.1/bConfigurationValue
exit


################
# Python stuff #
################

# Create a new venv with Python 3.5
mkdir ~/python-stuff
python3 -m venv ~/python-stuff/my-py-3.5
source ~/python-stuff/my-py-3.5/bin/activate
cd python-stuff
wget https://bootstrap.pypa.io/get-pip.py
python3 ./get-pip.py
# Install requirements
pip install -r requirements-pi.txt --no-cache-dir

echo "Setup for the Pi finished!"