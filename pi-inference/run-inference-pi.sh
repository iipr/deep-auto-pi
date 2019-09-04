#!/bin/bash

sleep 10
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.7
echo -e "\n"PATH=$PATH"\n"
echo -e PYTHONPATH=$PYTHONPATH"\n"
echo -e LD_LIBRARY_PATH=$LD_LIBRARY_PATH"\n"
echo -e OpenCV_DIR=$OpenCV_DIR"\n"
/usr/bin/python3.7 /home/pi/deep-autopi/pi-inference/inference.py
