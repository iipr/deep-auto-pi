#!/bin/bash

source /opt/intel/openvino/bin/setupvars.sh -pyver 3.7
echo -e "\n"PATH=$PATH"\n"
echo -e PYTHONPATH=$PYTHONPATH"\n"
echo -e LD_LIBRARY_PATH=$LD_LIBRARY_PATH"\n"
echo -e OpenCV_DIR=$OpenCV_DIR"\n"
python3.7 inference.py
#espeak "Loading inference engine" 2> /dev/null
