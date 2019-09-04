from armv7l.openvino.inference_engine import IEPlugin, IENetwork
import cv2, sys, subprocess, os, time
import numpy as np


MODEL_PATH='/home/pi/deep-autopi/data/models/openvino/vehicle_detection/'
MODEL_NAME='vehicle-detection-adas-0002'
OUT_PATH='/home/pi/deep-autopi/data/onroad/'
PRED_FILE='car_predictions.npy'
# Maximum time of inference
MAX_TIME=1

def load_model():
    '''Load and prepare models'''
    #  Plugin initialization for NCS
    plugin = IEPlugin(device="MYRIAD")
    #  Read in Graph file (IR)
    net = IENetwork(model=MODEL_PATH + MODEL_NAME +'.xml',
                    weights=MODEL_PATH + MODEL_NAME +'.bin')

    input_shape = net.inputs['data'].shape
    #  Load network to the plugin
    exec_net = plugin.load(network=net)
    del net
    return exec_net, input_shape

# Load model
exec_net, input_shape = load_model()
n, c, h, w = input_shape
# Create folder for outputs
folder = time.strftime("%Y_%m_%d-%H_%M_%S", time.gmtime())
out_path = OUT_PATH + folder + '/'
pred_path = out_path + PRED_FILE
if not os.path.exists(out_path):
    os.mkdir(out_path)
# Prepare camera
cap = cv2.VideoCapture(0)
count = 0
# Announce its ready
ready = ["espeak", "'The deep pi is ready'"]
process = subprocess.run(ready, check=True,
                         stderr=subprocess.DEVNULL)

# Go!
go = time.time()
while(True):
    # Capture frame-by-frame, infer and save
    tic = time.time()
    ret, frame = cap.read()
    if not ret: continue
    filename = out_path + 'frame{:04}-{}.jpg'.\
                          format(count, time.strftime("%H_%M_%S", time.gmtime(tic)))
    cv2.imwrite(filename, frame)
    # Preprocess frame
    frame = cv2.resize(frame, (w, h))
    frame = frame.transpose((2, 0, 1))
    # Change data layout from HWC to CHW
    frame = frame.reshape((n, c, h, w))
    # Start synchronous inference and get inference result
    req_handle = exec_net.start_async(0, inputs={'data': frame})
    # Get Inference Result
    status = req_handle.wait()
    res = req_handle.outputs['detection_out']
    # Define separate float64 array to include time
    arr = np.empty((10,6), dtype='float64')
    arr[:, 0] = tic
    arr[:, 1:] = res[0, 0, 0:10, 2:]
    # Append to npy file
    with open(pred_path, 'ab') as pred_file:
        np.save(pred_file, arr)
    count += 1
    if time.time() - go > MAX_TIME * 60:
        chkp = ["espeak", "'One minute recorded'"]
        process = subprocess.run(chkp, check=True,
                                 stderr=subprocess.DEVNULL)
        break
else:
    # When done, release the capture
    cap.release()
