import cv2, tables, time, sys
import numpy as np

def get_total_frames(file, path='./'):
    # Open video capture
    cap = cv2.VideoCapture(path + file)
    # Obtain (default) total number of frames
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return total_frames
    
def get_fps(file, path='./'):
    # Open video capture
    cap = cv2.VideoCapture(path + file)
    # Obtain (default) total number of frames
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def video_2_frames(file, path='./', outpath='./', timestep=100, left_offset=0, max_frames=1):
    '''
    Extract frames from a video and save them to an output HDF5 file.
    The `timestep` is the time between frames, and it is in ms.
    The `left_offset` is the offset to add to the begining of the video, in ms.
    The `max_frames` is the maximum number of frames to transfer.
    '''
    # Open the hdf5 file
    filters = tables.Filters(complib='blosc:lz4hc', complevel=9)
    hdf5_file = tables.open_file(outpath, mode='a', filters=filters)
    grp_name = file[:-4] #if file[:-4] not in ['V0830086', 'V0830087'] else 'V0830086_V0830087'
    try:
        storage = hdf5_file.get_node('/' + grp_name)
    except tables.NoSuchNodeError:
        # dtype is uint8 (unsigned integer 0-255)
        img_dtype = tables.UInt8Atom()
        data_shape = (0, 300, 300, 3)
        chunkshape=(1, 300, 300, 3)
        storage = hdf5_file.create_earray('/', grp_name, img_dtype, shape=data_shape, 
                                          chunkshape=chunkshape, expectedrows=max_frames)
    
    # Open video capture
    cap = cv2.VideoCapture(path + file)
    frames_ok, frame_count = 0, 0
    tic = time.time()
    
    while(cap.isOpened()):
        if frame_count % 1000 == 0:
            print('\r    Processed {} out of {} frames, progress: {:0.4}%, elapsed: {} sec'.format( 
                 frame_count, max_frames, 100 * (frame_count+1) / max_frames, round(time.time() - tic)), end='', file=sys.stderr)
        # Set the positions in ms for the next frame
        cap.set(cv2.CAP_PROP_POS_MSEC, left_offset + frame_count * timestep)
        # Read next frame
        ret, frame = cap.read()
        frames_ok += ret
        frame_count += 1
        if ret:
            # Transformations: flip around both axes and resize
            # https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#flip
            frame = cv2.resize(cv2.flip(frame, -1), (300, 300), interpolation=cv2.INTER_CUBIC)
            # Save into HDF5 file
            storage.append(np.expand_dims(frame, 0))
        # If we are done, release and finish
        # Also if we surpassed the end of the video
        if frame_count >= max_frames or not ret:
            cap.release()
            break
    hdf5_file.close()
    
    return frames_ok, frame_count