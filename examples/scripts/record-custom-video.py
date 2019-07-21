import cv2, sys
cap = cv2.VideoCapture(0) # Capture video from camera

# Get the width and height of frame, and the fps
width = int(sys.argv[1])
height = int(sys.argv[2])
fps = int(sys.argv[3])

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
out = cv2.VideoWriter('./data/output.mp4', fourcc, fps, (width, height))

# Set custom values
r1 = cap.set(3, width)
r2 = cap.set(4, height)
r3 = cap.set(5, fps)
print('3 ->', r1, ',', cap.get(propId=3))
print('4 ->', r2, ',', cap.get(propId=4))
print('5 ->', r3, ',', cap.get(propId=5))

while not cap.isOpened():
    cap.open()
    
g1 = cap.get(3)
g2 = cap.get(4)
g3 = cap.get(5)
print('Recording using {} x {} and fps {}!'.format(g1, g2, g3))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
            break
    else:
        break

# Release everything if job is finished
out.release()
cap.release()
cv2.destroyAllWindows()
