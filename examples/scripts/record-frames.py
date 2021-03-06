import numpy as np
import cv2

cap = cv2.VideoCapture(0)
count = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    filename = './data/frame{}.jpg'.format(count)
    cv2.imwrite(filename, frame)
    count += 1
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
