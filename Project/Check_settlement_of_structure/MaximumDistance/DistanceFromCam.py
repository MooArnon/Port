import numpy as np
import cv2
from numpy.lib.type_check import _imag_dispatcher
import cv2.aruco as aruco
import pandas as pd
import glob
camera_matrix = np.array([[9.3038761666e+02, 0.00000000e+00, 5.4552965892e+02,], 
                       [0.00000000e+00, 9.3609270917e+02, 3.8784425324e+02,],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
camera_distotion = np.array([[ 4.375128e-02, -2.7488186e-01, 5.61569e-03,  8.93889e-03, 4.8752157e-1]])

imges = glob.glob('/Users/moomacprom1/Data_science/Code/GitHub/Project/From_image/test.jpg')
imges = sorted(imges)
cap = cv2.VideoCapture(0)
FrameSize = (1080, 720)

firstMarkerID = 0
secondMarkerID = 1
thirdMarkerID = 2
forthMarkerID = 3

marker_size = 80 # ,[mm]
def processing(CameraMatrix, CameraDistotion, from_image=True):
    x = 0
    if from_image == True:
        for image in imges:
            img = cv2.imread(image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            arucodict = aruco.Dictionary_get(aruco.DICT_4X4_100)
            parameters = aruco.DetectorParameters_create()
            corners, ids, rejected_img_point = aruco.detectMarkers(gray, arucodict,
                                                                    parameters=parameters,
                                                                    cameraMatrix=camera_matrix,
                                                                    distCoeff=camera_distotion)
            
            if np.all(ids is not None):
                for i in range(0, len(ids)):
                    rvec, tvec, markerPoint = aruco.estimatePoseSingleMarkers(corners[i], marker_size, 
                                                                            CameraMatrix, CameraDistotion)
                    distant = round(tvec[0][0][2], 2)
                    aruco.drawDetectedMarkers(img, corners)
                    aruco.drawAxis(img, camera_matrix, camera_distotion, rvec, tvec, 20)

                    cv2.putText(img, 'Distance  ,[millimetres]', (100,350), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 250, 0), 2)
                    if ids[i] == firstMarkerID:
                        z = tvec[0][0][2]
                        print(z)
                        cv2.putText(img, '1st marker  '+str(z), (100,400), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 250, 0), 2)

                    elif ids[i] == secondMarkerID:
                        z = tvec[0][0][2]
                        print(z)
                        cv2.putText(img, '2nd marker  '+str(z), (100,425), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 250, 0), 2)

                    elif ids[i] == thirdMarkerID:
                        z = tvec[0][0][2]
                        print(z)
                        cv2.putText(img, '3rd marker  '+str(z), (100,450), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 250, 0), 2)

                    elif ids[i] == forthMarkerID:
                        z = tvec[0][0][2]
                        print(z)
                        cv2.putText(img, '4th marker  '+str(z), (100,475), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 250, 0), 2)

            cv2.imshow("Processing", img)
            cv2.waitKey(0)

            x += 1


    elif from_image == False:
        while cap.isOpened():
            ret, frame =  cap.read()
            frame = cv2.resize(frame, FrameSize)    # Resize camera
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Gray parameter, use for detect ArUco
            arucodict = aruco.Dictionary_get(aruco.DICT_5X5_250)    # Get 5x5 ArUco
            parameter = aruco.DetectorParameters_create()   #Create detect parametres
            (corners, ids, rejected_img_points) = aruco.detectMarkers(gray, arucodict,  #Detect corners and id of marker
                                                                        parameters=parameter,
                                                                        cameraMatrix=CameraMatrix,
                                                                        distCoeff=CameraDistotion)
            
            if np.all(ids is not None): # Detect marker
                for i in range(0, len(ids)): # Detect multi marker
                    rvec, tvec, marker_point = aruco.estimatePoseSingleMarkers(corners[i], marker_size,  # get tvec from marker
                                                                                CameraMatrix, CameraDistotion)
                    distant = round(tvec[0][0][2], 2)
                    aruco.drawDetectedMarkers(frame, corners)
                    aruco.drawAxis(frame, camera_matrix, camera_distotion, rvec, tvec, 20)
                    print('distant: ', distant)
            
            cv2.imshow('Processing', frame)
            key = cv2.waitKey(3) & 0xFF
            if key == ord('q'):  # Exit
                break

processing(camera_matrix, camera_distotion, from_image=True)



