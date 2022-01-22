import math
import pandas as pd
import cv2
import cv2.aruco as aruco
import numpy as np

#* Paremetres
# Calibration parameters
## From calibration code, for calibrate camera before using
camera_matrix = np.array([[9.3038761666e+02, 0.00000000e+00, 5.4552965892e+02,], 
                       [0.00000000e+00, 9.3609270917e+02, 3.8784425324e+02,],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
camera_distotion = np.array([[ 4.375128e-02, -2.7488186e-01, 5.61569e-03,  8.93889e-03, 4.8752157e-1]])
FrameSize = (1080, 720) # Resize of frame
# ArUco marker parameters
reffMarkerID = 0
firstMarkerID = 1
secondMarkerID = 2
thirdMarkerID = 3
forthMarkerID = 4
marker_size = 50 # [mm]
# Read video from device number 0th, build in camera.
cap = cv2.VideoCapture(0)
# Data collecting DataFrame
first_dist = []
second_dist = []
third_dist = []
forth_dist = []

yDiff_1 = []
yDiff_2 = []
yDiff_3 = []
yDiff_4 = []

data_dict = {
        'picture NO.' : [],
        '1st' : [],
        '2nd' : [],
        '3rd' : [],
        '4th' : []
}
data = pd.DataFrame(data_dict)

#* Distant between two point 
# In plane x-y
def calculateDistance_xy(x1,y1,x2,y2):  
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)      # Formula
    return dist    # Return result
# In plane x-y-z
def calculatedistance_xyz(x1,y1,z1,x2,y2,z2):
    dist = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    return dist

#* Coordinate distance calculation
def calculatedistance_coordinate(frame, yr,zr,y,z,id):
    if id == 1:
        yDiff_1 = round(yr-y, 2) # y diff coordinate
        zDiff_1 = round(zr-z, 2) # z diff coordinate
        cv2.putText(frame, 'y1 different  '+str(yDiff_1), (50,500), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 150, 0), 2) # y diff coordinate
        cv2.putText(frame, 'z1 different  '+str(zDiff_1), (250,500), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 150, 0), 2) # z diff coordinate
        first_dist.append(yDiff_1)
        
    elif id == 2:
        yDiff_2 = round(yr-y, 2) # y diff coordinate
        zDiff_2 = round(zr-z, 2) # z diff coordinate
        cv2.putText(frame, 'y2 different  '+str(yDiff_2), (50,550), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 150, 0), 2) # y diff coordinate
        cv2.putText(frame, 'z2 different  '+str(zDiff_2), (250,550), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 150, 0), 2) # z diff coordinate 
        second_dist.append(yDiff_2)                   
        
    elif id == 3:
        yDiff_3 = round(yr-y, 2) # y diff coordinate
        zDiff_3 = round(zr-z, 2) # z diff coordinate
        cv2.putText(frame, 'y3 different  '+str(yDiff_3), (50,600), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 150, 0), 2) # y diff coordinate
        cv2.putText(frame, 'z3 different  '+str(zDiff_3), (250,600), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 150, 0), 2) # z diff coordinate
        third_dist.append(yDiff_3)
        
    elif id == 4:
        yDiff_4 = round(yr-y, 2) # y diff coordinate
        zDiff_4 = round(zr-z, 2) # z diff coordinate
        cv2.putText(frame, 'y4 different  '+str(yDiff_4), (50,650), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 150, 0), 2) # y diff coordinate
        cv2.putText(frame, 'z4 different  '+str(zDiff_4), (250,650), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 150, 0), 2) # z diff coordinate
        forth_dist.append(yDiff_4)


#* Processing
def process(CameraMatrix, CameraDistotion):
    while cap.isOpened(): 
        ret, frame =  cap.read()    # read data from device number 0th
        frame = cv2.resize(frame, FrameSize)    # Resize camera
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Gray parameter, use for detect ArUco
        arucodict = aruco.Dictionary_get(aruco.DICT_5X5_250)    # Get 5x5 ArUco
        parameter = aruco.DetectorParameters_create()   #Create detect parametres
        (corners, ids, rejected_img_points) = aruco.detectMarkers(gray, arucodict,  #Detect corners and id of marker
                                                                    parameters=parameter,
                                                                    cameraMatrix=CameraMatrix,
                                                                    distCoeff=CameraDistotion)
        if np.all(ids is not None): # Detect marker
            for i in range(0, len(ids)): # Detect multiple marker
                rvec, tvec, marker_point = aruco.estimatePoseSingleMarkers(corners[i], marker_size,  # get parameter from marker
                                                                            CameraMatrix, CameraDistotion)
                # Global position of each markers, vector location in camera.                                                 
                if ids[i] == reffMarkerID:          # id = 0
                    xr = round(tvec[0][0][0], 3)    # [[[x, y, z]]]
                    yr = round(tvec[0][0][1], 3)    # [[[x, y, z]]]
                    zr = round(tvec[0][0][2], 3)    # [[[x, y, z]]]
                    # Print position of references marker
                    cv2.putText(frame, 'xr  '+str(xr), (50,50),  cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)
                    cv2.putText(frame, 'yr  '+str(yr), (200,50), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)
                    cv2.putText(frame, 'zr  '+str(zr), (350,50), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)
                
                elif ids[i] == firstMarkerID:       # id = 1
                    x1 = round(tvec[0][0][0], 3)    # [[[x, y, z]]]
                    y1 = round(tvec[0][0][1], 3)    # [[[x, y, z]]]
                    z1 = round(tvec[0][0][2], 3)    # [[[x, y, z]]]
                    # Print position of first maker
                    cv2.putText(frame, 'x1  '+str(x1), (50,100), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)
                    cv2.putText(frame, 'y1  '+str(y1), (200,100), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)
                    cv2.putText(frame, 'z1  '+str(z1), (350,100), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)

                elif ids[i] == secondMarkerID:      # id = 2
                    x2 = round(tvec[0][0][0], 3)    # [[[x, y, z]]]
                    y2 = round(tvec[0][0][1], 3)    # [[[x, y, z]]]
                    z2 = round(tvec[0][0][2] ,3)    # [[[x, y, z]]]
                    # Print position of second maker
                    cv2.putText(frame, 'x2  '+str(x2), (50,150), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)                        
                    cv2.putText(frame, 'y2  '+str(y2), (200,150), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)
                    cv2.putText(frame, 'z2  '+str(z2), (350,150), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)

                elif ids[i] == thirdMarkerID:           # id = 3
                    x3 = round(tvec[0][0][0], 3)        # [[[x, y, z]]]
                    y3 = round(tvec[0][0][1], 3)        # [[[x, y, z]]]
                    z3 = round(tvec[0][0][2] ,3)        # [[[x, y, z]]]
                    # Print position of third maker
                    cv2.putText(frame, 'x3  '+str(x3), (50,200), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)                        
                    cv2.putText(frame, 'y3  '+str(y3), (200,200), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)
                    cv2.putText(frame, 'z3  '+str(z3), (350,200), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)

                elif ids[i] == forthMarkerID:       # id = 4
                    x4 = round(tvec[0][0][0], 3)    # [[[x, y, z]]]
                    y4 = round(tvec[0][0][1], 3)    # [[[x, y, z]]]
                    z4 = round(tvec[0][0][2] ,3)    # [[[x, y, z]]]
                    # Print position of forth maker
                    cv2.putText(frame, 'x4  '+str(x4), (50,250), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)                        
                    cv2.putText(frame, 'y4  '+str(y4), (200,250), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)
                    cv2.putText(frame, 'z4  '+str(z4), (350,250), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)
                
                aruco.drawDetectedMarkers(frame, corners)
            if len(ids) == 5 : 
                # Calculate coordinate distance
                calculatedistance_coordinate(frame, yr, zr, y1, z1, 1)
                calculatedistance_coordinate(frame, yr, zr, y2, z2, 2)
                calculatedistance_coordinate(frame, yr, zr, y3, z3, 3)
                calculatedistance_coordinate(frame, yr, zr, y4, z4, 4)


            

            #######* Distance calculation process *########
            #* Coordinate diff
            # Show Topic
            cv2.putText(frame, 'Coordinate Displacement', (50,450), cv2.FONT_HERSHEY_SIMPLEX,0.5,  (0, 150, 0), 2)

            """
            Useable Idea
            if cal == 'xy':
                dist = round(calculateDistance_xy(xr, yr, x1, y1), 2) # , [millimetres]
                cv2.putText(frame, 'Distant, in x-y plane:  '+str(dist), (800,200), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
                aruco.drawDetectedMarkers(frame, corners)
            elif cal == 'xyz':
                dist = round(calculatedistance_xyz(xr, yr, zr, x1, y1, z1), 2) # , [millimetres]
                cv2.putText(frame, 'Distant, 3D:  '+str(dist), (800,200), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
                aruco.drawDetectedMarkers(frame, corners)
            """

        

        cv2.imshow('Processing', frame)

        key = cv2.waitKey(3) & 0xFF
        if key == ord('q'):  # Exit
            break
    #    elif key == ord('c'):
    
    cap.release()
    cv2.destroyAllWindows()


process(camera_matrix, camera_distotion)



print('################ The running is successful ##################')
#? Can we use for loop in distant calculation?
#TODO Can we detect reference marker first to prevent the error "local variable 'yr' referenced before assignment".
