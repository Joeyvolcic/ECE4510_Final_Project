import cv2
import cv2.aruco as aruco
import numpy as np
import math

import inverse_kinamatics as ik
from object_loader import *
import generate_obj as go
import aruco as ac
import render as rd



aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board_main = ac.Charuco_Board(squaresX = 6, squaresY = 10, squareLength = 250, markerLength = 175, dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
# board_main.draw_board()
# board_main.save_board("charuco_board.png")

obj_pos1 = ac.Aruco(squareLength = 300, markerLength = 200, dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50), id = 1)
# obj_pos1.draw_aruco()
# obj_pos1.save_aruco("aruco_1.png")

obj_pos2 = ac.Aruco(squareLength = 300, markerLength = 200, dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50), id = 2)
# obj_pos2.draw_aruco()
# obj_pos2.save_aruco("aruco_2.png")



def __main__():
    # Define the size of the board
    squaresX = 6
    squaresY = 10
    squareLength = 250  # The length of the squares in pixels
    markerLength = 75  # The length of the markers in pixels
    DEFAULT_COLOR = ((11, 94, 225))

    # Defines the flags for displaying different features on screen 
    save_points = 0
    show_distances = 0
    show_markers = 0
    show_axes = 1
    show_AR = 1
    sizeT = 0

    #positions are stored as an array of 2 elements, the first being the x coordinate and the second being the y coordinate
    positions = []

    joint_length = 100
    rotation = 0
    rvec = np.array([])
    tvec = np.array([])

    aruco_rvec = np.array([])
    aruco_tvec = np.array([])

    # Define the camera matrix and distortion coefficients
    # These values are usually obtained by calibrating the camera
    camera_matrix = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    dist_coeffs = np.zeros((4,1)) 

    #this will be moda's inverse kinamatics point calculations, the position should be stored as an array of 5 elements with the last one being the angle of rotation
    #[x,y,z,y,z,angle]
    # path = [0,0,0,0,0,0],[0,0,0,0,0,1],[0,0,0,0,0,2],[0,0,0,0,0,3],[0,0,0,0,0,4],..., [0,0,0,0,0,30], [0,0,1,1,0,30],[0,0,1,5,0,30],  
    path = [90,90,180],[90,90,180]
    go.generate_base_obj("C:\\Users\\JoeyV\\OneDrive\\4510\\Projects\\augmented-reality-master (1)\\augmented-reality-master\\models\\base.obj", 0, 0, 100, 0, 0, 200)
    base = OBJ("C:\\Users\\JoeyV\\OneDrive\\4510\\Projects\\augmented-reality-master (1)\\augmented-reality-master\\models\\base.obj",swapyz=True)


    cap = cv2.VideoCapture(1)
    print("connected to camera")

    while True:
        # read the current frame
        # try:
        ret, frame = cap.read()
        
    
        
        # frame = frame[0:480, 80:560] #crop the image to 480x480 to match the input image
        image = frame
        corners, ids, _ = aruco.detectMarkers(frame, aruco_dict)

        # If at least one marker detected
        if len(corners) > 0:
            _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, image, board_main.board)
            
            # If at least one Charuco corner detected
            if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:
                if show_markers == 1:
                    image = aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)

                # Estimate the pose of the Charuco board
                if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) >= 6:
                    _, rvec, tvec = aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board_main.board, camera_matrix, dist_coeffs, rvec, tvec)

                # Gets the current position of the object and displays it over the board
                if rvec is not None and tvec is not None:
                    # This traveses the path array and with itterate every 10th frame, this can be speed up or slowed down with the animation_speed variable
                    animation_speed = 4
                    if path.__len__() != 0:
                        rotation = (rotation + animation_speed )
                        if rotation >= path.__len__():
                            path = [90,90,180],[90,90,180]
                            rotation = 0 
                    else:
                        rotation = 0
                        
                    current_loc = path[math.floor(rotation)]
                    # print(path)

                    # try:
                    try:
                        projection_matrix = rd.compute_projection_matrix(camera_matrix, rvec, tvec)
                        h, w = squaresY * squareLength, squaresX * squareLength

                        #Draws the axes in the center of the board
                        translation_board = np.array([w/2, h/2, 0])
                        rotation_matrix, _ = cv2.Rodrigues(rvec)
                        translation_camera = np.dot(rotation_matrix, translation_board).reshape(3, 1)
                        tvec_center = (tvec + translation_camera)

                        if show_axes == 1:
                            image = cv2.drawFrameAxes(image, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs, rvec=rvec, tvec=tvec_center, length=100)
                        
                        #creates an object for the current position of the arm
                        # print(current_loc)
                        xyzpath = [0, joint_length * math.cos(math.radians(current_loc[0])),
                                    joint_length * math.sin(math.radians(current_loc[0])),
                                    joint_length * math.cos(math.radians(current_loc[0])) + joint_length * math.cos(math.radians(current_loc[0] + current_loc[1])),
                                    joint_length * math.sin(math.radians(current_loc[0])) + joint_length * math.sin(math.radians(current_loc[0] + current_loc[1])),
                                    current_loc[0], current_loc[1]]
                        # print(xyzpath)
                        # print(math.degrees(rvec[2]))
                        go.write_obj_file("C:\\Users\\JoeyV\\OneDrive\\4510\\Projects\\augmented-reality-master (1)\\augmented-reality-master\\models\\arm.obj", xyzpath[0], xyzpath[1], xyzpath[2], xyzpath[3], xyzpath[4]) #really need to use relative paths
                        obj = OBJ("C:\\Users\\JoeyV\\OneDrive\\4510\\Projects\\augmented-reality-master (1)\\augmented-reality-master\\models\\arm.obj",swapyz=False)
                        
                        if show_AR == 1:
                            if sizeT == 0:
                                image = rd.new_render(image, base, projection_matrix, scale_factor = 5, h = h, w = w, theta = 0, color= True) #the base should not rotate, its rendered first so that the rest is rendered on top of it
                                image = rd.new_render(image, obj, projection_matrix, scale_factor = 5, h = h, w = w, theta = np.radians(current_loc[2]), color= True)
                            else:
                                image = rd.new_render(image, base, projection_matrix, scale_factor = 10, h = h, w = w, theta = 0, color= True) #the base should not rotate, its rendered first so that the rest is rendered on top of it
                                image = rd.new_render(image, obj, projection_matrix, scale_factor = 10, h = h, w = w, theta = np.radians(current_loc[2]), color= True)
                                
                    except: 
                        image = frame
                        print("Unable to render object")

            aruco_corners, aruco_ids, _ = aruco.detectMarkers(frame, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50))
            aruco_points = []
            
            if len(aruco_corners) > 0:
                if show_markers == 1:
                    image = cv2.aruco.drawDetectedMarkers(image, aruco_corners, aruco_ids, (123, 51, 45))
                for aruco_corner in aruco_corners:
                    aruco_rvec, aruco_tvec, _ = cv2.aruco.estimatePoseSingleMarkers(aruco_corner, 0.05, camera_matrix, dist_coeffs, aruco_rvec, aruco_tvec)
                    if aruco_rvec is not None and aruco_tvec is not None:
                        if show_axes == 1:
                            image = cv2.drawFrameAxes(image, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs, rvec=aruco_rvec, tvec=aruco_tvec, length=0.01)
                        if rvec is not None and tvec is not None:
                            try:
                                # Finds the center of the charuco board 
                                h, w = squaresY * squareLength, squaresX * squareLength
                                translation_board = np.array([w/2, h/2, 0])
                                rotation_matrix, _ = cv2.Rodrigues(rvec)
                                translation_camera = np.dot(rotation_matrix, translation_board).reshape(3, 1)
                                tvec_center = (tvec + translation_camera)

                                # defines the center points of the charuco and aruco markers
                                charuco_point, _ = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), rvec, tvec_center, camera_matrix, dist_coeffs)
                                
                                # for rvecs in aruco_r:
                                aruco_point, _ = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), aruco_rvec, aruco_tvec, camera_matrix, dist_coeffs)

                                # print(aruco_point.ravel().astype(int))
                                if aruco_points.__len__() > 1:
                                    aruco_points.clear()
                                aruco_points.append(aruco_point.ravel().tolist())

                                #draws lines between the charuco and aruco points
                                if show_distances == 1:
                                    
                                    image = cv2.line(image, tuple(charuco_point.ravel().astype(int)), tuple(aruco_point.ravel().astype(int)), (34, 0, 34), 2)
                            except:
                                pass   

            if save_points == 1:
                positions = []
                print("board rotation", math.degrees(rvec[0]),math.degrees(rvec[1]),math.degrees(rvec[2]))
                try:
                    print("aruco points", aruco_points)
                    
                    print("charuco point", charuco_point.ravel())
                    point= []
                    for i in range(0,2):       
                        positions.append([(aruco_points[i][0] - charuco_point.ravel()[0]), (aruco_points[i][1] - charuco_point.ravel()[1])])
                        print("x",aruco_points[i][0] - charuco_point.ravel()[0] )
                    print("position", positions)   
                    save_points = 0 
                    path = ik.get_path(positions, math.degrees(rvec[2]),aruco_ids)
                    print("arcuo ids", aruco_ids)


                except:
                    print("Unable to save points")
                positions.clear()

                    
                              

        #button press inputs to display different data
        #shows AR if you click t

        if cv2.waitKey(1) & 0xFF == ord('d'):
            sizeT ^= 1
            print("Toggle Size")

        if cv2.waitKey(1) & 0xFF == ord('s'):
            print("Sample Path 1")
            path = ik.get_path3()

        if cv2.waitKey(1) & 0xFF == ord('a'):
            print("Sample Path 1")
            path = ik.get_path2()

        if cv2.waitKey(1) & 0xFF == ord('t'):
            show_AR ^= 1
            print("Toggled AR")
        
        #shows axes if you click r
        if cv2.waitKey(1) & 0xFF == ord('r'):
            show_axes ^= 1
            print("Toggled Axes")

        #shows distances if you click e
        if cv2.waitKey(1) & 0xFF == ord('e'):
            show_distances ^= 1
            print("Toggled Distances")

        #shows markers if you click w
        if cv2.waitKey(1) & 0xFF == ord('w'):
            show_markers ^= 1
            print("Toggled Markers")

        #saves the points if you click space
        if cv2.waitKey(1) & 0xFF == ord(' '):
            save_points = 1
            print("Toggled Points")
            # We can also call our path finding algorithm here
            
        cv2.imshow('frame', image)

    # except:
    #     # if a frame is not captured we can continue without throwing an error
    #     print("Unable to capture video")

    #ends video feed if you click q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

__main__()
