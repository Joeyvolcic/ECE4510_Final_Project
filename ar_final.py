import cv2
import cv2.aruco as aruco
import numpy as np

from objloader_simple import *
import generate_obj as go
import math

class Charuco_Board:

    def __init__(self, squaresX: int, squaresY: int, squareLength: int, markerLength:int, dictionary: dict): #might be worth adding a name parameter
        """
        Generates a new Charuco board with the given parameters.
        """
        self.board = cv2.aruco.CharucoBoard((squaresX, squaresY), squareLength, markerLength, dictionary)
        self.imboard = self.board.generateImage((squaresX * squareLength, squaresY * squareLength))
        # self.imboard = cv2.copyMakeBorder(self.imboard, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    def draw_board(self):
        """
        Displays the Charuco board image.
        """
        cv2.imshow('charuco board', self.imboard)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_board(self, name: str):
        """
        Saves the Charuco board as a png image.
        """
        cv2.imwrite(name, self.imboard)

class Aruco:

    def __init__(self, squareLength: int, markerLength: int, dictionary: dict, id: int):
        """
        Generates a new Aruco object with the given parameters.
        """
        self.aruco = cv2.aruco.generateImageMarker(dictionary = dictionary, id = id, sidePixels = squareLength, borderBits = 1)

    def draw_aruco(self):
        """
        Displays the Aruco as an image.
        """
        cv2.imshow('aruco tile', self.aruco)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_aruco(self, name: str):
        """
        Saves the Aruco tile as a png image.
        """
        cv2.imwrite(name, self.aruco)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# parameters =  cv2.aruco.DetectorParameters()

board_main = Charuco_Board(squaresX = 8, squaresY = 5, squareLength = 250, markerLength = 175, dictionary = aruco_dict)
# board_main.draw_board()
# board_main.save_board("charuco_board.png")

obj_pos1 = Aruco(squareLength = 400, markerLength = 300, dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50), id = 1)
# obj_pos1.draw_aruco()
# obj_pos1.save_aruco("aruco_1.png")

obj_pos2 = Aruco(squareLength = 400, markerLength = 300, dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50), id = 2)
# obj_pos2.draw_aruco()
# obj_pos2.save_aruco("aruco_2.png")

def compute_projection_matrix(camera_matrix, rvec, tvec):

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    # Create the extrinsic matrix [R | t]
    extrinsic_matrix = np.hstack((R, tvec.reshape((3, 1))))
    # Compute the projection matrix P = K * [R | t]
    projection_matrix = np.dot(camera_matrix, extrinsic_matrix)

    return projection_matrix

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

def new_render(img, obj, projection_matrix, scale_factor, h, w, theta, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * scale_factor
    flip_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    for face in obj.faces:
        face_vertices = face[0]
        face_color = face[1]

        #Gets the points that make up a face
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])

        #Scales the object, points = points * scale_matrix
        #Apply a 180 degree rotation around the x axis, points = points * flip_matrix
        points = np.dot(points, flip_matrix)
        #Rotate the object, points = points * rotation_matrix
        points = np.dot(points, rotation_matrix)
        points = np.dot(points, scale_matrix)

        #Model points are dispalaced from the origin, points = points [x + dispalacement, y + dispalacement, z]
        #This will render the object in the middle of the reference surface
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])

        #find the location of the object in the image, points_image = points * projection_matrix
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection_matrix)
        imgpts = np.int32(dst)

        if color is False:
            cv2.fillConvexPoly(img, imgpts, (11, 94, 225))
        else:
            color = hex_to_rgb(face_color)
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def __main__():
    # Define the size of the board
    squaresX = 8
    squaresY = 5
    squareLength = 250  # The length of the squares in pixels
    markerLength = 75  # The length of the markers in pixels
    DEFAULT_COLOR = ((11, 94, 225))

    # Defines the flags for displaying different features on screen 
    save_points = 0
    show_distances = 0
    show_markers = 0
    show_axes = 1
    show_AR = 1

    #positions are stored as an array of 2 elements, the first being the x coordinate and the second being the y coordinate
    positions = []

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
    path = [0, 86, 50, 136, 120, 0], [0, 86, 50, 136, 120, 3], [0, 86, 50, 136, 120, 6], [0, 86, 50, 136, 120, 9], [0, 86, 50, 136, 120, 12], [0, 86, 50, 136, 120, 15], [0, 86, 50, 136, 120, 18], [0, 86, 50, 136, 120, 21], [0, 86, 50, 136, 120, 24], [0, 86, 50, 136, 120, 27], [0, 86, 50, 136, 120, 30], [0, 86, 50, 136, 120, 25], [0, 86, 50, 136, 120, 20], [0, 86, 50, 136, 120, 15], [0, 86, 50, 136, 120, 10]

    cap = cv2.VideoCapture(1)

    while True:
        # read the current frame
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            return
        
        frame = frame[0:480, 80:560] #crop the image to 480x480 to match the input image
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
                    animation_speed = 0.1
                    rotation = (rotation + animation_speed ) % path.__len__()
                    current_loc = path[math.floor(rotation)]

                    try:
                        projection_matrix = compute_projection_matrix(camera_matrix, rvec, tvec)
                        h, w = squaresY * squareLength, squaresX * squareLength

                        #Draws the axes in the center of the board
                        translation_board = np.array([w/2, h/2, 0])
                        rotation_matrix, _ = cv2.Rodrigues(rvec)
                        translation_camera = np.dot(rotation_matrix, translation_board).reshape(3, 1)
                        tvec_center = (tvec + translation_camera)

                        if show_axes == 1:
                            image = cv2.drawFrameAxes(image, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs, rvec=rvec, tvec=tvec_center, length=100)
                        
                        #creates an object for the current position of the arm
                        go.write_obj_file("C:\\Users\\JoeyV\\OneDrive\\4510\\Projects\\augmented-reality-master (1)\\augmented-reality-master\\models\\arm.obj", current_loc[0], current_loc[1], current_loc[2], current_loc[3], current_loc[4]) #really need to use relative paths
                        obj = OBJ("C:\\Users\\JoeyV\\OneDrive\\4510\\Projects\\augmented-reality-master (1)\\augmented-reality-master\\models\\arm.obj",swapyz=True)
                        
                        if show_AR == 1:
                            image = new_render(image, obj, projection_matrix, scale_factor = 10, h = h, w = w, theta = np.radians(current_loc[5]), color= True)
                    
                    except: 
                        image = frame

            aruco_corners, aruco_ids, _ = aruco.detectMarkers(frame, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50))
            
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
                                h, w = squaresY * squareLength, squaresX * squareLength
                                translation_board = np.array([w/2, h/2, 0])
                                rotation_matrix, _ = cv2.Rodrigues(rvec)
                                translation_camera = np.dot(rotation_matrix, translation_board).reshape(3, 1)
                                tvec_center = (tvec + translation_camera)

                                charuco_point, _ = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), rvec, tvec_center, camera_matrix, dist_coeffs)
                                aruco_point, _ = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), aruco_rvec, aruco_tvec, camera_matrix, dist_coeffs)

                                if show_distances == 1:
                                    image = cv2.line(image, tuple(charuco_point.ravel().astype(int)), tuple(aruco_point.ravel().astype(int)), (34, 0, 34), 2)

                                # Convert the points to integer format
                                charuco_point = tuple(charuco_point.ravel().astype(int))
                                aruco_point = tuple(aruco_point.ravel().astype(int))
                                if save_points > 0 and save_points <= 2:
                                    if positions.__len__() >= 2:
                                        positions = []

                                    pointx = aruco_point[0] - charuco_point[0]
                                    pointy = aruco_point[1] - charuco_point[1]
                                    positions.append([pointx, pointy])
                                    print("coordinatesx: ", pointx)
                                    print("coordinatesy: ", pointy)    
                                    save_points = save_points + 1
                                elif save_points > 2:
                                    save_points = 0



                            except:
                                pass     

        if cv2.waitKey(1) & 0xFF == ord('t'):
            show_AR ^= 1
            print("Toggled AR")
        
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

        cv2.imshow('frame', image)
        #ends video feed if you click q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

__main__()

# if position[0][0] < 0 and position[0][1] < 0:
#     math.tan(position[0][0]/position[0][1]) + 180
# else 
#     math.tan(position[0][0]/position[0][1])
