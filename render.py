import cv2
import cv2.aruco as aruco
import numpy as np
import math


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