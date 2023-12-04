import cv2
import cv2.aruco as aruco
import numpy as np

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