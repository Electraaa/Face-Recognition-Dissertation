import re
import sqlite3

import cv2
import numpy as numpy
import os, time
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner


class SaveFace:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.face_aligner = FaceAligner(self.shape_predictor, desiredFaceWidth=250)
        self.FACE_DIR = "new_faces/"

    @staticmethod
    def __create_folder(folder_name):
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

    def start(self):
        self.__create_folder(self.FACE_DIR)
        # get face id
        face_id = len(os.listdir("faces")) + len(os.listdir("new_faces"))
        face_id = int(face_id)
        face_folder = self.FACE_DIR + str(face_id) + "/"
        self.__create_folder(face_folder)
        name = input("Enter Name")
        connection = sqlite3.connect("face_db.db",)
        query = "INSERT INTO faces VALUES ('"+str(face_id)+"','"+name+"');"
        print(query)
        cursor = connection.execute(query)
        connection.commit()
        cursor.close()
        # exit(9)  # TODO Remove after SQL works
        init_img_no = 1
        img_no = init_img_no
        total_imgs = 300

        # Open Video Capture
        webcam = cv2.VideoCapture(0)
        # Create Windows
        video_capture = "Video Capture"
        aligned = "Alignment"
        cv2.namedWindow(winname=video_capture, flags=cv2.WINDOW_NORMAL)
        cv2.namedWindow(winname=aligned, flags=cv2.WINDOW_NORMAL)
        cv2.resizeWindow(video_capture, 1000, 800)
        cv2.resizeWindow(aligned, 300, 300)
        cv2.moveWindow(video_capture, 0, 0)
        cv2.moveWindow(aligned, 1000, 200)

        # Loop Save
        while True:
            frame = webcam.read()[1]
            frame_with_instructions = frame
            cv2.imshow(video_capture, frame_with_instructions)
            frame_bw = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
            faces = self.detector(frame_bw)
            if len(faces) == 1:
                face = faces[0]
                (x, y, w, h) = face_utils.rect_to_bb(face)
                face_img = frame_bw[y:y + h, x:x + w]
                # align the face
                face_aligned = self.face_aligner.align(frame, frame_bw, face)

                # saving the face
                face_img = face_aligned
                img_path = face_folder + str(img_no) + ".jpg"
                cv2.imwrite(img_path, face_img)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
                cv2.imshow(aligned, face_img)
                img_no += 1
            cv2.imshow(video_capture, frame)
            cv2.waitKey(1)
            if img_no == init_img_no + total_imgs:
                break

        webcam.release()


if __name__ == '__main__':
    SaveFace().start()
