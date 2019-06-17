import cv2
import numpy as np
import dlib
from imutils import face_utils
from keras.models import load_model
import tensorflow as tf


class Recognise:
    def __init__(self):
        # Set TensorFlow Logging Level to Fatal
        tf.logging.set_verbosity(tf.logging.FATAL)
        # Data
        self.detector = dlib.get_frontal_face_detector()
        self.FACENET_MODEL = "dlib_face_recognition_resnet_model_v1.dat"
        self.SHAPE_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
        self.face_rec = dlib.face_recognition_model_v1(self.FACENET_MODEL)
        self.shape_predictor = dlib.shape_predictor(self.SHAPE_PREDICTOR)
        self.face_recognizer = load_model('mlp_model_keras2.h5')

    def __recognize_face(self, face_descriptor):
        prediction = self.face_recognizer.predict(face_descriptor)
        prediction_probability = prediction[0]
        prediction_class = list(prediction_probability).index(max(prediction_probability))
        return max(prediction_probability), prediction_class

    def __get_face_names(self):
        face_ids = dict()
        import sqlite3
        conn = sqlite3.connect("face_db.db")
        sql_cmd = "SELECT * FROM faces"
        cursor = conn.execute(sql_cmd)
        for row in cursor:
            face_ids[row[0]] = row[1]
        return face_ids

    def __extract_face_info(self, img, img_rgb, face_names):
        faces = self.detector(img_rgb)
        x, y, w, h = 0, 0, 0, 0
        face_descriptor = None
        if len(faces) > 0:
            for face in faces:
                shape = self.shape_predictor(img, face)
                (x, y, w, h) = face_utils.rect_to_bb(face)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
                face_descriptor = self.face_rec.compute_face_descriptor(img_rgb, shape)
                face_descriptor = np.array([face_descriptor, ])
                probability, face_id = self.__recognize_face(face_descriptor)
                try:
                    if probability > 0.9:
                        cv2.putText(img, "FaceID #" + str(face_id), (x, y - 70), cv2.FONT_HERSHEY_PLAIN, 1.5,
                                    (0, 255, 0),
                                    2)
                        cv2.putText(img, 'Name - ' + face_names[face_id], (x, y - 40), cv2.FONT_HERSHEY_PLAIN, 1.5,
                                    (0, 255, 0),
                                    2)
                        cv2.putText(img, "%s %.2f%%" % ('Probability', probability * 100), (x, y - 10),
                                    cv2.FONT_HERSHEY_PLAIN,
                                    1.5, (0, 255, 0), 2)
                    else:
                        cv2.putText(img, 'No matching faces', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
                except:
                    cv2.putText(img, 'No matching faces', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

    def start(self):
        # Load Faces
        self.face_recognizer.predict(np.random.rand(1, 128))
        face_names = self.__get_face_names()
        # Open Video Capture
        webcam = cv2.VideoCapture(0)
        # Create Window
        window = "Video Capture - Press Q to Quit"
        cv2.namedWindow(winname=window, flags=cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(window, 0, 0)
        # Loop Until Closed
        while True:
            frame = webcam.read()[1]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.__extract_face_info(frame, frame_rgb, face_names)

            # Show Frame
            cv2.imshow(window, frame)

            # On Exit Close
            if cv2.waitKey(1) == ord('q'):
                break

        webcam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    Recognise().start()
