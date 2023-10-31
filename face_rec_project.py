import face_recognition
import os, sys
import cv2
import numpy as np
import math

def face_confidence(face_distance, face_match_threshold=1.0):
    
    if face_distance <= face_match_threshold:
        confidence = 1.0 - (face_distance / face_match_threshold)
        confidence = min(confidence, 1.0)  # This makes sure the confidence will not reach above 100%
        return f"{confidence * 100:.2f}%"
    else:
        return "0.00%"
    

class FaceRecognition:
    
    face_locations = []
    face_encodings = []
    face_name = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True
    
    def __init__(self):
        self.encode_faces()
    
    def encode_faces(self):
        for image in os.listdir('face'):
            
            root_name, png = os.path.splitext(image)
            
            face_image = face_recognition.load_image_file(f'face/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]
            
            self.known_face_encodings.append(face_encoding)
            root_name_without_extension = root_name
            self.known_face_names.append(root_name_without_extension)
        

    def run_recognition(self):
        
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            sys.exit('Video source not found')
            
        while True:
            ret, frame = video_capture.read()
            
            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)
                

              
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
            
        
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
                
                self.face_name = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = 'Unknown'
                    confinence = 'Unknown'
                    
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confinence = face_confidence(face_distances[best_match_index])
                        
                    self.face_name.append(f'{name} ({confinence})')
                    
            self.process_current_frame = not self.process_current_frame
            

            for (top, right, bottom, left), name in zip(self.face_locations, self.face_name):
                
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)

            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) == ord('q'):
                break
            
        video_capture.release()
        cv2.destroyAllWindows()
            



if __name__ == '__main__':
    facerec = FaceRecognition()
    facerec.run_recognition()