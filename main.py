import time
import numpy as np
import face_recognition
import os
import datetime
import tkinter as tk
from tkinter import messagebox
from sklearn import svm

known_encodings = []
known_names = []

def tick():
        time_string = time.strftime('%H:%M:%S')
        clock.config(text='The time is '+ time_string)
        clock.after(1000,tick)

#############################--------- Code for marking attendance ------------###############################

def markAttendance(name):
    with open('Sheet.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.datetime.now()
            dtString = now.strftime(': %d-%m-%Y %H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            print("Access Granted!")


#############################---------Code for training SVM model------------###############################

import pickle

model_file = 'face_recognition_model.pkl'   

if os.path.exists(model_file):
    # Load the trained model from the file
    with open(model_file, 'rb') as f: 
        clf = pickle.load(f)
        
else:

    # If no model found - We should train the model.
    # Code for loading known encodings and names from dataset directory

    for person_dir in os.listdir('dataset'):
        person_path = os.path.join('dataset', person_dir)
        if os.path.isdir(person_path):
            for img_file in os.listdir(person_path):
                img_path = os.path.join(person_path, img_file)
                image = face_recognition.load_image_file(img_path)
                encoding = face_recognition.face_encodings(image)
                if len(encoding) > 0:
                    known_encodings.append(encoding[0])
                    known_names.append(person_dir)

    # Code for training SVM model using known encodings and names

    clf = svm.SVC(gamma='scale')
    clf.fit(known_encodings, known_names)

    with open(model_file, 'wb') as f:
        pickle.dump(clf, f)                   # These two lines are to save the trained model to our directory


#############################--------- Code for Capturing the frames by webcam ------------###############################


import cv2
video_capture = cv2.VideoCapture(0)


def Start_Capture():
    while True:
        ret, frame = video_capture.read()

        # Find faces in frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # Prepare face encodings for prediction
        if face_encodings:
            # Convert face encodings to a 2D array
            face_encodings_2d = [face_encoding.reshape(1, -1) for face_encoding in face_encodings]
            
            # Concatenate face encodings vertically to form a single 2D array
            face_encodings_combined = np.vstack(face_encodings_2d)
            
            # Predict using trained SVM model
            names = clf.predict(face_encodings_combined)

            # Loop through recognized names
            for name, (top, right, bottom, left) in zip(names, face_locations):
                # Draw rectangle around face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Write name of recognized person
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                markAttendance(name)

        # Display result
        cv2.imshow('Face-Capture Module', frame)

        # Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcam and close terminal
    video_capture.release()
    cv2.destroyAllWindows()



##################################---------- GUI ----------################################

def quit_app():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        terminal.destroy()


# Create Tkinter terminal
terminal = tk.Tk()
terminal.title("Face Attendance System")
terminal.geometry("600x400")
terminal.configure(background='black')

welcome_label = tk.Label(terminal, text="Welcome to the Face Attendance System!", fg="white" , bg = "#232529" , font=("comic", 18 , "bold"))
welcome_label.pack(pady=20)

clock = tk.Label(terminal ,fg="#f7a34f", bg = "black" ,width=55 ,height=1,font=('comic', 22, ' bold '))
clock.pack(pady=10)
tick()

start_button = tk.Button(terminal, text="Start", command=Start_Capture, fg="black"  ,bg="spring green", font=("comic", 14 , "bold"))
start_button.pack(pady=10)

quit_button = tk.Button(terminal, text="Quit", command=quit_app , fg="black" , bg="#e33232",  font=("comic", 14 , "bold"))
quit_button.pack(pady=10)

terminal.mainloop()

#####################################--------------- END -----------------------#################################