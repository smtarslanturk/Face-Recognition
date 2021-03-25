#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:30:44 2020

@author: bites
"""


# import all library
import cv2
import numpy as np
import face_recognition
# open cv ile bilgiayarın kamerasına erisim sağladık.
# face_recognize ile resimleri tanıma işlemi yaptık

video_capture = cv2.VideoCapture(0)
# (0) Bilgisayarın var olan kamerasını kullandğımızı söylüyoruz.

# Load a sample picture.
# jonas_image = face_recognition.load_image_file("images/jonas.jpg")
bartosz_image = face_recognition.load_image_file("images/bartosz.jpg")
# ulrich_image = face_recognition.load_image_file("images/ulrich.jpg")
# martha_image = face_recognition.load_image_file("images/martha.jpg")
samet_image = face_recognition.load_image_file("images/samet.jpg")
# furkan_image = face_recognition.load_image_file("images/furkan.jpg")
# dogukan_image = face_recognition.load_image_file("images/dogukan.jpg")
sena_image = face_recognition.load_image_file("images/sena.jpg")

#Örnek resimlerimizin olduğu klasöre gidiyoruz ve örneklerimizi alıyoruz. 

# now let's encode all of images
# jonas_face_encoding = face_recognition.face_encodings(jonas_image)[0]
bartosz_face_encoding = face_recognition.face_encodings(bartosz_image)[0]
# ulrich_face_encoding = face_recognition.face_encodings(ulrich_image)[0]
# martha_face_encoding = face_recognition.face_encodings(martha_image)[0]
samet_face_encoding = face_recognition.face_encodings(samet_image)[0]
# furkan_face_encoding = face_recognition.face_encodings(furkan_image)[0]
# dogukan_face_encoding = face_recognition.face_encodings(dogukan_image)[0]
sena_face_encoding = face_recognition.face_encodings(sena_image)[0]

# Her resim için kodlama yapılıyor ve 128 satırlık dizilere çeviriliyor.

# arrays of known face encodings 
known_face_encodings = [
    # jonas_face_encoding,
    bartosz_face_encoding,
    # ulrich_face_encoding,
    # martha_face_encoding,
    samet_face_encoding,
    # furkan_face_encoding,
    # dogukan_face_encoding,
    sena_face_encoding,
]
#Örnek resimlerimizi atadığımız değişkenleri yeni bir diziye atadık. 

# and their names
known_face_names = [
   # "Jonas Kahnwald",
    "Bartosz Tiedemann",
   # "Ulrich Nielsen",
   # "Martha Nielsen",
    "Samet Arslanturk Hosgeldin",
   # "Furkan Yanteri Welcome Back",
   # "Dogukan Bey Welcome"
   "Ben Sena, Selam!"
]
#Kullandığımız resim isimlerini listeye atadık ve bunlar çıktı olarakkarşımıza gelecek.

# some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    # Videodan anlık bir frame alınıyor.
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    # Kamera her yeri almkatadır ama bizim işimize yarayan yeri almamız yeterli.
    # Bundan dolayı alınan görüntüler resize ile tekrar boyutlandırılır.
    small_frame = cv2.resize(frame, None , fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    #OpenCv üzerinden alınan 
    rgb_small_frame = small_frame[:, :, ::-1]
    # BGR(opencv) türündeki resmi RGB(face_recognition) formatına çeviriyoruz

    
    if process_this_frame:
        # Uyumlu tüm yüzlerin lokasyonlarını bulan kodlar
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        # face_location sayesinde görüntüdeki yüzün konumu tespit edilir.
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        #tanımlanan isimlerin adları burda tutulmaktadır.
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            # return a list of TRUE or FALSE
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
             # Eşleşme bulunduysa
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                
            face_names.append(name)

    process_this_frame = not process_this_frame
    
    
# Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *=4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        #Yüzün etrafını kare ile kaplamak amacı ile rectangle fonksiyonu kullanılır.
        #Burada  (0,0,255) cerceve rengi, 2 ise cerceve kalınlıgını ifade eder. 
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        # Kişi etiketini oluştur
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)
    #imshow ile goruntu ekrana bastırılmaktadır.

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        video_capture.release()
        break