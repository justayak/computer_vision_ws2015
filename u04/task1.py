# Aufgabe 1
# (Fortsetzung der Aufgabe 2 des vorangeganenen Übungsblattes)
# implementiere die Berechnung des Fensterzentrums, -größe und Objektorientierung wie im Paper
# tracke das Auto vom ersten zum letzten Frame
# Abgabe 1.1 : letztes Einzelbild im Video plus Overlay des Suchfensters
import cv_helper

video = cv_helper.load_video_as_rgb("../u03/racecar.avi")

cv_helper.play_video(video)
