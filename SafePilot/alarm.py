import pygame
import threading

pygame.mixer.init()

alarm_playing = False

def _loop_alarm():
    global alarm_playing
    pygame.mixer.music.load("Beep.mp3")
    pygame.mixer.music.play(-1)
    alarm_playing = True

def trigger_critical():
    global alarm_playing
    if not alarm_playing:
        threading.Thread(target=_loop_alarm, daemon=True).start()

def trigger_warning():
    pygame.mixer.music.load("Beep.mp3")
    pygame.mixer.music.play()

def stop_alarm():
    global alarm_playing
    pygame.mixer.music.stop()
    alarm_playing = False
