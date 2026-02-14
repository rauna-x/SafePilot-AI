import pygame

pygame.mixer.init()

beep_sound = pygame.mixer.Sound("Beep.mp3")
warning_sound = pygame.mixer.Sound("Warning.mp3")

beep_channel = pygame.mixer.Channel(0)
warning_channel = pygame.mixer.Channel(1)


def play_beep():
    if not beep_channel.get_busy():
        beep_channel.play(beep_sound, loops=-1)


def stop_beep():
    beep_channel.stop()


def play_warning():
    if not warning_channel.get_busy():
        warning_channel.play(warning_sound, loops=-1)


def stop_warning():
    warning_channel.stop()
