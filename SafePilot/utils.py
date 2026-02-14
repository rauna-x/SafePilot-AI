import numpy as np

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_ear(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def calculate_mar(mouth):
    A = euclidean(mouth[2], mouth[10])
    B = euclidean(mouth[4], mouth[8])
    C = euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)
