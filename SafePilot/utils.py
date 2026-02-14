import math

def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def calculate_ear(points):
    v1 = distance(points[1], points[5])
    v2 = distance(points[2], points[4])
    h = distance(points[0], points[3])
    if h == 0:
        return 0
    return (v1 + v2) / (2.0 * h)

def calculate_mar(points):
    v = distance(points[2], points[6])
    h = distance(points[0], points[4])
    if h == 0:
        return 0
    return v / h
