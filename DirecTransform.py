import numpy as np

def AngleTransform(Curdir, RotateDir):
    #if Curdir[0] == np.array([0, 1]):
    if Curdir[0] == 0 and Curdir[1] == 1:
        if RotateDir == 'left':
            angle = 90
        elif RotateDir == 'right':
            angle = - 90
        elif RotateDir == 'forward':
            angle = 0
        else:
            angle = 180

    #elif Curdir == np.array([0, -1]):
    elif Curdir[0] == 0 and Curdir[1] == -1:
        if RotateDir == 'left':
            angle = 90
        elif RotateDir == 'right':
            angle = - 90
        elif RotateDir == 'forward':
            angle = 0
        else:
            angle = 180

    #elif Curdir == np.array([-1, 0]):
    elif Curdir[0] == -1 and Curdir[1] == 0:
        if RotateDir == 'left':
            angle = 90
        elif RotateDir == 'right':
            angle = - 90
        elif RotateDir == 'forward':
            angle = 0
        else:
            angle = 180
    #else:
    elif Curdir[0] == 1 and Curdir[1] == 0:
        if RotateDir == 'left':
            angle = 90
        elif RotateDir == 'right':
            angle = - 90
        elif RotateDir == 'forward':
            angle = 0
        else:
            angle = 180
    else:
        print('error')

    return angle