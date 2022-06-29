import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos
import sys

a = plt.imread("Dali.png")
a1 = plt.imread("VENUS.png")
a2 = plt.imread("cath.png")


## JPEG
def C(x):
    return 1 / np.sqrt(2) if x == 0 else 1

## DCT 8x8

def DCT8x8sum(i, j, im):
    return np.sum([im[x, y] * cos((2*x + 1) * i * pi / 16) * cos((2 * y + 1)*j * pi/ 16) for x in range(8) for y in range(8)])

def DCT8x8(im):
    return np.array([[2 / 8 * C(i) * C(j) * DCT8x8sum(i, j, im) for j in range(8)] for i in range(8)])

## Quantification 8x8

def quantificatiox8x8(F, Q):
    return np.floor((F + np.floor(Q / 2)) / Q)

## Parcours en diagonale 

def iter_util(x, y, bool):
    x, y = x + 1, y - 1
    if x == 8: # bordures supérieures (axe x ou y = taille image
        return 7, y + 2, not bool
    elif y == -1: # bordures inférieures (axe x ou y = 0) 
        return x, 0, not bool
    else: # avancée dans la diagonale
        return x, y, bool

def iterateur(x, y, sens):
    if sens: # diagonale vers le haut
        return iter_util(x, y, True)
    else: # diagonale vers le bas
        y, x, sens = iter_util(y, x, False) # par symétrie de l'algorithme
        return x, y, sens

def parcours_diagonal(M):
    x, y, sens = 0, 0, False # True vers le haut et False vers le bas
    li = []
    derniernon0 = 0
    while (x, y) != (7, 7):
        li.append(M[x][y])
        if M[x][y] != 0:
            derniernon0 = len(li)
        x, y , sens = iterateur(x, y ,sens)
    return li[:derniernon0]
    
## Compression 8x8

def complete(bloc):
    result = np.zeros((8 ,8))
    result[:bloc.shape[0],:bloc.shape[1]] = bloc
    return  result

def compression8x8(bloc, Q):
    return parcours_diagonal(quantificatiox8x8(DCT8x8(complete(bloc)), Q))


## Compression Image

def compression(im, Q):
    return [[[compression8x8(im[x:x+8,y:y+8, color]*255, Q) for color in range(len(im[0, 0]))] for y in range(0, len(im[0]), 8)] for x in range(0, len(im), 8)]


## Parcours en diagonale inverse 

def parcours_diagonal_inverse(coefs):
    result = np.zeros((8 ,8))
    x, y, sens = 0, 0, False
    for c in coefs:
        result[x, y] = c
        x, y, sens = iterateur(x, y, sens)
    return result

## Decantification

def decantification8x8(F, Q):
    return F * Q

## DCT -1

def NDCT8x8sum(x, y, DCT):
    return np.sum([C(i) * C(j) * DCT[j, i] * cos((2*x + 1) * i * pi / 16) * cos((2 * y + 1)*j * pi/ 16) for i in range(8) for j in range(8)])

def NDCT8x8(DCT):
    return np.array([[2 / 8 * NDCT8x8sum(x, y, DCT) for x in range(8)] for y in range(8)])

## decompression 8x8

def decompression8x8(bloc, Q):
    return NDCT8x8(decantification8x8(parcours_diagonal_inverse(bloc), Q))


## decompression

def decompression(im, Q):
    decompressed_blocs = np.array([[[decompression8x8(im[y][x][color], Q) for color in range(len(im[0][0]))] for x in range(len(im[0]))] for y in range(len(im))])

    image = np.array([[[decompressed_blocs[y, x, c, i, j] for c in range(decompressed_blocs.shape[2])] for x in range(decompressed_blocs.shape[1]) for j in range(8)] for y in range(decompressed_blocs.shape[0]) for i in range(8)])
    """ equivalent
    image = np.zeros((decompressed_blocs.shape[0]*8, decompressed_blocs.shape[1]*8, decompressed_blocs.shape[2]))
    for y in range(decompressed_blocs.shape[0]):
        for x in range(decompressed_blocs.shape[1]):
            for c in range(decompressed_blocs.shape[2]):
                for i in range(8):
                    for j in range(8):
                        image[y*8+i, x*8+j, c] = decompressed_blocs[y, x, c, i, j]
    """
    return image / 255

## TESTS (matrice de test)

test = np.array([
    [139, 144, 149, 153, 155 ,155, 155, 155], 
    [144, 151, 153, 156, 159, 156, 156, 156],
    [150, 155, 160, 163, 158, 156, 156, 156],
    [159, 161, 162, 160, 160, 159, 159, 159],
    [159, 160, 161, 162, 162, 155, 155, 155],
    [161, 161, 161, 161, 160, 157, 157, 157],
    [162, 162, 161, 163, 162, 157, 157, 157],
    [162, 162, 161, 161, 163, 158, 158, 158]
])

def Q(q):
    return np.array([[(1 + (1+i+j)*q) for j in range(8)] for i in range(8)])

if True:
    """
        Code utilisé pour créer les résultats préchargés
    """
    # La compression des 5 images peut prendre du temps
    image = a
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
    
    axs[0, 0].imshow(image)
    axs[0, 0].title.set_text('Matrice originale')

    def plot(axs, q):
        jpeg = compression(image, Q(q))
        # les taux de compression sont calculés sur la taille prise en mémoire par les variables python
        r = (1 - sys.getsizeof(str(jpeg)) / sys.getsizeof(image)) * 100
        axs.imshow(decompression(jpeg, Q(q)))
        axs.title.set_text(f'Matrice Q = {q}\nCompression : {round(r)}%')

    plot(axs[0, 1], 1)
    plot(axs[0, 2], 2)
    plot(axs[1, 0], 5)
    plot(axs[1, 1], 10)
    plot(axs[1, 2], 25)
    
else:
    """
        Code utilisé pour un exemple 8x8
    """
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
    
    axs[0, 0].imshow(test)
    axs[0, 0].title.set_text('Matrice originale')
    
    def plot(axs, q):
        m = compression8x8(test, Q(q))
        r = (1 - sys.getsizeof(m) / sys.getsizeof(test)) * 100
        axs.imshow(decompression8x8(m, Q(q)))
        axs.title.set_text(f'Matrice Q = {q}\nCompression : {round(r, 3)}%')

    plot(axs[0, 1], 1)
    plot(axs[0, 2], 2)
    plot(axs[1, 0], 5)
    plot(axs[1, 1], 10)
    plot(axs[1, 2], 15)

plt.show()

