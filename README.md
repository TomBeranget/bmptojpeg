# bmptojpeg
Compression jpeg sans bibliothèques.

## Principe
Application de la méthode de compression jpeg en utilisant différentes matrices de quantifications.
L'encodage en zig-zag est utilisé.
Ces matrices 8x8 sont de la forme : 
```python
np.array([[(1 + (1+i+j)*q) for j in range(8)] for i in range(8)])
```
Donc pour `q = 2`, par exemple :
![matrice](https://github.com/TomBeranget/bmptojpeg/blob/main/pictures/matrix.PNG?raw=true)

## Résultat sur un échantillon
On remarque que la matrice de quantification `q = 2` qui est utilisée généralement pour la compression diminue bien la taille mémoire tout en conservant une représentation relativement fidèle de la matrice initiale.
![matrice](https://github.com/TomBeranget/bmptojpeg/blob/main/pictures/echantillon.png?raw=true)
## Résultat des exemple d'image
### Le premier exemple
Sur cette image de taille 256x256, on ne remarque presque pas les effets de la pluspart des compressions utilisées qui pourtant font gagner beaucoup de place en mémoire.
![matrice](https://github.com/TomBeranget/bmptojpeg/blob/main/pictures/a2_base.png?raw=true)
En revanche, lors d'un zoom, on remarque qu'au delà de `q = 5`, on observe une grande perte d'information jusqu'a repérer les matrices 8x8 utilisées lors de la compression.
![matrice](https://github.com/TomBeranget/bmptojpeg/blob/main/pictures/a2_zoom.png?raw=true)
