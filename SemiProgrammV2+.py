import cv2, os, glob
import numpy as np
from PIL import Image

global RGBdBild # Durchschnittlicher RGB über alle Bilder
global MAximwarhier

def Ordnerauslesen(Bildpfad):
    bilder_liste = glob.glob(os.path.join(Bildpfad, '*.jpeg'))
    print(f'Es wurden {str(len(bilder_liste))} Bilder in dem Ordner gefunden')
    #ydalleBild = HelligkeitjedesBilder(bilder_liste)
    #print(f'Durchschnittliche Helligkeit aller Bilder: {ydalleBild}')
    Bilderzusammenfügen(bilder_liste, Image.open(bilder_liste[0]).size)

def Helligkeitverändern(dBild, Faktor):

    img = Image.open(dBild)
    data = np.array(img)

    #Berechnen Sie den Durchschnitt der Pixelwerte
    durchschnitt = np.mean(data)
    # Berechnen Sie die Differenz zwischen jedem Pixel und dem Durchschnitt
    differenz = data - durchschnitt
    # Pixelwerte unter dem Durchschnitt verdunkeln
    data[data < durchschnitt] -= np.abs(Faktor * differenz[data < durchschnitt]).astype(data.dtype)
    print(Faktor * differenz[data < durchschnitt])
    # Pixelwerte über dem Durchschnitt aufhellen
    data[data > durchschnitt] += np.abs(Faktor * differenz[data > durchschnitt]).astype(data.dtype)
    #for i in range(img.size[1]):
    #     for j in range(img.size[0]):
    #         if data[i][j] < durchschnitt:
    #             data[i][j] -= np.abs(data[i][j] - durchschnitt) * Faktor
    #             print(data[i][j])
    #         else:
    #            data[i][j] += np.abs(data[i][j]-durchschnitt) * Faktor
    #            print(data[i][j])
    # Auf den Wertebereich von uint8 beschränken (0-255)
    data = np.clip(data, 0, 255).astype(np.uint8)
    Image.fromarray(data.astype(np.uint8)).save("manipuliertes_bild.jpg")
def Bilderzusammenfügen(ListeBilder, imgsize):
    umgedrehte_imgsize = (imgsize[1], imgsize[0])
    durchschnittsfoto = np.zeros(umgedrehte_imgsize, dtype=np.float64)
    i=0
    for Bild in ListeBilder:
        img = Image.open(Bild)

        # Bild in Graustufen konvertieren und zu einem Numpy-Array konvertieren
        bild_array = np.array(img.convert('L'), dtype=np.float64)
        print(i)
        i+=1

        durchschnittsfoto += bild_array
    durchschnittsfoto /=len(ListeBilder)
    durchschnittsfoto = durchschnittsfoto.astype(np.uint8)
    Image.fromarray(durchschnittsfoto.astype(np.uint8)).save("Ausgansbild.jpg")

Ordnerauslesen('extrahierte_bilder')
Helligkeitverändern(r'C:\Users\soere\PycharmProjects\Seminarfach\Ausgansbild.jpg',2)
