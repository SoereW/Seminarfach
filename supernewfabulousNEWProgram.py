import cv2, os, glob
import numpy as np
from PIL import Image


def VideoBilder(Videopfad):
    #vorherige Ergebnisse löschen
    for filename in os.listdir('extrahierte_bilder'):
        os.remove(os.path.join('extrahierte_bilder', filename))

    video_path = Videopfad
    output_folder = 'extrahierte_bilder'
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Fehler beim Öffnen des Videos.")

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_path = os.path.join(output_folder, f'frame_{count}.jpg')
        cv2.imwrite(frame_path, frame)

        count += 1
    cap.release()
    cv2.destroyAllWindows()
def Ordnerauslesen(Bildpfad):
    bilder_liste = glob.glob(os.path.join(Bildpfad, '*.jpg'))
    print(f'Es wurden {str(len(bilder_liste))} Bilder in dem Ordner gefunden' + " ---  geschätzte Zeit:" + str(len(bilder_liste)/57))
    Bilderzusammenfügen(bilder_liste, Image.open(bilder_liste[0]).size)

def Bilderzusammenfügen(ListeBilder, imgsize):
    umgedrehte_imgsize = (imgsize[1], imgsize[0])
    durchschnittsfoto = np.zeros(umgedrehte_imgsize, dtype=np.float64)
    for Bild in ListeBilder:
        img = Image.open(Bild)
        # Bild in Graustufen konvertieren und zu einem Numpy-Array konvertieren
        bild_array = np.array(img.convert('L'), dtype=np.float64)
        durchschnittsfoto += bild_array
    durchschnittsfoto /=len(ListeBilder)
    print("Durschnittshelligkeit vom Ausgangsbild beträgt:" + str(np.sum(durchschnittsfoto)/(imgsize[0]*imgsize[1])))
    durchschnittsfoto = durchschnittsfoto.astype(np.uint8)

    Image.fromarray(durchschnittsfoto.astype(np.uint8)).save("Ausgansbild.jpg")
def analyse(im, faktor):
    #sw_bild = im.convert('L')
    # sw_bild.show()
    or_brightness = np.array(im)
    brightness_values = or_brightness
    # print(array(sw_bild))

    breite, hohe = im.size
    kontrast_bild = Image.new('L', (breite, hohe))
    counter = 0
    for i in range(hohe):
        for j in range(breite):
            counter += brightness_values[i][j]

    median = counter / (breite * hohe)

    for i in range(hohe):
        for j in range(breite):
            if brightness_values[i][j] > median:
                if brightness_values[i][j] + faktor * (brightness_values[i][j] - median) <= 255:
                    brightness_values[i][j] += faktor * (brightness_values[i][j] - median)
                else:
                    brightness_values[i][j] = 255
            elif brightness_values[i][j] < median:
                if brightness_values[i][j] + faktor * (brightness_values[i][j] - median) >= 0:
                    brightness_values[i][j] += faktor * (brightness_values[i][j] - median)
                else:
                    brightness_values[i][j] = 0
            # print(type(brightness_values[i][j]))
            kontrast_bild.putpixel((j, i), int(brightness_values[i][j]))  # https://www.youtube.com/watch?v=5QR-dG68eNE
    return kontrast_bild
VideoBilder(r"C:\Users\soere\OneDrive\Videos\SeminarfachProjektVideos\Aufnahmen28.2.24ExperimentPolarisation\Geschnitten\Video32\117.mp4")
Ordnerauslesen('extrahierte_bilder')
img = Image.open('Ausgansbild.jpg')
neuesBild = analyse(img,5)
neuesBild.save('Harald.jpg')