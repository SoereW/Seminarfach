from PIL import Image
from numpy import array
import os

image = Image.open('example.jpg')
images = []
grayscale_image_arrays = []
cwd = os.getcwd()


def averaging_pictures(folder_name):
    for foto in os.listdir(cwd + r'\\' + folder_name):  # durchgehen des bilder-ordners, https://www.youtube.com/watch?v=EbHgUA4Sgq4
        images.append(Image.open(cwd + r'\\' + folder_name + r'\\' + foto))  # bildobjekte in liste 'images' gespeichert

    for i in images:
        grayscale_image_arrays.append(array(i.convert('L')))  # umwandlung in grayscale-bilder umgewandelt, https://www.youtube.com/watch?v=5QR-dG68eNE
        # brightness-arrays werden in der liste 'grayscale_brightness_arrays' gespeichert
    width, height = images[0].size
    averaged_picture = Image.new('L', (width, height))  # https://pythonexamples.org/python-pillow-create-image/

    for i in range(width):
        for j in range(height):
            brightness_sum = 0
            for bild_array in grayscale_image_arrays:
                brightness_sum += bild_array[j][i]
            averaged_picture.putpixel((i, j), int(brightness_sum / (len(grayscale_image_arrays)))) #https://www.youtube.com/watch?v=5QR-dG68eNE
    return averaged_picture





def analyse(im):
    #sw_bild = im.convert('L')
    # sw_bild.show()
    or_brightness = array(im)
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
                if brightness_values[i][j] + 3 * (brightness_values[i][j] - median) <= 255:
                    brightness_values[i][j] += 3 * (brightness_values[i][j] - median)
                else:
                    brightness_values[i][j] = 255
            elif brightness_values[i][j] < median:
                if brightness_values[i][j] + 3 * (brightness_values[i][j] - median) >= 0:
                    brightness_values[i][j] += 3 * (brightness_values[i][j] - median)
                else:
                    brightness_values[i][j] = 0
            # print(type(brightness_values[i][j]))
            kontrast_bild.putpixel((j, i), int(brightness_values[i][j]))  # https://www.youtube.com/watch?v=5QR-dG68eNE
    return kontrast_bild


durchschnitt_bild = averaging_pictures('bilder')
new_fabulous_extremly_sophisticated_artwork = analyse(durchschnitt_bild)
new_fabulous_extremly_sophisticated_artwork.save('supernew_fabulous_extremly_sophisticated_artwork.jpg')
durchschnitt_bild.save('average.jpg')
durchschnitt_bild.show()
new_fabulous_extremly_sophisticated_artwork.show()
