import pyscreenshot as imgrab
import os

if __name__ == '__main__':
    saving_path = "C:/Users/JoongsooPark/Desktop/training/"
    path, dirs, files = os.walk(saving_path).__next__()
    file_count = len(files)
    im = imgrab.grab(bbox=(0, 0, 360, 720))
    im.save(saving_path + str(file_count+1) + ".png")

