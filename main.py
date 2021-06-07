from tkinter import *
from tkinter import filedialog
import os
import math
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# class containing all the functions to be performed on image


class project:

    def plotimage(self):

        fig = plt.figure(figsize=(3, 3))
        plt.imshow(self.res.astype(np.uint8))

        fig.savefig('result.png')
        photo = PhotoImage(file="result.png")

        lbl.configure(image=photo)
        lbl.image = photo

    def showimage(self):
        self.fln = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image", filetypes=(
            ("JPEG FILE", "*.jpeg"), ("PNG FILE", "*.png"), ("ALL FILES", "*.*")))
        self.img = Image.open(self.fln)
        self.img.thumbnail((300, 300))
        self.photo = ImageTk.PhotoImage(self.img)

        lbl.configure(image=self.photo)
        lbl.image = self.photo

    def original(self):
        lbl.configure(image=self.photo)
        lbl.image = self.photo

    def salt_pepper_noise(self):
        fname = self.fln
        img = cv.imread(fname)
        c, h, w = img.shape
        r = np.random.choice((0, 1, 2), size=(c, h, w), p=[0.9, 0.05, 0.05])
        self.res = np.copy(img)
        self.res[r == 1] = 255
        self.res[r == 2] = 0

        p1.plotimage()
        pass

    # Add gaussian noise with mean 0 and variance 0.59
    def Gaussian_noise(self):
        fname = self.fln
        A = cv.imread(fname)

        noise = np.random.normal(0, .59, A.shape)

        self.res = A + noise
        p1.plotimage()

    def Rayleigh_noise(self):
        # Rayleigh noise
        fname = self.fln
        img = cv.imread(fname)
        c, h, w = img.shape
        r = np.random.choice((10, 11, 8, 16, 0, 10), size=(c, h, w))

        a = 10
        b = 8
        imgsp = np.copy(img)

        for x in r:
            for y in x:
                for z in y:
                    if (z >= a):
                        z = 2 * (z - a) * math.exp(-((z - a) ** 2) / b) / b
                    else:
                        z = 0
        self.res = imgsp + r
        p1.plotimage()

    def Exponential_noise(self):
        # Exponential Noise

        fname = self.fln
        img = cv.imread(fname)

        c, h, w = img.shape
        r = np.random.choice((-20, -10, -15, 0, 10, 6, 7), size=(c, h, w))

        a = 10
        img_sp3 = np.copy(img)

        for x in r:
            for y in x:
                for z in y:
                    if (z >= 0):
                        z = a * math.exp(-a * z)
                    else:
                        z = 0
        self.res = img_sp3 + r
        p1.plotimage()

    def Gamma_noise(self):

        fname = self.fln
        img = cv.imread(fname)
        c, h, w = img.shape
        r = np.random.choice((-20, 10, 11, 8, 6, 0, 10), size=(c, h, w))

        a = 10
        b = 8
        img_sp4 = np.copy(img)

        for x in r:
            for y in x:
                for z in y:
                    if (z >= 0):
                        z = a ** b * \
                            z ** (b - 1) * math.exp(-a * z) / \
                            math.factorial(b - 1)
                    else:
                        z = 0
        self.res = img_sp4 + r
        p1.plotimage()

    def Uniform_noise(self):
        fname = self.fln
        img = cv.imread(fname)
        c, h, w = img.shape
        r = np.random.choice((0, 5, 12, 10, 13, 14), size=(c, h, w))
        a = 10
        b = 14
        img_sp2 = np.copy(img)

        for x in r:
            for y in x:
                for z in y:
                    if (z >= a or z <= b):
                        z = 1 / b - a
                    else:
                        z = 0
        self.res = img_sp2 + r
        p1.plotimage()

    def fourier_transform(self):
        fname = self.fln
        img = cv.imread(fname, 0)
        f = np.fft.fft2(img)
        self.res = 10 * np.log(np.abs(f))
        p1.plotimage()

    def shift_operation(self):
        fname = self.fln
        img = cv.imread(fname, 0)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        self.res = 10 * np.log(np.abs(fshift))
        p1.plotimage()

    def inverse_fourier(self):
        fname = self.fln
        img = cv.imread(fname, 0)
        f = np.fft.fft2(img)
        finverse = np.fft.ifft2(f)
        self.res = 10 * np.log(np.abs(finverse))
        p1.plotimage()

    def fourier_spectrum(self):
        fname = self.fln
        img = cv.imread(fname, 0)
        f = np.fft.fft2(img)
        fspec = np.abs(f)
        self.res = np.fft.ifft2(fspec)
        p1.plotimage()

    def power_spectrum(self):
        fname = self.fln
        img = cv.imread(fname, 0)
        f = np.fft.fft2(img)
        fspec = np.abs(f)
        f_pow_spec = np.square(fspec)
        img_back = np.fft.ifft2(f_pow_spec)
        self.res = 10 * np.log(np.abs(img_back))
        p1.plotimage()

    def averaging(self):
        fname = self.fln
        img = cv.imread(fname)
        kernel = np.ones((3, 3), np.float32) / 9
        self.res = cv.filter2D(img, -1, kernel)
        p1.plotimage()

    def blurr(self):
        fname = self.fln
        img = cv.imread(fname)
        self.res = cv.blur(img, (5, 5))
        p1.plotimage()

    def guass_blurr(self):
        fname = self.fln
        img = cv.imread(fname)
        self.res = cv.GaussianBlur(img, (5, 5), 0)
        p1.plotimage()

    def median_blurr(self):
        fname = self.fln
        img = cv.imread(fname)
        self.res = cv.medianBlur(img, 5)
        p1.plotimage()

# function to resize backgroung image along with the size of window


def resize_image(event):
    new_width = event.width
    new_height = event.height
    image = copy_of_image.resize((new_width, new_height))
    photo = ImageTk.PhotoImage(image)
    back.config(image=photo)
    back.image = photo  # avoid garbage collection


# defining wingow of gui
root = Tk()

root.title("DIP Calculator")
root.geometry("800x800")

root.minsize(500, 500)
root.maxsize(1800, 500)

# applyig background image
image = Image.open('sky.png')
copy_of_image = image.copy()
photo = ImageTk.PhotoImage(image)
back = ttk.Label(root, image=photo)
back.bind('<Configure>', resize_image)
back.pack(fill=BOTH, expand=YES)

# object of class project
p1 = project()

# frame containing all functioning buttons
frm = Frame(back, bg='#895f69')
frm.pack(anchor="nw", side=LEFT, padx=15, pady=25)

# frame containing image on which operation is going to be done
frm1 = Frame(back, width=500, height=500)
frm1.pack(side=TOP, fill=None, expand=False,
          padx=30, pady=100,  anchor="se",)

# label to show result of image after applying operations
lbl = Label(frm1, borderwidth=3, relief=SUNKEN, text="Your Image here.")
lbl.pack(side=RIGHT, fill=None, expand=False)

# Grid to arrange buttons in specific row and column
Grid.rowconfigure(root, 0, weight=1)
Grid.columnconfigure(root, 0, weight=1)

Grid.rowconfigure(root, 1, weight=1)

# all buttons with specific function
browser = Button(frm, text="Browse_Image", bg='grey',
                 fg='white',  command=p1.showimage)
browser.grid(row=6, column=0, padx=10, pady=10, sticky="NSEW")

btn2 = Button(frm, text="Original_Image", bg='grey',
              fg='white', command=p1.original)
btn2.grid(row=0, column=0, padx=10, pady=10, sticky="NSEW")


btn3 = Button(frm, text="Salt_noise", bg='grey',
              fg='white', command=p1.salt_pepper_noise)
btn3.grid(row=0, column=2, padx=10, pady=10, sticky="NSEW")

btn4 = Button(frm, text="Gaussian_noise", bg='grey',
              fg='white', command=p1.Gaussian_noise)
btn4.grid(row=1, column=0, padx=10, pady=10, sticky="NSEW")

btn5 = Button(frm, text="Rayleigh_noise", bg='grey',
              fg='white', command=p1.Rayleigh_noise)
btn5.grid(row=1, column=1, padx=10, pady=10, sticky="NSEW")

btn6 = Button(frm, text="Exponential_noise", bg='grey',
              fg='white', command=p1.Exponential_noise)
btn6.grid(row=1, column=2, padx=10, pady=10, sticky="NSEW")

btn7 = Button(frm, text="Gamma_noise", bg='grey',
              fg='white', command=p1.Gamma_noise)
btn7.grid(row=2, column=0, padx=10, pady=10, sticky="NSEW")

btn8 = Button(frm, text="Uniform_noise", bg='grey',
              fg='white', command=p1.Uniform_noise)
btn8.grid(row=2, column=1, padx=10, pady=10, sticky="NSEW")

btn9 = Button(frm, text="Fourier_transform", bg='grey',
              fg='white', command=p1.fourier_transform)
btn9.grid(row=2, column=2, padx=10, pady=10, sticky="NSEW")

btn10 = Button(frm, text="Shift_operation", bg='grey',
               fg='white', command=p1.shift_operation)
btn10.grid(row=3, column=0, padx=10, pady=10, sticky="NSEW")

btn11 = Button(frm, text="Inverse_fourier", bg='grey',
               fg='white', command=p1.inverse_fourier)
btn11.grid(row=3, column=1, padx=10, pady=10, sticky="NSEW")

btn12 = Button(frm, text="Fourier_spectrum", bg='grey',
               fg='white', command=p1.fourier_spectrum)
btn12.grid(row=3, column=2, padx=10, pady=10, sticky="NSEW")

btn13 = Button(frm, text="Power_spectrum", bg='grey',
               fg='white', command=p1.power_spectrum)
btn13.grid(row=4, column=0, padx=10, pady=10, sticky="NSEW")

btn14 = Button(frm, text="Averaging_effect", bg='grey',
               fg='white', command=p1.averaging)
btn14.grid(row=4, column=1, padx=10, pady=10, sticky="NSEW")

btn15 = Button(frm, text="Blur", bg='grey', fg='white', command=p1.blurr)
btn15.grid(row=4, column=2, padx=10, pady=10, sticky="NSEW")

btn16 = Button(frm, text="Gaussian_blur", bg='grey',
               fg='white', command=p1.guass_blurr)
btn16.grid(row=5, column=0, padx=10, pady=10, sticky="NSEW")

btn17 = Button(frm, text="Median_blur", bg='grey',
               fg='white', command=p1.median_blurr)
btn17.grid(row=5, column=2, padx=10, pady=10, sticky="NSEW")

exit_btn = Button(frm, text="Exit", bg='grey',
                  fg='white', command=lambda: exit())
exit_btn.grid(row=6, column=2, padx=10, pady=10, sticky="NSEW")

# window loop so that it will run untill we press cross button
root.mainloop()
