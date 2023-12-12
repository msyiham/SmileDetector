import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Fungsi untuk menampilkan gambar di Tkinter window
def display_image(image, title, row, column):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    photo = ImageTk.PhotoImage(image=image)
    
    label = ttk.Label(root, image=photo)
    label.image = photo
    label.grid(row=row, column=column, padx=10, pady=10)
    label.bind('<Button-1>', lambda e: print(title))  # Menggunakan bind untuk menangani klik pada gambar

# Fungsi untuk mendeteksi wajah dan senyum
def detect_features(image, smile_cascade, face_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Deteksi senyum
    smiles = smile_cascade.detectMultiScale(gray, 1.3, 60)
    for (x3, y3, w3, h3) in smiles:
        cv2.rectangle(image, (x3, y3), ((x3 + w3), (y3 + h3)), (0, 0, 255), 2)

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 4)
    
    return image

# Baca gambar
img = cv2.imread(r"C:\Users\Dell Latitude 5480\Documents\SMT 5\Pengolahan Citra\Tugas Besar\GetFoto3.jpg")

# Inisialisasi cascade classifiers
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')

# Deteksi wajah dan senyum pada gambar asli
deteksmileawal = img.copy()
deteksmileawal = detect_features(deteksmileawal, smile_cascade, face_cascade)

# Pengaturan brightness & kontras
hasilimageenhanc = cv2.convertScaleAbs(img, alpha=2.7, beta=55)

# Deteksi wajah dan senyum pada gambar setelah perubahan brightness & kontras
gambarsmile = hasilimageenhanc.copy()
gambarsmile = detect_features(gambarsmile, smile_cascade, face_cascade)

# Buat window Tkinter
root = tk.Tk()
root.title("Deteksi Wajah dan Senyum")

# Tambahkan gambar ke Tkinter window
display_image(img, "Gambar Asli", row=0, column=0)
display_image(deteksmileawal, "Deteksi Wajah & Senyum Sebelum", row=0, column=1)
display_image(hasilimageenhanc, "Setelah Brightness & Kontras", row=1, column=0)
display_image(gambarsmile, "Deteksi Wajah & Senyum Setelah", row=1, column=1)

# Jalankan loop utama Tkinter
root.mainloop()
