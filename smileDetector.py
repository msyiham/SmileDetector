import cv2
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

# Fungsi untuk menampilkan gambar di Tkinter window
def display_image(image, title, row, column, width=None, height=None):
    if width and height:
        image = cv2.resize(image, (width, height))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    photo = ImageTk.PhotoImage(image=image)

    label = ttk.Label(root, image=photo, text=title, compound=tk.BOTTOM)
    label.image = photo
    label.grid(row=row, column=column, padx=10, pady=10)
    label.bind('<Button-1>', lambda e: print(title))

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

# Fungsi untuk membuka file dialog dan memilih gambar
def open_file_dialog():
    global file_path
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        process_image(file_path)
        hide_welcome_message()

# Fungsi untuk memproses gambar
def process_image(file_path):
    img = cv2.imread(file_path)

    # Atur ukuran yang diinginkan (misalnya, lebar=300, tinggi=400)
    desired_width = 320
    desired_height = 200

    # Resize gambar
    img_resized = cv2.resize(img, (desired_width, desired_height))

    # Deteksi wajah dan senyum pada gambar asli
    deteksmileawal = img.copy()
    deteksmileawal = detect_features(deteksmileawal, smile_cascade, face_cascade)

    # Pengaturan brightness & kontras
    hasilimageenhanc = cv2.convertScaleAbs(img, alpha=2.7, beta=55)

    # Deteksi wajah dan senyum pada gambar setelah perubahan brightness & kontras
    gambarsmile = hasilimageenhanc.copy()
    gambarsmile = detect_features(gambarsmile, smile_cascade, face_cascade)

    # Tampilkan hasil di Tkinter window
    display_image(img_resized, "Gambar Asli", row=0, column=0, width=desired_width, height=desired_height)
    display_image(deteksmileawal, "Deteksi Wajah & Senyum Sebelum", row=0, column=1, width=desired_width, height=desired_height)
    display_image(hasilimageenhanc, "Setelah Brightness & Kontras", row=1, column=0, width=desired_width, height=desired_height)
    display_image(gambarsmile, "Deteksi Wajah & Senyum Setelah", row=1, column=1, width=desired_width, height=desired_height)

# Fungsi untuk menampilkan pesan selamat datang
def display_welcome_message():
    welcome_label.config(text="Selamat Datang! Silakan pilih gambar untuk memulai.")

# Fungsi untuk menyembunyikan pesan selamat datang
def hide_welcome_message():
    welcome_label.config(text="")

# Inisialisasi cascade classifiers
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')

# Buat window Tkinter
root = tk.Tk()
root.title("Deteksi Wajah dan Senyum")

# Atur ukuran tetap untuk window
fixed_width = 700
fixed_height = 550
root.geometry(f"{fixed_width}x{fixed_height}")

# Label untuk pesan selamat datang
welcome_label = ttk.Label(root, text="Selamat Datang! Silakan pilih gambar untuk memulai.")
welcome_label.grid(row=2, column=0, columnspan=2, pady=10)

# Tombol untuk memilih gambar
button = ttk.Button(root, text="Pilih Gambar", command=open_file_dialog)
button.grid(row=3, column=0, columnspan=2, pady=10)

# Jalankan loop utama Tkinter
while True:
    root.mainloop()  # Tkinter main loop

    # If the user closes the Tkinter window or chooses to exit the program, break out of the loop
    if not button.instate(['pressed']):
        break

    # Process the selected image
    process_image(file_path)
    hide_welcome_message()

    # Reinitialize Tkinter window
    root = tk.Tk()
    root.configure(bg="black")
    root.title("Deteksi Wajah dan Senyum")
    root.geometry(f"{fixed_width}x{fixed_height}")
    

    # Label untuk pesan selamat datang
    welcome_label = ttk.Label(root, text="Selamat Datang! Silakan pilih gambar untuk memulai.")
    welcome_label.grid(row=2, column=0, columnspan=2, pady=10)

    # Tombol untuk memilih gambar
    button = ttk.Button(root, text="Pilih Gambar", command=open_file_dialog)
    button.grid(row=3, column=0, columnspan=2, pady=10)

    # Update the Tkinter window
    root.update_idletasks()

# End of the program
