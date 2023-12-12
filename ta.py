import cv2
from matplotlib import pyplot as plt

#gambar asli
img = plt.imread(r"C:\Users\Dell Latitude 5480\Documents\SMT 5\Pengolahan Citra\Tugas Besar\GetFoto3.jpg") #mengambil gambar dalam mode matplotlib

# deteksi senyum pada gambar asli
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml') 
deteksmileawal = plt.imread(r"C:\Users\Dell Latitude 5480\Documents\SMT 5\Pengolahan Citra\Tugas Besar\GetFoto3.jpg")
gray = cv2.cvtColor(deteksmileawal, cv2.COLOR_BGR2GRAY)
smiles = smile_cascade.detectMultiScale(gray, 1.3, 60)

for (x3, y3, w3, h3) in smiles:
    cv2.rectangle(deteksmileawal, (x3, y3), ((x3 + w3), (y3 + h3)), (0, 0, 255), 2)

#deteksi wajah pada gambar asli
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_alt.xml") #module deteksi wajah
detekwajahawal = plt.imread(r"C:\Users\Dell Latitude 5480\Documents\SMT 5\Pengolahan Citra\Tugas Besar\GetFoto3.jpg") 
gray = cv2.cvtColor(detekwajahawal, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4) #untuk menemukan wajah

for(x,y,w,h) in faces:
    cv2.rectangle(detekwajahawal, (x,y), (x+w, y+h), (255, 0,0), 4) #proses memberikan kotak di wajah

#pengaturan brightness & kontras
hasilimageenhanc = cv2.convertScaleAbs(img, alpha=2.7, beta=55)

# deteksi senyum pada gambar after
gambarsmile = cv2.convertScaleAbs(img, alpha=2.7, beta=55)
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml') 
faceROI = cv2.cvtColor(gambarsmile, cv2.COLOR_BGR2GRAY)
smiles = smile_cascade.detectMultiScale(faceROI, 1.3, 60)

for (x3, y3, w3, h3) in smiles:
            cv2.rectangle(gambarsmile, (x3, y3), ((x3 + w3), (y3 + h3)), (0, 0, 255), 2)

#deteksi wajah pada gambar after
gambarfinal =  cv2.convertScaleAbs(img, alpha=2.7, beta=55) #mengambil gambar imageenhanc
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_alt.xml") #module deteksi wajah
gray = cv2.cvtColor(gambarfinal, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for(x,y,w,h) in faces:
    cv2.rectangle(gambarfinal, (x,y), (x+w, y+h), (255, 0,0), 4) #proses memberikan kotak di wajah

titles = ['gambar before','deteksi senyum before','deteksi wajah before','after Brightness&Contrast','deteksi senyum after','deteksi wajah after']
images = [img, deteksmileawal, detekwajahawal, hasilimageenhanc, gambarsmile, gambarfinal]

for i in range(6): #mengambil 6 gambar
    plt.subplot(2,3,i+1),plt.imshow(images[i], cmap="gray") #menempatkan dan memunculkan gambar di matplotlib
    plt.title(titles[i]) #memberi judul pad gambar
    plt.xticks([]),plt.yticks([]) #menghapus koordinat sumbu x dan y

plt.show()

cv2.waitKey(0)
