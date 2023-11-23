import cv2
import cv2.aruco as aruco
import numpy as np
import os
import time

def loadImages(path):
    mylist = os.listdir(path)
    noOfMarkers = len(mylist)
    print("Total Number of Markers detected:", noOfMarkers)
    augDic = {}
    for imgPath in mylist:
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv2.imread(f'{path}/{imgPath}')
        augDic[key] = imgAug
    return augDic

def findArucoMarkers(img, markerSize=6, totalMarkers=250, draw=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    aruco_dict = aruco.Dictionary_get(key)
    aruco_param = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(imgGray, aruco_dict, parameters=aruco_param)
    if draw and ids is not None:
        aruco.drawDetectedMarkers(img, bboxs)

    return [bboxs, ids]

def augmentAruco(bbox, id, img, imgAug, drawId=True, greenscreen_path="greenscreen.png"):
    tl = bbox[0][0]
    tr = bbox[0][1]
    br = bbox[0][2]
    bl = bbox[0][3]
    h, w, c = imgAug.shape
    pts1 = np.array([tl, tr, br, bl])
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix, _ = cv2.findHomography(pts2, pts1)
    imgOut = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))

    # Load the greenscreen image
    greenscreen = cv2.imread(greenscreen_path)

    # Resize greenscreen image to match the size of the augmented ArUco marker
    greenscreen = cv2.resize(greenscreen, (imgOut.shape[1], imgOut.shape[0]))

    # Create a mask to extract the ArUco marker region from the greenscreen image
    _, marker_mask = cv2.threshold(cv2.cvtColor(imgOut, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY_INV)
    greenscreen_extracted = cv2.bitwise_and(greenscreen, greenscreen, mask=marker_mask)

    # Create a mask to extract the background from the original image
    _, bg_mask = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    background = cv2.bitwise_and(img, img, mask=bg_mask)

    # Combine the ArUco marker with the greenscreen background
    imgOut = cv2.add(background, greenscreen_extracted)

    if drawId:
        cv2.putText(imgOut, str(id), tl, cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)

    return imgOut

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def distance_cm(p1, p2, pixels_to_cm):
    return distance(p1, p2) * pixels_to_cm

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera")
        return

    augDic = loadImages("Markers")
    prev_frames = {}
    prev_times = {}
    pixels_to_cm = 0.1 # Sesuaikan dengan informasi fisik dari ArUco marker atau objek yang digunakan
    update_interval = 0.5  # Interval pembacaan kecepatan (dalam detik)

    while True:
        ret, img = cap.read()
        if not ret:
            print("Error: Unable to capture frame")
            break

        arucoFound = findArucoMarkers(img)

        # Loop semua marker augmented satu per satu
        if len(arucoFound[0]) != 0:
            for bbox, ids in zip(arucoFound[0], arucoFound[1]):
                id = ids[0]
                if id in augDic.keys():
                    img = augmentAruco(bbox, id, img, augDic[id], drawId=False)

                    # Hitung kecepatan pergerakan marker
                    center = np.mean(bbox[0], axis=0)
                    if id in prev_frames:
                        distance_moved = distance(center, prev_frames[id])

                        # Hitung waktu yang diperlukan untuk perubahan tersebut
                        time_elapsed = time.time() - prev_times[id]

                        # Hitung kecepatan (jarak_perubahan / waktu_perubahan)
                        velocity = distance_cm(center, prev_frames[id], pixels_to_cm) / time_elapsed

                        x_start = int(bbox[0][0][0])
                        y_start = int(bbox[0][0][1])

                        # Ambil ukuran teks kecepatan
                        (text_width, text_height), _ = cv2.getTextSize(f"V: {velocity:.2f} cm/s",
                                                                       cv2.FONT_HERSHEY_PLAIN, 1.5, 2)

                        # Atur posisi teks ID sedikit lebih tinggi agar tidak terhalang
                        y_text = y_start - text_height - 15

                        # Tampilkan teks ID dan kecepatan pada frame
                        cv2.putText(img, f"ID: {id}", (x_start, y_text), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                        cv2.putText(img, f"V: {velocity:.2f} cm/s", (x_start, y_text + 20),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

                    # Simpan frame dan waktu sebelumnya untuk perhitungan selanjutnya
                    prev_frames[id] = center
                    prev_times[id] = time.time()

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Jeda interval pembacaan kecepatan
        time.sleep(update_interval)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

