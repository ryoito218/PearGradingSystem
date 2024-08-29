import cv2

print(cv2.__version__)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("カメラを開けません")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        print("フレームのキャプチャに失敗しました")
        break

    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == 13:
        cv2.imwrite("captured_iamge.jpg", frame)
        print("写真を撮影し，'captured_image.jpg'として保存しました")
    
    elif cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()