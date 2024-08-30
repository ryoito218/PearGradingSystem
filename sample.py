import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("カメラを開くことができませんでした")
else:

    max_width = 1920
    max_height = 1080
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max_height)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"カメラの解像度: 幅={width} 高さ={height} ピクセル")

cap.release()