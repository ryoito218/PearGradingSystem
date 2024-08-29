import cv2
import os
import time
from tritonclient.utils import *
import tritonclient.http as httpclient


print(cv2.__version__)


cap = cv2.VideoCapture(0)

count = 0

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
        
        count += 1
        async_responses = []
        client = httpclient.InferenceServerClient("133.35.129.4:8000")

        input_folder_path = f"images/input/{count}"
        output_folder_path = f"images/output/{count}"

        if not os.path.exists(input_folder_path):
            os.makedirs(input_folder_path)
            
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        
        for i in range(3):
            
            print(f"{i+1}回目の撮影 Enterを押してください")
            if cv2.waitKey(1) & 0xFF == 13:
                input_name = f"images/input/{count}/{count}_{i+1}.png"

                cv2.imwrite(input_name, frame)
                print("撮影完了")
                time.sleep(0.1)
            
            inputs = [
                httpclient.InferInput("IMAGE", frame.shape, np_to_triton_dtype(frame.dtype)),
            ]

            inputs[0].set_data_from_numpy(frame)

            outputs = [
                httpclient.InferRequestedOutput("AREA"),
                httpclient.InferRequestedOutput("NUMBER"),
                httpclient.InferRequestedOutput("OUTPUT_IMAGE"),
                httpclient.InferRequestedOutput("SPEED"),
            ]

            async_responses.append(
                client.async_infer(
                    model_name="pear_evaluator",
                    inputs=inputs,
                    outputs=outputs
                )
            )
        
        areas = np.array([0,0,0,0,0,0]).astype(np.uint64)
        num = 0

        for i in range(len(async_responses)):
            result = async_responses[i].get_result()
            area = result.as_numpy("AREA")
            number = result.as_numpy("NUMBER")
            speed = result.as_numpy("SPEED")
            output_image = result.as_numpy("OUTPUT_IMAGE")

            output_name = f"images/output/{count}/{count}_{i+1}.png"
            cv2.imwrite(output_name, output_image)

            areas += area
            num += number[0]
        
        evaluation = [0,0,0,0,0]

        # 黒斑病
        if num <= 1:
            pass
        elif num <= 3 and areas[0]/areas[5] <= 1/3:
            evaluation[0] = 1
        else:
            evaluation[0] = 2
        
        # 外傷痕
        if areas[1]/areas[5] <= 1/10:
            pass
        elif areas[1]/areas[5] <= 1/3:
            evaluation[1] = 1
        else:
            evaluation[1] =2
            
        # 斑点状汚損
        if areas[2]/areas[5] <= 1/10:
            pass
        elif areas[2]/areas[5] <= 1/3:
            evaluation[2] = 1
        else:
            evaluation[2] = 2
            
        # 面状汚損
        if areas[3]/areas[5] <= 1/10:
            pass
        elif areas[3]/areas[5] <= 1/3:
            evaluation[3] = 1
        else:
            evaluation[3] = 2
            
        # 薬班
        if areas[4]/areas[5] <= 1/10:
            pass
        elif areas[4]/areas[5] <= 1/3:
            evaluation[4] = 1
        else:
            evaluation[4] = 2
            
        max_value = max(evaluation)

        if max_value == 0:
            print('検査終了')
            print('検査結果は 赤秀 です。')
        elif max_value == 1:
            print('検査終了')
            print('検査結果は 青秀 です。')
        else:
            print('検査終了')
            print('検査結果は 良 です。')
    
    elif cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()