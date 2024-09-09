from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.core.text import LabelBase, DEFAULT_FONT
from kivy.resources import resource_add_path
from kivy.utils import platform

from tritonclient.utils import *
import tritonclient.http as httpclient

import os
import cv2
import time
import threading

font_path = os.path.join(os.path.dirname(__file__), "font", "NotoSansJP-Regular.ttf")
resource_add_path(os.path.dirname(font_path))
LabelBase.register(DEFAULT_FONT, font_path)

class Root_Layout(GridLayout):

    def __init__(self, **kwargs):
        super(Root_Layout, self).__init__(**kwargs)

        self.pear_num = 0
        self.img_num = 0
        self.is_capturing = False
        self.enter_key_pressed_flag =False

        self.client = httpclient.InferenceServerClient("133.35.129.4:8000")

        Window.bind(on_key_down=self.on_key_down)

    def evaluate(self):
        self.is_capturing = True
        threading.Thread(target=self._evaluate_background).start()
    
    def _evaluate_background(self):
        
        self.pear_num += 1
        self.img_num = 0
        async_response = []

        self.create_folder()

        self.show_message("No.1\nPlease press Enter!")

        while self.img_num < 3 and self.is_capturing:
            
            if self.ids.camera_view is not None:
                time.sleep(0.1)

                if self.enter_key_pressed():
                    frame = self.capture_image()

                    # inputs = [
                    #     httpclient.InferInput("IMAGE", frame.shape, np_to_triton_dtype(frame.dtype)),
                    # ]

                    # inputs[0].set_data_from_numpy(frame)

                    # outputs = [
                    #     httpclient.InferRequestedOutput("AREA"),
                    #     httpclient.InferRequestedOutput("NUMBER"),
                    #     httpclient.InferRequestedOutput("OUTPUT_IMAGE"),
                    #     httpclient.InferRequestedOutput("SPEED"),
                    # ]

                    # async_response.append(
                    #     self.client.async_infer(
                    #         model_name="pear_evaluator",
                    #         inputs=inputs,
                    #         outputs=outputs
                    #     )
                    # )
        
        # areas = np.array([0,0,0,0,0,0]).astype(np.uint64)
        # num = 0

        # for i in range(len(async_response)):
        #     result = async_response[i].get_result()
        #     area = result.as_numpy("AREA")
        #     number = result.as_numpy("NUMBER")
        #     speed = result.as_numpy("SPEED")
        #     output_image = result.as_numpy("OUTPUT_IMAGE")

        #     output_name = f"images/output/{self.pear_num}/{self.pear_num}_{i+1}.png"
        #     cv2.imwrite(output_name, output_image)

        #     areas += area
        #     num += number[0]
        
        # evaluation = [0,0,0,0,0]

        # # 黒斑病
        # if num <= 1:
        #     pass
        # elif num <= 3 and areas[0]/areas[5] <= 1/3:
        #     evaluation[0] = 1
        # else:
        #     evaluation[0] = 2
        
        # # 外傷痕
        # if areas[1]/areas[5] <=1/10:
        #     pass
        # elif areas[1]/areas[5] <= 1/3:
        #     evaluation[1] = 1
        # else:
        #     evaluation[1] = 2
        
        # # 斑点状汚損
        # if areas[2]/areas[5] <= 1/10:
        #     pass
        # elif areas[2]/areas[5] <= 1/3:
        #     evaluation[2] = 1
        # else:
        #     evaluation[2] = 2
        
        # # 面状汚損
        # if areas[3]/areas[5] <= 1/10:
        #     pass
        # elif areas[3]/areas[5] <= 1/3:
        #     evaluation[3] = 1
        # else:
        #     evaluation[3] = 2
        
        # # 薬班
        # if areas[4]/areas[5] <= 1/10:
        #     pass
        # elif areas[4]/areas[5] <= 1/3:
        #     evaluation[4] = 1
        # else:
        #     evaluation[4] = 2
        
        # max_value = max(evaluation)

        # if max_value == 0:
        #     self.show_result("Red")
        # elif max_value == 1:
        #     self.show_result("Blue")
        # else:
        #     self.show_result("Normal")

        self.show_result("赤秀")

        self.show_message("Please execute")
        self.is_capturing = False
    
    def show_message(self, message):
        self.ids.navigation.text = message
    
    def show_result(self, result):
        self.ids.result.text = result

    def create_folder(self):
        input_folder_path = f"images/input/{self.pear_num}"
        output_folder_path = f"images/output/{self.pear_num}"

        if not os.path.exists(input_folder_path):
            os.makedirs(input_folder_path)

        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
    
    def enter_key_pressed(self):
        if self.enter_key_pressed_flag:
            self.enter_key_pressed_flag = False
            return True
        return False

    def on_key_down(self, instance, keyboard, keycode, text, modifiers):
        if keycode == 40:
            self.enter_key_pressed_flag = True

    def capture_image(self):
        camera_view = self.ids.camera_view
        frame = camera_view.frame

        if frame is not None:
            self.img_num += 1
            input_name = f"images/input/{self.pear_num}/{self.img_num}.png"
            cv2.imwrite(input_name, frame)
            self.show_message(f"No. {self.img_num + 1}\nPlease press Enter!")
            return frame
        return False

class CameraView(Image):
    def __init__(self, **kwargs):
        super(CameraView, self).__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/30)
    
    def update(self, dt):
        ret, self.frame = self.capture.read()
        if ret:
            buf = cv2.flip(self.frame, 0).tobytes()
            texture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt="bgr")
            texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
            self.texture = texture

    def on_stop(self):
        self.capture.release()

class Main_app(App):
    title = "Pear Grading System"

if __name__ == "__main__":
    Main_app().run()