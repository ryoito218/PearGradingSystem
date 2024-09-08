import cv2
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.core.window import Window
import os
import cv2
import time
import threading

class Root_Layout(GridLayout):

    def __init__(self, **kwargs):
        super(Root_Layout, self).__init__(**kwargs)
        self.pear_num = 0
        self.img_num = 0
        self.is_capturing = False
        self.enter_key_pressed_flag =False

        Window.bind(on_key_down=self.on_key_down)


    def show_message(self, message):
        self.ids.navigation.text = message

    def create_folder(self):
        input_folder_path = f"images/input/{self.pear_num}"
        output_folder_path = f"images/output/{self.pear_num}"

        if not os.path.exists(input_folder_path):
            os.makedirs(input_folder_path)

        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

    def evaluate(self):
        self.is_capturing = True
        threading.Thread(target=self._evaluate_background).start()
    
    def _evaluate_background(self):
        
        self.pear_num += 1
        img_num = 0
        async_response = []

        self.create_folder()

        self.show_message("No.1\nPlease press Enter!")

        while self.img_num < 3 and self.is_capturing:
            if self.ids.camera_view is not None:
                time.sleep(0.1)

                if self.enter_key_pressed():
                    self.capture_image()
            
        self.show_message("Capture complete")
        self.is_capturing = False
    
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