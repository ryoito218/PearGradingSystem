from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
import time
import threading

class Root_Layout(GridLayout):
    def evaluate(self):
        threading.Thread(target=self._evaluate_background).start()
    
    def _evaluate_background(self):

        self.ids.navigation.text = "Processing..."
        time.sleep(3)
        self.ids.navigation.text = "Clicked !!"
        
        # img_num = 0
        # async_response = []

        # while True:
        #     frame = self.ids.camera_view

        #     if cv2.waitKey(1) & 0xFF == 13:
        #         img_num += 1
                
        #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
        #         inputs = 

        # if camera_view.frame is not None:
        #     image_frame = camera_view.frame
        #     print("Frame captured for grading:", image_frame.shape)

        # self.ids.navigation.text = "Clicked !!"

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