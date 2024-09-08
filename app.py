import cv2
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

class RootLayout(BoxLayout):
    def __init__(self, **kwargs):
        super(RootLayout, self).__init__(**kwargs)

        self.capture = cv2.VideoCapture(0)

        Clock.schedule_interval(self.update, 1.0 / 30.0)
    
    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf = frame.tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            
            if "img_R1" in self.ids:
                self.ids.img_R1.texture = texture
    
    def on_stop(self):
        self.capture.release()

class Main_app(App):
    def build(self):
        return RootLayout()

if __name__ == "__main__":
    Main_app().run()