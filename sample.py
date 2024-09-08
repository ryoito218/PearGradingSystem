from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2

class MainScreen(Widget):
    pass

class MyCameraApp(App):
    def build(self):
        return MainScreen()

class CameraPreview(Image):
    def __init__(self, **kwargs):
        super(CameraPreview, self).__init__(**kwargs)
        # 0番目のカメラに接続
        self.capture = cv2.VideoCapture(0)
        # 描画のインターバルを設定
        Clock.schedule_interval(self.update, 1.0 / 30)

    # インターバルで実行する描画メソッド
    def update(self, dt):
        # フレームを読み込み
        ret, self.frame = self.capture.read()
        # Kivy Textureに変換
        buf = cv2.flip(self.frame, 0).tostring()
        texture = Texture.create(size=(self.frame.shape[1], self.frame.shape[0]), colorfmt='bgr') 
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # インスタンスのtextureを変更
        self.texture = texture

if __name__ == '__main__':
    MyCameraApp().run()
