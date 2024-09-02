from kivy.app import App
from kivy.uix.gridlayout import GridLayout

class Root_Layout(GridLayout):
    pass

class Main_app(App):
    title = "GUI Application"

if __name__ == "__main__":
    Main_app().run()