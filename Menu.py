from Recognise import Recognise
from SaveFace import SaveFace

class Menu:
    def __init__(self):
        # Ts and Cs
        self.accepted = False
        self.main_menu()

    def main_menu(self):
        action = input("""
           Face Recognition

           1 - Enrolment
           2 - Identification
           
           """)

        if action == "1":
            if not self.accepted:
                input(open("Terms and Conditions.txt", "r").read())
                self.accepted = True
            if self.accepted:
                SaveFace().start()

        if action == "2":
            Recognise().start()


if __name__ == '__main__':
    Menu()
