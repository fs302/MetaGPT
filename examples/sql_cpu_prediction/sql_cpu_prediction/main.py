## main.py
from server import Server

class Main:
    @staticmethod
    def run_server():
        server = Server()
        server.start()

if __name__ == "__main__":
    Main.run_server()
