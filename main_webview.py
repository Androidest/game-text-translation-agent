import webview
import threading
import requests
import time
from main_gradio import demo

def wait_for_server(url, timeout=30):
    for _ in range(timeout * 2):
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return True
        except:
            time.sleep(0.5)
    return False

if __name__ == "__main__":
    SERVER_PORT = 7860
    URL = f"http://127.0.0.1:{SERVER_PORT}"

    thread = threading.Thread(target=lambda: demo.launch(server_name="127.0.0.1", server_port=SERVER_PORT), daemon=False)
    thread.start()
    
    if not wait_for_server(URL):
        print("Gradio server failed to start")
        exit(1)
    
    webview.create_window(
        title="My Gradio App",
        url=URL,
        width=1200,
        height=800,
    )
    webview.start()