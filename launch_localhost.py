import contextlib
import functools
import http.server
import socket
import threading
import webbrowser
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8000
WEB_ROOT = Path(__file__).parent / "web"


def _find_free_port(host: str, preferred: int) -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        if sock.connect_ex((host, preferred)) != 0:
            return preferred
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def main() -> None:
    if not WEB_ROOT.exists():
        raise FileNotFoundError(f"Missing web root at {WEB_ROOT}")

    port = _find_free_port(HOST, PORT)
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(WEB_ROOT)
    )

    httpd = http.server.ThreadingHTTPServer((HOST, port), handler)
    httpd.timeout = 1

    url = f"http://{HOST}:{port}"
    print(f"Serving {WEB_ROOT} at {url}")
    print("Open the browser devtools console to see the log.")

    def _serve() -> None:
        with contextlib.suppress(KeyboardInterrupt):
            httpd.serve_forever()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    webbrowser.open(url)

    try:
        while thread.is_alive():
            thread.join(timeout=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        thread.join()


if __name__ == "__main__":
    main()
