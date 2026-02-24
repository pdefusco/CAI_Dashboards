from app import app
import os

if __name__ == "__main__":
    app.run_server(
        host="127.0.0.1",
        port=int(os.environ.get("CDSW_APP_PORT")),
        debug=True
    )
