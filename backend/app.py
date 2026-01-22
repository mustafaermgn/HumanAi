from flask import Flask, jsonify
from flask_cors import CORS

from config import Config


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app, resources={r"/api/*": {"origins": app.config.get("CORS_ORIGINS")}})

    @app.route("/api/health", methods=["GET"])
    def health_check():
        return jsonify({"status": "healthy", "message": "API is running"})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
