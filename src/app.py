"""Flask API for PM-USP CSSS scholarship assistant.
Run: python -m src.app
"""
import sys
from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from src import rag, eligibility

app = Flask(__name__, static_folder=None)


@app.route("/")
def index():
    """Serve the web interface."""
    templates_dir = Path(__file__).parent / "templates"
    return send_from_directory(templates_dir, "index.html")


@app.route("/health")
def health():
    """Health check endpoint."""
    try:
        rag.load_index()
        index_loaded = True
    except:
        index_loaded = False

    return jsonify({
        "status": "ok",
        "index_loaded": index_loaded
    })


@app.route("/ask", methods=["POST"])
def ask():
    """Answer a question using RAG."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        question = data.get("question", "").strip()

        if not question:
            return jsonify({"error": "Question is required"}), 400

        if len(question) > 500:
            return jsonify({"error": "Question too long (max 500 characters)"}), 400

        try:
            result = rag.answer(question)
            return jsonify(result)
        except RuntimeError as e:
            if "Could not connect" in str(e):
                return jsonify({"error": str(e)}), 503
            raise

    except Exception as e:
        return jsonify({"error": f"Internal error: {str(e)}"}), 500


@app.route("/eligibility", methods=["POST"])
def check_eligibility():
    """Check eligibility for PM-USP CSSS scholarship."""
    try:
        profile = request.get_json()
        if not profile:
            return jsonify({"error": "Request body must be JSON"}), 400

        result = eligibility.check_eligibility(profile)

        if "error" in result:
            return jsonify(result), 400

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Internal error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
