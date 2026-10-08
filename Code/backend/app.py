"""Flask backend for the Version A historical-data hackathon demo.

Layering: routes -> services/*.py -> scored CSVs. Routes contain no pandas.
The future frontend only needs the JSON APIs below; Code/frontend holds
the replaceable UI files.
"""

from pathlib import Path

from data.building_registry import BUILDINGS
from flask import Flask, jsonify, redirect, request, send_from_directory
from services import analytics_service, building_service

app = Flask(__name__)

CODE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = CODE_DIR.parent / "frontend"


@app.errorhandler(404)
def _not_found(_err):
    return jsonify({"error": "not found"}), 404


@app.errorhandler(400)
def _bad_request(err):
    return jsonify({"error": str(err.description or "bad request")}), 400


def _require_building(building_id: str) -> None:
    from werkzeug.exceptions import NotFound

    if building_id not in BUILDINGS:
        raise NotFound(f"unknown building: {building_id}")


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/building/<building_id>")
def building_page(building_id: str):
    _require_building(building_id)
    return send_from_directory(FRONTEND_DIR, f"{building_id}.html")


@app.route("/<page>.html")
def legacy_page(page: str):
    """Compatibility for in-page prev/next links (mess.html, c-block.html…)."""
    _require_building(page)
    return redirect(f"/building/{page}")


@app.route("/about")
def about():
    """Full interactive About page (replaces the old header modal)."""
    return send_from_directory(FRONTEND_DIR, "about.html")


@app.route("/api/buildings")
def api_buildings():
    return jsonify(building_service.get_campus_overview())


@app.route("/api/buildings/<building_id>")
def api_building(building_id: str):
    _require_building(building_id)
    return jsonify(building_service.get_overview(building_id))


@app.route("/api/buildings/<building_id>/trend")
def api_trend(building_id: str):
    _require_building(building_id)
    span = request.args.get("range", "7d")
    try:
        return jsonify(analytics_service.get_trend(building_id, span))
    except ValueError as exc:
        from werkzeug.exceptions import BadRequest

        raise BadRequest(str(exc)) from exc


@app.route("/api/buildings/<building_id>/anomalies")
def api_anomalies(building_id: str):
    _require_building(building_id)
    try:
        return jsonify(
            analytics_service.get_anomalies(
                building_id,
                limit=request.args.get("limit", 20),
                offset=request.args.get("offset", 0),
                severity=request.args.get("severity", "p98"),
                start=request.args.get("start"),
                end=request.args.get("end"),
            )
        )
    except ValueError as exc:
        from werkzeug.exceptions import BadRequest

        raise BadRequest(str(exc)) from exc


@app.route("/api/buildings/<building_id>/insights")
def api_insights(building_id: str):
    _require_building(building_id)
    return jsonify(analytics_service.get_insights(building_id))


if __name__ == "__main__":
    app.run(debug=True)
