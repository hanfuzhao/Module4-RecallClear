"""RecallClear web application.

Runs the fine-tuned model that rewrites official NHTSA vehicle-recall notices
into plain language a car owner can act on. Start it with:

    python main.py                # http://127.0.0.1:7860
    gunicorn --bind 0.0.0.0:7860 main:app

Routes
------
GET  /             the single-page interface
GET  /api/health   readiness probe (also reports whether weights are loaded)
GET  /api/examples curated held-out notices for one-click demos
POST /api/lookup   fetch a live recall by NHTSA campaign number
POST /api/explain  rewrite a notice, optionally alongside the untuned baseline
"""

from __future__ import annotations

import argparse
import logging
import os

from flask import Flask, jsonify, render_template, request

from scripts import config
from scripts.app_service import DemoLibrary, ExplainerService, load_evaluation_summary
from scripts.recall_lookup import (
    CampaignNotFoundError,
    fetch_campaign,
    is_lookup_enabled,
)

MAX_NOTICE_CHARACTERS = 6000

def configure_logging() -> logging.Logger:
    """Configure application logging and return the app logger.

    Done at import rather than in ``main()`` because a WSGI server such as
    gunicorn imports ``app`` directly and never calls ``main()``.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger("recallclear")


logger = configure_logging()

app = Flask(__name__)
explainer_service = ExplainerService()
demo_library = DemoLibrary()


@app.route("/")
def index() -> str:
    """Render the single-page interface."""
    return render_template(
        "index.html",
        examples=demo_library.examples(),
        evaluation=load_evaluation_summary(),
        base_model=config.BASE_MODEL_ID,
        lookup_enabled=is_lookup_enabled(),
    )


@app.route("/api/health")
def health() -> tuple:
    """Report service readiness without forcing the model to load."""
    return jsonify(
        {
            "status": "ok",
            "model_loaded": explainer_service.is_loaded,
            "base_model": config.BASE_MODEL_ID,
            "adapter": explainer_service.adapter_source,
        }
    ), 200


@app.route("/api/examples")
def examples() -> tuple:
    """Return the curated demo notices."""
    return jsonify({"examples": demo_library.examples()}), 200


@app.route("/api/lookup", methods=["POST"])
def lookup() -> tuple:
    """Fetch a live recall notice from NHTSA by campaign number."""
    if not is_lookup_enabled():
        return jsonify({"error": "Live lookup is disabled in this deployment."}), 503

    payload = request.get_json(silent=True) or {}
    campaign_number = (payload.get("campaign_number") or "").strip()
    if not campaign_number:
        return jsonify({"error": "Enter a recall campaign number, for example 23V123000."}), 400

    try:
        record = fetch_campaign(campaign_number)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except CampaignNotFoundError:
        return jsonify({"error": f"NHTSA has no recall numbered {campaign_number}."}), 404
    except Exception as error:  # upstream outage or network failure
        logger.warning("NHTSA lookup failed: %s", error)
        return jsonify({"error": "Could not reach the NHTSA recalls service. Try again shortly."}), 502

    notice = ExplainerService.notice_from_record(record)
    return jsonify(
        {
            "record": record,
            "notice": notice,
            "notice_metrics": ExplainerService.notice_metrics(notice),
            "official_warnings": ExplainerService.official_warnings(record),
        }
    ), 200


@app.route("/api/explain", methods=["POST"])
def explain() -> tuple:
    """Rewrite a notice into a plain-language card."""
    payload = request.get_json(silent=True) or {}
    notice = (payload.get("notice") or "").strip()
    include_baseline = bool(payload.get("include_baseline"))

    if not notice:
        return jsonify({"error": "Paste a recall notice first."}), 400
    if len(notice) > MAX_NOTICE_CHARACTERS:
        return jsonify(
            {"error": f"That notice is longer than {MAX_NOTICE_CHARACTERS} characters."}
        ), 413

    try:
        result = {
            "notice_metrics": ExplainerService.notice_metrics(notice),
            "tuned": explainer_service.explain(notice),
        }
        if include_baseline:
            result["base"] = explainer_service.explain(
                notice, mode=explainer_service.explainer().MODE_BASE
            )
    except RuntimeError as error:  # adapter missing
        logger.error("Generation failed: %s", error)
        return jsonify({"error": str(error)}), 503

    return jsonify(result), 200


@app.errorhandler(404)
def not_found(_error) -> tuple:
    """Return JSON for unknown API paths and the app shell for anything else."""
    if request.path.startswith("/api/"):
        return jsonify({"error": "Unknown endpoint."}), 404
    return render_template(
        "index.html",
        examples=demo_library.examples(),
        evaluation=load_evaluation_summary(),
        base_model=config.BASE_MODEL_ID,
        lookup_enabled=is_lookup_enabled(),
    ), 404


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for local development runs."""
    parser = argparse.ArgumentParser(description="Run the RecallClear web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load the model at start-up instead of on the first request.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point for local development."""
    arguments = parse_arguments()
    if arguments.preload:
        logger.info("Loading model ...")
        explainer_service.explainer()
    app.run(host=arguments.host, port=arguments.port, debug=arguments.debug, threaded=True)


if __name__ == "__main__":
    main()
