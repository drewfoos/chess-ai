"""Flask backend for the training visualization dashboard.

Serves metrics JSON files and static frontend assets.

Usage:
    python -m visualization.server --metrics-dir selfplay_output/metrics --port 5000
"""

import argparse
import json
import os

from flask import Flask, jsonify, send_from_directory


def create_app(metrics_dir: str) -> Flask:
    """Create Flask app configured to serve metrics from the given directory."""
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    app = Flask(__name__, static_folder=static_dir)

    @app.route('/')
    def index():
        return send_from_directory(static_dir, 'index.html')

    @app.route('/api/status')
    def status():
        return jsonify({'status': 'ok', 'metrics_dir': metrics_dir})

    @app.route('/api/summary')
    def summary():
        summary_path = os.path.join(metrics_dir, 'summary.json')
        if not os.path.exists(summary_path):
            return jsonify({
                'total_generations': 0,
                'total_games': 0,
                'total_positions': 0,
                'total_time_s': 0,
                'generations': [],
            })
        with open(summary_path) as f:
            return jsonify(json.load(f))

    @app.route('/api/generation/<int:gen>')
    def generation(gen: int):
        gen_path = os.path.join(metrics_dir, f'gen_{gen:03d}.json')
        if not os.path.exists(gen_path):
            return jsonify({'error': f'Generation {gen} not found'}), 404
        with open(gen_path) as f:
            return jsonify(json.load(f))

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training visualization server')
    parser.add_argument('--metrics-dir', type=str, default='selfplay_output/metrics',
                        help='Path to metrics directory')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', type=str, default='127.0.0.1')
    args = parser.parse_args()

    app = create_app(args.metrics_dir)
    print(f"Dashboard: http://{args.host}:{args.port}")
    print(f"Metrics dir: {args.metrics_dir}")
    app.run(host=args.host, port=args.port, debug=True)
