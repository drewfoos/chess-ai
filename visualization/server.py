"""Flask backend for the training visualization dashboard.

Serves metrics JSON files and static frontend assets.

Usage:
    python -m visualization.server --metrics-dir selfplay_output/metrics --port 5000
"""

import argparse
import json
import os
import subprocess
import threading

from flask import Flask, jsonify, send_from_directory, request


def create_app(metrics_dir: str) -> Flask:
    """Create Flask app configured to serve metrics from the given directory."""
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    app = Flask(__name__, static_folder=static_dir)

    # Engine process management
    engine_process = None
    engine_lock = threading.Lock()

    def get_engine():
        """Get or create the UCI engine subprocess."""
        nonlocal engine_process
        if engine_process is None or engine_process.poll() is not None:
            engine_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'build', 'Release', 'chess_engine.exe')
            if not os.path.exists(engine_path):
                return None
            engine_process = subprocess.Popen(
                [engine_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            # Initialize UCI
            engine_process.stdin.write("uci\n")
            engine_process.stdin.flush()
            while True:
                line = engine_process.stdout.readline().strip()
                if line == "uciok":
                    break
            engine_process.stdin.write("isready\n")
            engine_process.stdin.flush()
            while True:
                line = engine_process.stdout.readline().strip()
                if line == "readyok":
                    break
        return engine_process

    @app.route('/')
    def index():
        return send_from_directory(static_dir, 'index.html')

    @app.route('/play')
    def play():
        return send_from_directory(static_dir, 'play.html')

    @app.route('/api/play/move', methods=['POST'])
    def play_move():
        """Send position to engine and get best move back."""
        data = request.get_json()
        moves = data.get('moves', [])
        think_time = data.get('think_time', 1000)

        with engine_lock:
            engine = get_engine()
            if engine is None:
                return jsonify({
                    'error': 'Engine not available. Build with: cmake --build build --config Release'
                }), 503

            # Send position
            if moves:
                pos_cmd = f"position startpos moves {' '.join(moves)}\n"
            else:
                pos_cmd = "position startpos\n"

            engine.stdin.write(pos_cmd)
            engine.stdin.flush()

            # Search
            engine.stdin.write(f"go movetime {think_time}\n")
            engine.stdin.flush()

            # Read response
            info_lines = []
            best_move = None
            score_cp = None
            nodes = 0
            nps = 0

            while True:
                line = engine.stdout.readline().strip()
                if line.startswith("bestmove"):
                    best_move = line.split()[1]
                    break
                elif line.startswith("info"):
                    info_lines.append(line)
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "score" and i + 2 < len(parts) and parts[i + 1] == "cp":
                            score_cp = int(parts[i + 2])
                        elif part == "score" and i + 2 < len(parts) and parts[i + 1] == "mate":
                            try:
                                score_cp = None  # handled as mate
                                mate_in = int(parts[i + 2])
                            except ValueError:
                                mate_in = None
                        elif part == "nodes" and i + 1 < len(parts):
                            try:
                                nodes = int(parts[i + 1])
                            except ValueError:
                                pass
                        elif part == "nps" and i + 1 < len(parts):
                            try:
                                nps = int(parts[i + 1])
                            except ValueError:
                                pass

            # Check for mate score
            mate_val = None
            for line in info_lines:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "score" and i + 2 < len(parts) and parts[i + 1] == "mate":
                        try:
                            mate_val = int(parts[i + 2])
                        except ValueError:
                            pass

            result = {
                'bestmove': best_move,
                'score_cp': score_cp,
                'nodes': nodes,
                'nps': nps,
            }
            if mate_val is not None:
                result['mate'] = mate_val
            return jsonify(result)

    @app.route('/api/play/new', methods=['POST'])
    def new_game():
        """Reset engine for a new game."""
        with engine_lock:
            engine = get_engine()
            if engine:
                engine.stdin.write("ucinewgame\n")
                engine.stdin.flush()
                engine.stdin.write("isready\n")
                engine.stdin.flush()
                while True:
                    line = engine.stdout.readline().strip()
                    if line == "readyok":
                        break
        return jsonify({'status': 'ok'})

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
