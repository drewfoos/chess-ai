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
import time

from flask import Flask, jsonify, send_from_directory, request


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_PATH = os.path.join(REPO_ROOT, 'build', 'Release', 'chess_engine.exe')
STOCKFISH_PATH = os.path.join(
    REPO_ROOT, 'engines', 'stockfish', 'stockfish-windows-x86-64-avx2.exe')


def _engine_env() -> dict:
    """Return a copy of os.environ with LibTorch / TensorRT bin dirs prepended
    to PATH so chess_engine.exe (a subprocess) can resolve its native DLLs.

    chess_mcts registers these via os.add_dll_directory for this Python
    process, but that's not inherited by child processes — they need PATH.
    """
    env = os.environ.copy()
    extras: list[str] = []
    try:
        from chess_mcts import _paths as _p  # type: ignore
        for d in (getattr(_p, 'LIBTORCH_LIB', ''), getattr(_p, 'TENSORRT_BIN', '')):
            if d and os.path.isdir(d):
                extras.append(d)
    except Exception:
        pass
    trt_root = os.environ.get('TENSORRT_PATH')
    if trt_root:
        trt_bin = os.path.join(trt_root, 'bin')
        if os.path.isdir(trt_bin) and trt_bin not in extras:
            extras.append(trt_bin)
    if extras:
        env['PATH'] = os.pathsep.join(extras) + os.pathsep + env.get('PATH', '')
    return env


def _scan_runs():
    """Find all model runs with metrics. Returns list of
    {id, name, path, total_generations, mtime} sorted newest-first.
    """
    out = []
    roots = [
        os.path.join(REPO_ROOT, 'models'),
        os.path.join(REPO_ROOT, 'selfplay_output'),
    ]
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            run_dir = os.path.join(root, name)
            metrics_dir = os.path.join(run_dir, 'metrics')
            summary_path = os.path.join(metrics_dir, 'summary.json')
            if not os.path.isfile(summary_path):
                # also handle the flat `selfplay_output/metrics` layout
                alt = os.path.join(run_dir, 'summary.json')
                if os.path.isfile(alt):
                    metrics_dir = run_dir
                    summary_path = alt
                else:
                    continue
            try:
                with open(summary_path) as f:
                    s = json.load(f)
                total_gens = s.get('total_generations', len(s.get('generations', [])))
            except Exception:
                total_gens = 0
            rel = os.path.relpath(metrics_dir, REPO_ROOT).replace('\\', '/')
            out.append({
                'id': rel,
                'name': name,
                'path': metrics_dir,
                'total_generations': total_gens,
                'mtime': os.path.getmtime(summary_path),
            })
    out.sort(key=lambda r: -r['mtime'])
    return out


def _resolve_default_metrics_dir(requested: str) -> str:
    """If the user didn't override --metrics-dir (or the requested dir is empty),
    auto-pick the newest run that actually has data. Falls back to `requested`.
    """
    abs_req = requested if os.path.isabs(requested) else os.path.join(REPO_ROOT, requested)
    summary = os.path.join(abs_req, 'summary.json')
    if os.path.isfile(summary):
        return abs_req
    runs = _scan_runs()
    if runs:
        return runs[0]['path']
    return abs_req


def _scan_models():
    """Return a list of discoverable model files under models/.

    Each entry is {'id': relative_path, 'type': 'trt'|'pt', 'run': run_name}.
    """
    models_dir = os.path.join(REPO_ROOT, 'models')
    if not os.path.isdir(models_dir):
        return []
    out = []
    for root, _dirs, files in os.walk(models_dir):
        for f in files:
            lower = f.lower()
            if not (lower.endswith('.pt') or lower.endswith('.trt')):
                continue
            full = os.path.join(root, f)
            rel = os.path.relpath(full, REPO_ROOT).replace('\\', '/')
            # Skip raw state_dict checkpoints — the engine needs TorchScript.
            # Training saves per-generation checkpoints under `checkpoints/`;
            # exported TorchScript / TRT models sit at the run root.
            if '/checkpoints/' in rel + '/':
                continue
            parts = rel.split('/')
            run = parts[1] if len(parts) > 2 else ''
            out.append({
                'id': rel,
                'type': 'trt' if lower.endswith('.trt') else 'pt',
                'run': run,
                'name': f,
            })
    # TRT first (faster), then by path
    out.sort(key=lambda m: (m['type'] != 'trt', m['id']))
    return out


def _spawn_engine(model_path: str | None):
    """Launch chess_engine.exe in UCI mode, optionally loading a model.

    Retries once on handshake failure (can happen when training rewrites the
    .trt mid-spawn). On final failure, includes engine stderr in the message.
    """
    if not os.path.exists(ENGINE_PATH):
        raise RuntimeError(f"Engine not built: {ENGINE_PATH}")

    if model_path is None or model_path == '' or model_path == 'random':
        cmd = [ENGINE_PATH]
    else:
        abs_model = model_path
        if not os.path.isabs(abs_model):
            abs_model = os.path.join(REPO_ROOT, abs_model)
        if not os.path.isfile(abs_model):
            raise RuntimeError(f"Model file not found: {abs_model}")
        if abs_model.lower().endswith('.trt'):
            cmd = [ENGINE_PATH, 'uci_trt', abs_model]
        else:
            cmd = [ENGINE_PATH, 'uci', abs_model, 'cuda']

    def _attempt():
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=_engine_env(),
        )
        try:
            proc.stdin.write("uci\n")
            proc.stdin.flush()
            while True:
                line = proc.stdout.readline()
                if not line:
                    err = (proc.stderr.read() or '').strip()
                    raise RuntimeError(
                        "Engine terminated during UCI handshake"
                        + (f": {err}" if err else "")
                    )
                if line.strip() == "uciok":
                    break
            proc.stdin.write("isready\n")
            proc.stdin.flush()
            while True:
                line = proc.stdout.readline()
                if not line:
                    err = (proc.stderr.read() or '').strip()
                    raise RuntimeError(
                        "Engine terminated during readyok handshake"
                        + (f": {err}" if err else "")
                    )
                if line.strip() == "readyok":
                    break
            return proc
        except Exception:
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
            raise

    try:
        return _attempt()
    except RuntimeError:
        time.sleep(0.5)
        return _attempt()


def _spawn_stockfish():
    if not os.path.exists(STOCKFISH_PATH):
        raise RuntimeError(f"Stockfish not found: {STOCKFISH_PATH}")
    proc = subprocess.Popen(
        [STOCKFISH_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=_engine_env(),
    )
    proc.stdin.write("uci\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("Stockfish terminated during UCI handshake")
        if line.strip() == "uciok":
            break
    proc.stdin.write("isready\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("Stockfish terminated during readyok handshake")
        if line.strip() == "readyok":
            break
    return proc


def _parse_info_scores(lines):
    """Extract the last-seen cp / mate / nodes / nps / depth from a list of info lines."""
    score_cp = None
    mate = None
    nodes = 0
    nps = 0
    depth = 0
    for line in lines:
        parts = line.split()
        for i, part in enumerate(parts):
            if part == "score" and i + 2 < len(parts):
                if parts[i + 1] == "cp":
                    try:
                        score_cp = int(parts[i + 2])
                        mate = None
                    except ValueError:
                        pass
                elif parts[i + 1] == "mate":
                    try:
                        mate = int(parts[i + 2])
                        score_cp = None
                    except ValueError:
                        pass
            elif part == "nodes" and i + 1 < len(parts):
                try: nodes = int(parts[i + 1])
                except ValueError: pass
            elif part == "nps" and i + 1 < len(parts):
                try: nps = int(parts[i + 1])
                except ValueError: pass
            elif part == "depth" and i + 1 < len(parts):
                try: depth = int(parts[i + 1])
                except ValueError: pass
    return score_cp, mate, nodes, nps, depth


def create_app(metrics_dir: str) -> Flask:
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    app = Flask(__name__, static_folder=static_dir)

    # Mutable wrapper so /api/runs/select can swap the active directory
    # without restarting the server.
    state = {'metrics_dir': metrics_dir}

    # Our engine (swappable by model)
    engine_proc = {'proc': None, 'model': None}
    engine_lock = threading.Lock()
    # Stockfish (persistent)
    sf_proc = {'proc': None}
    sf_lock = threading.Lock()

    def ensure_engine(model_path: str | None):
        """Ensure our engine is running with the requested model. Returns proc."""
        current = engine_proc['proc']
        current_model = engine_proc['model']
        needs_restart = (
            current is None
            or current.poll() is not None
            or (model_path or '') != (current_model or '')
        )
        if needs_restart:
            if current is not None and current.poll() is None:
                try:
                    current.stdin.write("quit\n")
                    current.stdin.flush()
                    current.wait(timeout=2)
                except Exception:
                    current.kill()
            engine_proc['proc'] = _spawn_engine(model_path)
            engine_proc['model'] = model_path or ''
        return engine_proc['proc']

    def ensure_stockfish():
        p = sf_proc['proc']
        if p is None or p.poll() is not None:
            sf_proc['proc'] = _spawn_stockfish()
        return sf_proc['proc']

    @app.route('/')
    def index():
        return send_from_directory(static_dir, 'index.html')

    @app.route('/play')
    def play():
        return send_from_directory(static_dir, 'play.html')

    @app.route('/api/models')
    def list_models():
        return jsonify({
            'models': _scan_models(),
            'stockfish_available': os.path.isfile(STOCKFISH_PATH),
        })

    @app.route('/api/play/move', methods=['POST'])
    def play_move():
        data = request.get_json() or {}
        moves = data.get('moves', [])
        model = data.get('model')  # None or 'random' → no model
        # Time control: either a clock dict or a fixed think time.
        tc = data.get('tc')  # {wtime, btime, winc, binc} in ms

        if tc and isinstance(tc, dict):
            go_cmd = "go"
            for key in ('wtime', 'btime', 'winc', 'binc', 'movestogo'):
                if key in tc and tc[key] is not None:
                    go_cmd += f" {key} {int(tc[key])}"
        else:
            think_time = int(data.get('think_time', 1000))
            go_cmd = f"go movetime {think_time}"

        with engine_lock:
            try:
                engine = ensure_engine(model)
            except RuntimeError as e:
                return jsonify({'error': str(e)}), 503

            pos_cmd = ("position startpos moves " + ' '.join(moves) + "\n"
                       if moves else "position startpos\n")
            engine.stdin.write(pos_cmd)
            engine.stdin.flush()
            engine.stdin.write(go_cmd + "\n")
            engine.stdin.flush()

            info_lines = []
            best_move = None
            while True:
                line = engine.stdout.readline()
                if not line:
                    return jsonify({'error': 'Engine closed unexpectedly'}), 503
                line = line.strip()
                if line.startswith("bestmove"):
                    best_move = line.split()[1]
                    break
                if line.startswith("info"):
                    info_lines.append(line)

            score_cp, mate, nodes, nps, depth = _parse_info_scores(info_lines)
            result = {
                'bestmove': best_move,
                'score_cp': score_cp,
                'nodes': nodes,
                'nps': nps,
                'depth': depth,
                'model': engine_proc['model'],
            }
            if mate is not None:
                result['mate'] = mate
            return jsonify(result)

    @app.route('/api/play/new', methods=['POST'])
    def new_game():
        data = request.get_json() or {}
        model = data.get('model')
        with engine_lock:
            try:
                engine = ensure_engine(model)
            except RuntimeError as e:
                return jsonify({'error': str(e)}), 503
            engine.stdin.write("ucinewgame\n")
            engine.stdin.flush()
            engine.stdin.write("isready\n")
            engine.stdin.flush()
            while True:
                line = engine.stdout.readline()
                if not line:
                    return jsonify({'error': 'Engine closed unexpectedly'}), 503
                if line.strip() == "readyok":
                    break
        return jsonify({'status': 'ok', 'model': engine_proc['model']})

    @app.route('/api/play/stockfish_eval', methods=['POST'])
    def stockfish_eval():
        """Return Stockfish's eval of the current position (from side-to-move POV)."""
        data = request.get_json() or {}
        moves = data.get('moves', [])
        depth = int(data.get('depth', 14))

        with sf_lock:
            try:
                sf = ensure_stockfish()
            except RuntimeError as e:
                return jsonify({'error': str(e)}), 503

            pos_cmd = ("position startpos moves " + ' '.join(moves) + "\n"
                       if moves else "position startpos\n")
            sf.stdin.write(pos_cmd)
            sf.stdin.flush()
            sf.stdin.write(f"go depth {depth}\n")
            sf.stdin.flush()

            info_lines = []
            best_move = None
            while True:
                line = sf.stdout.readline()
                if not line:
                    return jsonify({'error': 'Stockfish closed unexpectedly'}), 503
                line = line.strip()
                if line.startswith("bestmove"):
                    best_move = line.split()[1] if len(line.split()) > 1 else None
                    break
                if line.startswith("info"):
                    info_lines.append(line)

            score_cp, mate, _nodes, _nps, d = _parse_info_scores(info_lines)
            # Stockfish reports from side-to-move. Flip to white POV for consistency
            # with our engine's eval bar convention.
            white_to_move = (len(moves) % 2 == 0)
            if not white_to_move:
                if score_cp is not None:
                    score_cp = -score_cp
                if mate is not None:
                    mate = -mate

            out = {
                'score_cp': score_cp,
                'depth': d,
                'bestmove': best_move,
            }
            if mate is not None:
                out['mate'] = mate
            return jsonify(out)

    @app.route('/api/status')
    def status():
        return jsonify({'status': 'ok', 'metrics_dir': state['metrics_dir']})

    @app.route('/api/runs')
    def list_runs():
        runs = _scan_runs()
        current = os.path.normpath(state['metrics_dir'])
        for r in runs:
            r['current'] = os.path.normpath(r['path']) == current
            r.pop('mtime', None)
        return jsonify({'runs': runs, 'current': current.replace('\\', '/')})

    @app.route('/api/runs/select', methods=['POST'])
    def select_run():
        data = request.get_json() or {}
        run_id = data.get('id')
        if not run_id:
            return jsonify({'error': 'missing run id'}), 400
        candidate = run_id if os.path.isabs(run_id) else os.path.join(REPO_ROOT, run_id)
        if not os.path.isfile(os.path.join(candidate, 'summary.json')):
            return jsonify({'error': f'no summary.json under {run_id}'}), 404
        state['metrics_dir'] = candidate
        return jsonify({'status': 'ok', 'metrics_dir': candidate.replace('\\', '/')})

    @app.route('/api/summary')
    def summary():
        summary_path = os.path.join(state['metrics_dir'], 'summary.json')
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
        gen_path = os.path.join(state['metrics_dir'], f'gen_{gen:03d}.json')
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

    resolved = _resolve_default_metrics_dir(args.metrics_dir)
    if os.path.normpath(resolved) != os.path.normpath(
            args.metrics_dir if os.path.isabs(args.metrics_dir)
            else os.path.join(REPO_ROOT, args.metrics_dir)):
        print(f"(auto-resolved --metrics-dir {args.metrics_dir!r} → {resolved})")
    app = create_app(resolved)
    print(f"Dashboard: http://{args.host}:{args.port}")
    print(f"Metrics dir: {resolved}")
    app.run(host=args.host, port=args.port, debug=True)
