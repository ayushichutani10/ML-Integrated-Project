"""Local web dashboard for the Churn project.
Run:  python app.py      (opens http://localhost:8000)
Uses only the Python standard library + the project's own requirements."""
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import joblib
import pandas as pd
from config import FIG_DIR, MODEL_PATH, TARGET
from data_prep import clean, load_data

PORT = 8000
SOURCES = ["data_prep.py", "train.py", "predict.py", "config.py"]


def ensure_trained():
    if not MODEL_PATH.exists() or not (ROOT / "reports" / "metrics.json").exists():
        print("No trained model found - training now (about a minute)...")
        import train
        train.main()


def build_overview():
    raw = load_data()
    df = clean(raw)
    tc = pd.to_numeric(raw["TotalCharges"], errors="coerce")
    schema = []
    for col in df.drop(columns=TARGET).columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 2:
            schema.append({"name": col, "type": "number", "min": float(s.min()), "max": float(s.max()),
                           "default": round(float(s.median()), 2)})
        else:
            opts = sorted(s.unique().tolist(), key=str)
            schema.append({"name": col, "type": "select", "options": opts,
                           "numeric": bool(pd.api.types.is_numeric_dtype(s)),
                           "default": s.mode()[0].item() if hasattr(s.mode()[0], "item") else s.mode()[0]})
    return {
        "rows": int(raw.shape[0]), "cols": int(raw.shape[1]),
        "churn_rate": float(df[TARGET].mean()),
        "blank_total_charges": int(tc.isna().sum()),
        "raw_head": json.loads(raw.head(8).to_json(orient="records")),
        "clean_head": json.loads(df.head(8).to_json(orient="records")),
        "dtypes": {c: str(t) for c, t in raw.dtypes.items()},
        "schema": schema,
    }


class Handler(BaseHTTPRequestHandler):
    bundle = None
    overview = None

    def _send(self, body, ctype="application/json", code=200):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._send((ROOT / "web" / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/overview":
            return self._send(Handler.overview)
        if path == "/api/metrics":
            return self._send((ROOT / "reports" / "metrics.json").read_bytes())
        if path.startswith("/figures/"):
            f = FIG_DIR / Path(path).name
            if f.exists():
                return self._send(f.read_bytes(), "image/png")
        if path.startswith("/api/source/"):
            name = Path(path).name
            if name in SOURCES:
                return self._send({"name": name, "code": (ROOT / "src" / name).read_text()})
        self._send({"error": "not found"}, code=404)

    def do_POST(self):
        if self.path != "/api/predict":
            return self._send({"error": "not found"}, code=404)
        try:
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            b = Handler.bundle
            row = {}
            for f in Handler.overview["schema"]:
                v = data[f["name"]]
                row[f["name"]] = float(v) if f["type"] == "number" or f.get("numeric") else v
            X = pd.DataFrame([row])[b["columns"]]
            model = b["model"]
            p = float(model.predict_proba(X)[0, 1])
            pre = model.named_steps["prep"]
            vec = pre.transform(X)
            vec = vec.toarray()[0] if hasattr(vec, "toarray") else vec[0]
            contrib = vec * model.named_steps["clf"].coef_[0]
            names = pre.get_feature_names_out()
            top = sorted(zip(names, contrib), key=lambda t: -abs(t[1]))[:6]
            self._send({"probability": p, "churn": p >= 0.5,
                        "factors": [{"feature": n.split("__", 1)[1], "effect": float(c)} for n, c in top]})
        except Exception as e:  # show the error in the UI instead of crashing
            self._send({"error": str(e)}, code=400)

    def log_message(self, *a):
        pass


def main():
    ensure_trained()
    Handler.bundle = joblib.load(MODEL_PATH)
    Handler.overview = build_overview()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"Dashboard running at {url}  (Ctrl+C to stop)")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
