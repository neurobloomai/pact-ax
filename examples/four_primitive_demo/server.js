/**
 * Browser demo server for the four-primitive + pact-hx demo.
 * Streams demo output live to the browser via SSE.
 *
 * Usage:
 *   ANTHROPIC_API_KEY=sk-ant-... node examples/four_primitive_demo/server.js
 */

const express = require("express");
const { spawn } = require("child_process");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 4000;

app.get("/", (req, res) => {
  res.send(HTML);
});

app.get("/run", (req, res) => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();

  const env = { ...process.env, PYTHONUNBUFFERED: "1" };
  const child = spawn("python", [
    path.join(__dirname, "demo.py"),
  ], { env, cwd: path.join(__dirname, "../..") });

  const send = (data) => {
    const escaped = JSON.stringify(data);
    res.write(`data: ${escaped}\n\n`);
  };

  child.stdout.on("data", (chunk) => send({ type: "line", text: chunk.toString() }));
  child.stderr.on("data", (chunk) => send({ type: "err",  text: chunk.toString() }));
  child.on("close", (code) => {
    send({ type: "done", code });
    res.end();
  });

  req.on("close", () => child.kill());
});

app.listen(PORT, () => {
  console.log(`\nFour-Primitive Demo → http://localhost:${PORT}\n`);
});

// ── HTML ──────────────────────────────────────────────────────────────────────

const HTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PACT-AX · Four-Primitive Demo</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #0d1117;
    color: #e6edf3;
    font-family: "SF Mono", "Fira Code", "Cascadia Code", monospace;
    font-size: 13px;
    line-height: 1.6;
    padding: 32px 24px;
    min-height: 100vh;
  }

  header {
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 28px;
    flex-wrap: wrap;
  }

  h1 {
    font-size: 18px;
    font-weight: 600;
    color: #f0f6fc;
    letter-spacing: 0.3px;
  }

  h1 span { color: #58a6ff; }

  #run-btn {
    padding: 8px 20px;
    background: #238636;
    color: #fff;
    border: none;
    border-radius: 6px;
    font-family: inherit;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.15s;
  }
  #run-btn:hover  { background: #2ea043; }
  #run-btn:disabled { background: #21262d; color: #8b949e; cursor: not-allowed; }

  #status {
    font-size: 12px;
    color: #8b949e;
  }
  #status.running { color: #d29922; }
  #status.done    { color: #3fb950; }
  #status.err     { color: #f85149; }

  #output {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 20px 24px;
    white-space: pre-wrap;
    word-break: break-word;
    max-width: 900px;
    min-height: 200px;
    overflow-y: auto;
    tab-size: 2;
  }

  /* colour map */
  .s-header  { color: #58a6ff; font-weight: 600; }
  .s-step    { color: #d2a8ff; font-weight: 600; }
  .s-ok      { color: #3fb950; }
  .s-warn    { color: #d29922; }
  .s-box     { color: #79c0ff; }
  .s-label   { color: #ffa657; }
  .s-summary { color: #f0f6fc; font-weight: 600; }
  .s-err     { color: #f85149; }
</style>
</head>
<body>

<header>
  <h1>PACT-AX · <span>Five-Primitive</span> Integration Demo</h1>
  <button id="run-btn" onclick="runDemo()">▶ Run Demo</button>
  <span id="status">ready</span>
</header>

<div id="output">Click "Run Demo" to start…</div>

<script>
const out    = document.getElementById("output");
const btn    = document.getElementById("run-btn");
const status = document.getElementById("status");

function colorize(text) {
  return text
    .replace(/^(═+.+═+)$/gm,     '<span class="s-header">$1</span>')
    .replace(/^(─+.+─+)$/gm,     '<span class="s-step">$1</span>')
    .replace(/^(  Step \\d[^\\n]+)$/gm, '<span class="s-step">$1</span>')
    .replace(/(✓)/g,              '<span class="s-ok">✓</span>')
    .replace(/(✗)/g,              '<span class="s-err">✗</span>')
    .replace(/(↑[^\\n]+)/g,       '<span class="s-ok">$1</span>')
    .replace(/(↓[^\\n]+)/g,       '<span class="s-warn">$1</span>')
    .replace(/^(  [┌└│].+)$/gm,   '<span class="s-box">$1</span>')
    .replace(/(  ▸ [A-Z].+)/g,    '<span class="s-label">$1</span>')
    .replace(/^(  Summary[^\\n]+)$/gm, '<span class="s-summary">$1</span>');
}

function append(text, cls) {
  const span = document.createElement("span");
  if (cls) span.className = cls;
  span.innerHTML = cls ? text : colorize(text);
  out.appendChild(span);
  out.scrollTop = out.scrollHeight;
}

function runDemo() {
  out.innerHTML = "";
  btn.disabled = true;
  status.textContent = "running…";
  status.className = "running";

  const es = new EventSource("/run");

  es.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === "line") {
      append(msg.text);
    } else if (msg.type === "err") {
      append(msg.text, "s-err");
    } else if (msg.type === "done") {
      es.close();
      btn.disabled = false;
      if (msg.code === 0) {
        status.textContent = "done ✓";
        status.className = "done";
      } else {
        status.textContent = "exited with code " + msg.code;
        status.className = "err";
      }
    }
  };

  es.onerror = () => {
    es.close();
    btn.disabled = false;
    status.textContent = "connection error";
    status.className = "err";
  };
}
</script>
</body>
</html>`;
