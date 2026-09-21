import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";

const windows = process.platform === "win32";
const python =
  process.env.VISCANER_PYTHON ||
  path.resolve(windows ? ".venv/Scripts/python.exe" : ".venv/bin/python");
if (!existsSync(python)) {
  console.error(
    "Создайте .venv и установите backend/requirements.txt. Инструкция: README.md",
  );
  process.exit(1);
}
const api = spawn(
  python,
  [
    "-m",
    "uvicorn",
    "backend.main:app",
    "--host",
    "127.0.0.1",
    "--port",
    "8000",
    "--reload",
  ],
  { stdio: "inherit" },
);
const web = spawn(
  process.execPath,
  ["node_modules/vite/bin/vite.js", "--host", "127.0.0.1"],
  { stdio: "inherit" },
);
let stopping = false;
function stop() {
  if (stopping) return;
  stopping = true;
  for (const child of [api, web]) {
    if (!child.pid) continue;
    if (windows) {
      // Python's Windows venv launcher and uvicorn reload create descendants.
      // Stop only the process trees owned by this launcher.
      spawn("taskkill", ["/PID", String(child.pid), "/T", "/F"], {
        stdio: "ignore",
        windowsHide: true,
      });
    } else {
      child.kill("SIGTERM");
    }
  }
}
for (const child of [api, web]) {
  child.on("error", (e) => {
    console.error(e.message);
    stop();
    process.exitCode = 1;
  });
  child.on("exit", (code) => {
    if (!stopping) {
      process.exitCode = code || 0;
      stop();
    }
  });
}
process.on("SIGINT", stop);
process.on("SIGTERM", stop);
