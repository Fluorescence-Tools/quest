const { app, BrowserWindow, dialog } = require("electron");
const { spawn, spawnSync } = require("node:child_process");
const fs = require("node:fs");
const http = require("node:http");
const net = require("node:net");
const os = require("node:os");
const path = require("node:path");

const children = new Set();
let mainWindow = null;
let frontendPortCache = null;

function repoRoot() {
  return path.resolve(__dirname, "..", "..", "..");
}

function resourceRoot() {
  return app.isPackaged ? process.resourcesPath : repoRoot();
}

function packagedPath(...parts) {
  return path.join(resourceRoot(), ...parts);
}

function sourcePath(...parts) {
  return path.join(repoRoot(), ...parts);
}

function getFrontendServerPath() {
  return app.isPackaged
    ? packagedPath("frontend", "standalone", "server.js")
    : sourcePath("webui", "frontend", ".next", "standalone", "server.js");
}

function getBackendCwd() {
  return app.isPackaged ? packagedPath("python") : repoRoot();
}

function getBundledPythonExecutable() {
  const platformDir = process.platform === "darwin" ? `macos-${process.arch}` : `${process.platform}-${process.arch}`;
  const envRoot = app.isPackaged
    ? packagedPath("python-env", platformDir)
    : sourcePath("webui", "desktop", "python-env", platformDir);
  const candidate =
    process.platform === "win32" ? path.join(envRoot, "python.exe") : path.join(envRoot, "bin", "python");
  return fs.existsSync(candidate) ? candidate : null;
}

function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      const port = typeof address === "object" && address ? address.port : 0;
      server.close(() => resolve(port));
    });
  });
}

function waitForHttp(url, timeoutMs = 45000) {
  const startedAt = Date.now();
  return new Promise((resolve, reject) => {
    const attempt = () => {
      const request = http.get(url, (response) => {
        response.resume();
        if (response.statusCode && response.statusCode < 500) {
          resolve();
          return;
        }
        retry();
      });
      request.on("error", retry);
      request.setTimeout(1500, () => {
        request.destroy();
        retry();
      });
    };
    const retry = () => {
      if (Date.now() - startedAt > timeoutMs) {
        reject(new Error(`Timed out waiting for ${url}`));
        return;
      }
      setTimeout(attempt, 350);
    };
    attempt();
  });
}

function pipeChildLogs(child, name) {
  child.stdout?.on("data", (chunk) => process.stdout.write(`[${name}] ${chunk}`));
  child.stderr?.on("data", (chunk) => process.stderr.write(`[${name}] ${chunk}`));
  child.on("exit", () => children.delete(child));
}

function spawnTracked(name, command, args, options) {
  const child = spawn(command, args, {
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true,
    ...options,
  });
  children.add(child);
  pipeChildLogs(child, name);
  return child;
}

function pythonImports(command, prefixArgs, moduleName) {
  const check = spawnSync(command, [...prefixArgs, "-c", `import ${moduleName}`], {
    stdio: "ignore",
    windowsHide: true,
  });
  return check.status === 0;
}

function absolutePythonCandidates() {
  const candidates = [];
  const home = os.homedir();
  for (const prefix of [
    process.env.CONDA_PREFIX,
    path.join(home, "mambaforge", "envs", "quest"),
    path.join(home, "mambaforge"),
    path.join(home, "miniforge3", "envs", "quest"),
    path.join(home, "miniforge3"),
    path.join(home, "miniconda3", "envs", "quest"),
    path.join(home, "miniconda3"),
  ]) {
    if (prefix) {
      candidates.push(path.join(prefix, "bin", "python"));
    }
  }
  candidates.push("/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3");
  return [...new Set(candidates)].filter((candidate) => fs.existsSync(candidate));
}

function pythonFromLoginShell() {
  const check = spawnSync(
    "/bin/zsh",
    ["-lic", "python - <<'PY'\nimport sys, uvicorn, fastapi\nprint(sys.executable)\nPY"],
    { encoding: "utf8", windowsHide: true },
  );
  if (check.status !== 0) return null;
  const executable = check.stdout.trim().split(/\r?\n/).at(-1);
  return executable && fs.existsSync(executable) ? executable : null;
}

function pythonCommand() {
  const bundledPython = getBundledPythonExecutable();
  if (bundledPython) {
    if (!pythonImports(bundledPython, [], "uvicorn")) {
      throw new Error(`Bundled Python exists at ${bundledPython}, but it cannot import uvicorn.`);
    }
    return { command: bundledPython, args: ["-m", "uvicorn"] };
  }
  if (process.env.QUEST_PYTHON) {
    if (!pythonImports(process.env.QUEST_PYTHON, [], "uvicorn")) {
      throw new Error(`QUEST_PYTHON is set to ${process.env.QUEST_PYTHON}, but it cannot import uvicorn.`);
    }
    return { command: process.env.QUEST_PYTHON, args: ["-m", "uvicorn"] };
  }
  const candidates =
    process.platform === "win32"
      ? [
          { command: "py", prefix: ["-3"] },
          { command: "python", prefix: [] },
          { command: "python3", prefix: [] },
        ]
      : [
          ...absolutePythonCandidates().map((command) => ({ command, prefix: [] })),
          { command: "python", prefix: [] },
          { command: "python3", prefix: [] },
        ];
  for (const candidate of candidates) {
    if (pythonImports(candidate.command, candidate.prefix, "uvicorn")) {
      return { command: candidate.command, args: [...candidate.prefix, "-m", "uvicorn"] };
    }
  }
  const shellPython = process.platform === "win32" ? null : pythonFromLoginShell();
  if (shellPython && pythonImports(shellPython, [], "uvicorn")) {
    return { command: shellPython, args: ["-m", "uvicorn"] };
  }
  throw new Error("Could not find Python with uvicorn/fastapi installed. Set QUEST_PYTHON.");
}

function startBackend(port) {
  const python = pythonCommand();
  const cwd = getBackendCwd();
  const env = {
    ...process.env,
    QUEST_WEBUI_DATA_DIR: process.env.QUEST_WEBUI_DATA_DIR || path.join(app.getPath("userData"), "data"),
    PYTHONPATH: [cwd, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
    PYTHONNOUSERSITE: "1",
    MPLBACKEND: "Agg",
  };
  fs.mkdirSync(env.QUEST_WEBUI_DATA_DIR, { recursive: true });
  const args = [...python.args, "webui.backend.app.main:app", "--host", "127.0.0.1", "--port", String(port)];
  return spawnTracked("fastapi", python.command, args, { cwd, env });
}

function startFrontend(port, backendPort) {
  const serverPath = getFrontendServerPath();
  if (!fs.existsSync(serverPath)) {
    throw new Error(`Frontend server missing at ${serverPath}. Run npm run build:frontend first.`);
  }
  const env = {
    ...process.env,
    HOSTNAME: "127.0.0.1",
    PORT: String(port),
    QUEST_BACKEND_URL: `http://127.0.0.1:${backendPort}`,
  };
  if (!app.isPackaged) {
    return spawnTracked("next", "node", [serverPath], { cwd: path.dirname(serverPath), env });
  }
  return spawnTracked("next", process.execPath, [serverPath], {
    cwd: path.dirname(serverPath),
    env: { ...env, ELECTRON_RUN_AS_NODE: "1" },
  });
}

function createWindow(frontendPort) {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 960,
    minWidth: 1100,
    minHeight: 720,
    show: false,
    title: "Quenching Estimator",
    webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true },
  });
  mainWindow.once("ready-to-show", () => mainWindow?.show());
  mainWindow.loadURL(`http://127.0.0.1:${frontendPort}`);
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

function showStartupError(error) {
  const detail = error instanceof Error ? error.stack || error.message : String(error);
  dialog.showErrorBox("quest failed to start", detail);
}

function stopChildren() {
  for (const child of children) {
    if (!child.killed) child.kill();
  }
}

async function boot() {
  await app.whenReady();
  try {
    const [backendPort, frontendPort] = await Promise.all([freePort(), freePort()]);
    startBackend(backendPort);
    await waitForHttp(`http://127.0.0.1:${backendPort}/health`);
    startFrontend(frontendPort, backendPort);
    await waitForHttp(`http://127.0.0.1:${frontendPort}`);
    frontendPortCache = frontendPort;
    createWindow(frontendPort);
  } catch (error) {
    showStartupError(error);
  }
}

app.on("window-all-closed", () => app.quit());
app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0 && frontendPortCache) createWindow(frontendPortCache);
});
app.on("before-quit", stopChildren);
process.on("exit", stopChildren);
process.on("SIGTERM", () => app.quit());
process.on("SIGINT", () => app.quit());

boot();

