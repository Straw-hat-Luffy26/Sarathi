#!/usr/bin/env node
/**
 * Picks the fastest llama.cpp backend this machine can actually build, then runs
 * the requested tauri command with the matching cargo feature.
 *
 * Backend selection in llama.cpp is a compile-time decision: `llama-cpp-sys-2`
 * links GGML with CUDA or Vulkan support baked in, so a binary built without one
 * cannot acquire it at runtime no matter what hardware it later finds. That is
 * why this lives in the build and not in the app.
 *
 * Nothing here is specific to one machine. Everything is probed:
 *   - CUDA   requires an NVIDIA driver *and* the toolkit (nvcc), not just a GPU.
 *   - Vulkan requires the SDK (glslc/VULKAN_SDK), not just the runtime loader.
 *   - Neither present -> CPU, which always works.
 *
 * Usage:  node scripts/select-backend.mjs <dev|build> [extra tauri args...]
 */

import { spawnSync, spawn } from 'node:child_process';
import { existsSync, readdirSync, writeFileSync, unlinkSync } from 'node:fs';
import path from 'node:path';
import os from 'node:os';

const isWindows = process.platform === 'win32';

/** True when `cmd` runs and exits cleanly. */
function probe(cmd, args) {
  const r = spawnSync(cmd, args, { stdio: 'ignore', shell: isWindows });
  return r.status === 0;
}

/** Trimmed stdout of `cmd`, or '' if it could not be run. */
function capture(cmd, args) {
  const r = spawnSync(cmd, args, { encoding: 'utf8', shell: isWindows });
  return r.status === 0 ? (r.stdout || '').trim() : '';
}

/**
 * The CUDA architectures to compile kernels for, read from the GPUs present.
 *
 * This matters more than it looks. `llama-cpp-sys-2` does not set
 * `CMAKE_CUDA_ARCHITECTURES`, so the target list falls back to llama.cpp's own
 * default — a fixed set chosen when that version was released. A GPU newer than
 * the list gets a binary with no kernels it can run: CUDA initialisation fails,
 * llama.cpp falls back to CPU, and nothing says so. The symptom is a GPU sitting
 * at 0% while the CPU saturates, which is indistinguishable from not having
 * built with CUDA at all.
 *
 * `nvidia-smi` reports compute capability as `12.0`; CMake wants `120`. Every
 * distinct capability found is passed, so a machine with two different cards
 * gets kernels for both.
 *
 * Returns null when the capability cannot be read, which leaves llama.cpp's
 * default in place rather than guessing at one.
 */
function cudaArchitectures() {
  const out = capture('nvidia-smi', ['--query-gpu=compute_cap', '--format=csv,noheader']);
  if (!out) return null;

  const archs = [
    ...new Set(
      out
        .split(/\r?\n/)
        .map((l) => l.trim())
        .filter((l) => /^\d+\.\d+$/.test(l))
        .map((l) => l.replace('.', ''))
    ),
  ];

  return archs.length > 0 ? archs.join(';') : null;
}

function detectBackend() {
  if (process.env.SARATHI_BACKEND) {
    const forced = process.env.SARATHI_BACKEND.toLowerCase();
    return { feature: forced === 'cpu' ? null : forced, why: 'forced by SARATHI_BACKEND' };
  }

  const hasNvidiaGpu = probe('nvidia-smi', ['--query-gpu=name', '--format=csv,noheader']);
  const hasNvcc = probe('nvcc', ['--version']);
  if (hasNvidiaGpu && hasNvcc) {
    return { feature: 'cuda', why: 'NVIDIA GPU + CUDA toolkit (nvcc) present' };
  }

  const hasVulkanSdk = Boolean(process.env.VULKAN_SDK) || probe('glslc', ['--version']);
  if (hasVulkanSdk) {
    return { feature: 'vulkan', why: 'Vulkan SDK present (vendor-neutral GPU offload)' };
  }

  if (hasNvidiaGpu && !hasNvcc) {
    return { feature: null, why: 'NVIDIA GPU found but no CUDA toolkit — install it for GPU offload' };
  }
  return { feature: null, why: 'no GPU build toolchain found' };
}

/**
 * Locates a Visual Studio environment script.
 *
 * Windows CUDA builds need two things the default path lacks: a host compiler on
 * PATH, and a generator that does not depend on the CUDA MSBuild integration
 * (which the toolkit ships but does not always register). Searching rather than
 * assuming a path keeps this working across VS editions and years.
 */
function findVcvars() {
  const roots = [process.env['ProgramFiles(x86)'], process.env.ProgramFiles].filter(Boolean);

  const vswhere = roots
    .map((r) => path.join(r, 'Microsoft Visual Studio', 'Installer', 'vswhere.exe'))
    .find(existsSync);

  if (vswhere) {
    const r = spawnSync(vswhere, ['-latest', '-products', '*', '-property', 'installationPath'], {
      encoding: 'utf8',
    });
    const base = (r.stdout || '').trim().split(/\r?\n/)[0];
    if (base) {
      const candidate = path.join(base, 'VC', 'Auxiliary', 'Build', 'vcvars64.bat');
      if (existsSync(candidate)) return candidate;
    }
  }

  // vswhere is absent on Build Tools-only installs; walk the standard layout.
  for (const root of roots) {
    const vsRoot = path.join(root, 'Microsoft Visual Studio');
    if (!existsSync(vsRoot)) continue;
    for (const year of readdirSync(vsRoot)) {
      const editions = path.join(vsRoot, year);
      let entries = [];
      try {
        entries = readdirSync(editions);
      } catch {
        continue;
      }
      for (const edition of entries) {
        const candidate = path.join(editions, edition, 'VC', 'Auxiliary', 'Build', 'vcvars64.bat');
        if (existsSync(candidate)) return candidate;
      }
    }
  }
  return null;
}

const [, , mode = 'build', ...rest] = process.argv;
if (!['dev', 'build'].includes(mode)) {
  console.error(`usage: select-backend.mjs <dev|build> [tauri args]`);
  process.exit(2);
}

const { feature, why } = detectBackend();
const tauriArgs = [mode, ...(feature ? ['--features', feature] : []), ...rest];

console.log(`[sarathi] backend: ${feature ?? 'cpu'} — ${why}`);
console.log(`[sarathi] tauri ${tauriArgs.join(' ')}`);

// Building CPU-only on a machine with a GPU is almost always an accident, and
// the resulting binary is indistinguishable from a working one until someone
// watches a model saturate the CPU. Saying so here is cheaper than discovering
// it during inference.
if (!feature && probe('nvidia-smi', ['--query-gpu=name', '--format=csv,noheader'])) {
  console.warn('');
  console.warn('[sarathi] ####################################################################');
  console.warn('[sarathi] #  An NVIDIA GPU is present, but this build will be CPU-ONLY.      #');
  console.warn('[sarathi] #  GPU support in llama.cpp is compiled in, not detected at run    #');
  console.warn('[sarathi] #  time, so this binary cannot use the card no matter what it      #');
  console.warn('[sarathi] #  later finds. Models will load into system RAM and run on CPU.   #');
  console.warn('[sarathi] #                                                                  #');
  console.warn('[sarathi] #  Install the CUDA Toolkit (nvcc) and run this command again.     #');
  console.warn('[sarathi] ####################################################################');
  console.warn('');
}

// A CUDA build on Windows needs the MSVC environment and the Ninja generator;
// everywhere else the default toolchain is already correct.
if (isWindows && feature === 'cuda') {
  const vcvars = findVcvars();
  if (!vcvars) {
    console.error('[sarathi] CUDA selected but no Visual Studio C++ environment was found.');
    console.error('[sarathi] Install VS Build Tools with the C++ workload, or set SARATHI_BACKEND=cpu.');
    process.exit(1);
  }
  console.log(`[sarathi] msvc env: ${vcvars}`);

  // CMake reads CUDAARCHS as the default for CMAKE_CUDA_ARCHITECTURES when the
  // project does not set one, which is exactly the gap here.
  const archs = cudaArchitectures();
  console.log(
    archs
      ? `[sarathi] cuda arch: ${archs} (from the GPUs present)`
      : '[sarathi] cuda arch: llama.cpp default — compute capability could not be read'
  );

  const quoted = tauriArgs.map((a) => (a.includes(' ') ? `"${a}"` : a)).join(' ');

  // Handed to cmd as a script file rather than as a `cmd /c "<line>"` argument.
  //
  // Node escapes embedded quotes when it builds a Windows command line, so the
  // quotes around the vcvars path arrived at cmd as `\"C:\Program Files...\"`.
  // cmd read that as a command literally named `\"C:\Program`, reported
  // "is not recognized as an internal or external command", and the `&&` chain
  // short-circuited — so the build never ran while the script reported success.
  // A batch file has no such quoting layer: what is written is what cmd reads.
  const EOL = String.fromCharCode(13, 10);
  const script = path.join(os.tmpdir(), `sarathi-build-${process.pid}.bat`);
  writeFileSync(
    script,
    [
      '@echo off',
      `call "${vcvars}" >nul || exit /b 1`,
      'set CMAKE_GENERATOR=Ninja',
      ...(archs ? [`set CUDAARCHS=${archs}`] : []),
      `npx tauri ${quoted}`,
    ].join(EOL) + EOL
  );

  const child = spawn('cmd', ['/c', script], { stdio: 'inherit' });
  child.on('exit', (code) => {
    try {
      unlinkSync(script);
    } catch {
      // A leftover temp file is not worth failing a successful build over.
    }
    process.exit(code ?? 1);
  });
} else {
  const env = { ...process.env };
  if (feature === 'cuda') {
    // Same reasoning as the Windows branch: a GPU newer than llama.cpp's
    // default architecture list gets a binary with no kernels it can run.
    const archs = cudaArchitectures();
    if (archs) {
      env.CUDAARCHS = archs;
      console.log(`[sarathi] cuda arch: ${archs} (from the GPUs present)`);
    }
  }
  const child = spawn('npx', ['tauri', ...tauriArgs], { stdio: 'inherit', shell: isWindows, env });
  child.on('exit', (code) => process.exit(code ?? 1));
}
