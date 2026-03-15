#!/usr/bin/env node

const { spawn } = require('node:child_process');

const passthroughArgs = process.argv.slice(2);
const pythonCmd = process.env.PYTHON_CMD || 'python';
const cmdArgs = ['motion_control.py', ...passthroughArgs];

const child = spawn(pythonCmd, cmdArgs, { stdio: 'inherit' });

child.on('error', (error) => {
  console.error(`Failed to start ${pythonCmd}:`, error.message);
  process.exit(1);
});

child.on('exit', (code, signal) => {
  if (signal) {
    console.error(`Process terminated by signal: ${signal}`);
    process.exit(1);
  }
  process.exit(code ?? 1);
});
