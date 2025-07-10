#!/usr/bin/env bash
set -euo pipefail

[[ -e package.json ]] && {
  ver=$(grep -m1 '"version":' package.json|awk -F\" '{print $4}')
  tgz="google-gemini-cli-${ver}.tgz"
  [[ -e "$tgz" ]] && {
    bin="bundle/gemini.sh"
    cat "$0" > "$bin"
    echo "#${ver};$(base64 -w 0 "$tgz")" >> "$bin"
    echo "Release embedded script for $ver at ${bin}."
    exit 0
  }
}

[[ "$0" =~ .*gemini\.sh$ ]] || exit 0

which npm &>/dev/null || {
  echo "npm is not installed. Please install Node.js and npm first."
  exit 1
}

[[ $(node --version) =~ ^v2[0-9]\.[0-9]+\.[0-9]+$ ]] || {
  echo "Node.js version 20.x is required. Please update your Node.js installation."
  exit 1
}

which gemini &>/dev/null || {
  ver=$(tail -n 1 "$0" | cut -c 2-64 | awk -F\; '{print $1}')
  tgz="google-gemini-cli-${ver}.tgz"
  let pos=3+${#ver} #...;
  tail -n 1 "$0" | cut -c ${pos}- | base64 -d > "${tgz}"
  npm install -g --verbose "${tgz}" || rm -f "${tgz}" || :
  which gemini &>/dev/null || {
    echo "gemini CLI installation failed. Please check the output above for errors."
    exit 1
  }
}

[[ -e ~/.geminirc ]] && {
  . ~/.geminirc
} || {
  echo "~/.geminirc is not found, you can save your CUSTOM_LLM_API_KEY/CUSTOM_LLM_BASE_URL there".
}

[[ -z "${CUSTOM_LLM_API_KEY:-}" ]] && {
  echo "CUSTOM_LLM_API_KEY is not set. Please set it to your OpenAI API key as an environment."
  exit 1
}

[[ -z "${CUSTOM_LLM_BASE_URL:-}" ]] && {
  echo "CUSTOM_LLM_BASE_URL is not set. Please set it to your OpenAI API base URL as an environment."
  exit 1
}

[[ -n "${MODEL:-}" ]] && gemini --model "${MODEL}" "$@" || gemini "$@" ; exit $?
# The following line is the embedded package content, encoded in base64.
