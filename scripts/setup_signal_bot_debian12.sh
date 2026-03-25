#!/usr/bin/env bash
# Quick setup for signal_bot on Debian 12 (bookworm).
# Usage (as a normal user, NOT root — venv must belong to you):
#   cd /path/to/Binance-analyst
#   bash scripts/setup_signal_bot_debian12.sh
# The script will call `sudo` only to install apt packages.
#
# After setup: edit .env at repo root, then run the bot or enable user systemd (see end).

set -euo pipefail

if [[ "${EUID}" -eq 0 ]]; then
  echo "Do not run this script as root. Run as your login user so .venv is owned by you; sudo is used only for apt." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV="${REPO_ROOT}/.venv"
SERVICE_NAME="binance-signal-bot.service"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"
SERVICE_FILE="${SYSTEMD_USER_DIR}/${SERVICE_NAME}"

need_sudo() {
  sudo "$@"
}

echo "==> Repo root: ${REPO_ROOT}"

if [[ ! -f "${REPO_ROOT}/signal_bot/runner.py" ]]; then
  echo "ERROR: signal_bot/runner.py not found under ${REPO_ROOT}" >&2
  exit 1
fi

echo "==> Installing OS packages (python3-venv, venv support)..."
export DEBIAN_FRONTEND=noninteractive
need_sudo apt-get update -qq
need_sudo apt-get install -y -qq \
  python3 \
  python3-venv \
  python3-pip \
  ca-certificates \
  curl

echo "==> Creating venv at ${VENV}..."
if [[ ! -d "${VENV}" ]]; then
  python3 -m venv "${VENV}"
fi
# shellcheck source=/dev/null
source "${VENV}/bin/activate"
python -m pip install -q --upgrade pip
python -m pip install -q -r "${REPO_ROOT}/signal_bot/requirements.txt"

ENV_TARGET="${REPO_ROOT}/.env"
EXAMPLE="${REPO_ROOT}/signal_bot/.env.example"
if [[ ! -f "${ENV_TARGET}" ]]; then
  if [[ -f "${EXAMPLE}" ]]; then
    cp "${EXAMPLE}" "${ENV_TARGET}"
    chmod 600 "${ENV_TARGET}"
    echo "==> Created ${ENV_TARGET} from signal_bot/.env.example — edit TELEGRAM_* before running."
  else
    echo "==> Create ${ENV_TARGET} with TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID"
  fi
else
  echo "==> ${ENV_TARGET} already exists; not overwriting."
fi

mkdir -p "${REPO_ROOT}/data/signal_klines_cache"

echo "==> Smoke test import..."
cd "${REPO_ROOT}"
PYTHONPATH="${REPO_ROOT}" "${VENV}/bin/python" -c \
  "from signal_bot.runner import evaluate_symbol; from backtest.strategies.swing_strategy import SwingTradingStrategy; print('import ok')"

echo ""
echo "----- Manual run (foreground) -----"
echo "  cd ${REPO_ROOT}"
echo "  source .venv/bin/activate"
echo "  # set TELEGRAM_* in .env at repo root, or export them"
echo "  python -m signal_bot.runner"
echo ""

ans="n"
if [[ -t 0 ]]; then
  read -r -p "Install systemd --user service to run bot on login? [y/N] " ans || true
fi
if [[ "${ans:-}" =~ ^[yY]$ ]]; then
  mkdir -p "${SYSTEMD_USER_DIR}"
  cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=Binance swing signal bot (Telegram)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${REPO_ROOT}
EnvironmentFile=${ENV_TARGET}
ExecStart=${VENV}/bin/python -m signal_bot.runner
Restart=on-failure
RestartSec=30

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  echo "==> Service file: ${SERVICE_FILE}"
  echo "    Enable & start:"
  echo "      systemctl --user enable --now ${SERVICE_NAME}"
  echo "    Logs:"
  echo "      journalctl --user -u ${SERVICE_NAME} -f"
  echo ""
  echo "    If the service must run without an interactive login:"
  echo "      sudo loginctl enable-linger \$USER"
else
  echo "Skipped systemd. You can run the bot manually (see above)."
fi

echo ""
echo "Done."
