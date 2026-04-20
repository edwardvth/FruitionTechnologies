#!/usr/bin/env bash
# Install & enable the goose-detector systemd service.
# Run once: sudo bash scripts/install_systemd.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_SRC="${SCRIPT_DIR}/goose-detector.service"
UNIT_DST="/etc/systemd/system/goose-detector.service"

if [[ $EUID -ne 0 ]]; then
    echo "Run as root: sudo bash scripts/install_systemd.sh" >&2
    exit 1
fi

install -m 0644 "${UNIT_SRC}" "${UNIT_DST}"
mkdir -p /home/fruitiontech/fruitiontechnologies/goose_detection/results
chown fruitiontech:fruitiontech /home/fruitiontech/fruitiontechnologies/goose_detection/results

systemctl daemon-reload
systemctl enable goose-detector.service
systemctl restart goose-detector.service
systemctl --no-pager status goose-detector.service
