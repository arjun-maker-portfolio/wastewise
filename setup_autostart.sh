#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_USER="${SUDO_USER:-}"

if [ "$EUID" -ne 0 ] || [ -z "$RUN_USER" ] || [ "$RUN_USER" = root ]; then
    echo 'Run sudo bash setup_autostart.sh from your normal Pi account.'
    exit 1
fi

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/data"
chown "$RUN_USER:$(id -gn "$RUN_USER")" "$PROJECT_DIR/logs" "$PROJECT_DIR/data"
chmod +x "$PROJECT_DIR/scripts/start_detector.sh"

python3 - "$PROJECT_DIR" "$RUN_USER" <<'PY'
from pathlib import Path
import sys
root, user = Path(sys.argv[1]), sys.argv[2]
text = (root / 'foodwaste-detector.service').read_text()
text = text.replace('@PROJECT_DIR@', str(root)).replace('@USER@', user)
Path('/etc/systemd/system/foodwaste-detector.service').write_text(text)
PY

systemctl daemon-reload
systemctl enable foodwaste-detector.service
printf '%s\n' 'Installed. Start with: sudo systemctl start foodwaste-detector' \
    'Logs: journalctl -u foodwaste-detector -f'
