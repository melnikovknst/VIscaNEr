#!/usr/bin/env bash
# winescanner: build and start everything with one command.
#
#   ./run.sh          start (GPU if available, otherwise CPU)
#   ./run.sh --cpu    force CPU
#   ./run.sh stop     stop the service
#   ./run.sh logs     follow the service log
#
# Needs Docker with the compose plugin and Git LFS; on Ubuntu/Debian both are
# installed automatically (sudo) when missing.
set -euo pipefail
cd "$(dirname "$0")"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
step() { printf '\n\033[1;35m▸ %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

apt_install() {
    command -v apt-get >/dev/null || fail "Установите вручную: $*"
    step "Устанавливаю: $*"
    sudo apt-get update -qq
    sudo apt-get install -y -qq "$@"
}

# --- Docker ------------------------------------------------------------------
if ! command -v docker >/dev/null; then
    apt_install docker.io docker-compose-v2
    sudo systemctl enable --now docker >/dev/null 2>&1 || true
fi
SUDO=()
if ! docker info >/dev/null 2>&1; then
    sudo docker info >/dev/null 2>&1 || fail "Docker не запущен: sudo systemctl start docker"
    SUDO=(sudo)
fi
if ! "${SUDO[@]}" docker compose version >/dev/null 2>&1; then
    apt_install docker-compose-v2
fi
command -v curl >/dev/null || apt_install curl
FILES=(-f compose.yaml)
TORCH_INDEX=
# `env` rather than exported variables: sudo would drop them.
compose() { "${SUDO[@]}" env TORCH_INDEX="$TORCH_INDEX" docker compose "${FILES[@]}" "$@"; }

case "${1:-}" in
    stop) compose down; exit 0 ;;
    logs) compose logs -f --tail 100; exit 0 ;;
esac

# --- Model weights (Git LFS) ---------------------------------------------------
# A file not yet downloaded is a small text pointer starting with "version".
if [ "$(head -c 7 models/stage2c/manual_stage2c_best.pt)" = "version" ]; then
    step "Скачиваю веса моделей и каталог (Git LFS, ~1 ГБ)"
    command -v git-lfs >/dev/null || apt_install git-lfs
    git lfs install --local >/dev/null
    git lfs pull
fi

[ -f .env ] || cp .env.example .env
PORT=$(grep -E '^PORT=' .env | cut -d= -f2 || true)
PORT=${PORT:-8080}

# --- GPU or CPU ----------------------------------------------------------------
GPU=0
if [ "${1:-}" != "--cpu" ] && command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    if grep -qi nvidia <<<"$("${SUDO[@]}" docker info 2>/dev/null)"; then
        GPU=1
    else
        bold "Найдена видеокарта NVIDIA, но нет NVIDIA Container Toolkit — запускаю на CPU."
        bold "Для GPU: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    fi
fi
if [ "$GPU" = 1 ]; then
    FILES+=(-f compose.gpu.yaml)
    TORCH_INDEX=https://download.pytorch.org/whl/cu128
    step "Сборка и запуск на GPU (первый раз — 5–10 минут)"
else
    TORCH_INDEX=https://download.pytorch.org/whl/cpu
    step "Сборка и запуск на CPU (первый раз — 5–10 минут)"
fi
if ! compose up -d --build; then
    [ "$GPU" = 1 ] || fail "Не удалось запустить контейнер, смотрите вывод выше."
    bold "GPU-контейнер не запустился — пробую CPU."
    FILES=(-f compose.yaml)
    TORCH_INDEX=https://download.pytorch.org/whl/cpu
    compose up -d --build
fi

# --- Wait for the models -------------------------------------------------------
step "Загружаю модели"
for _ in $(seq 1 180); do
    health=$(curl -fsS "http://127.0.0.1:${PORT}/api/health" 2>/dev/null || true)
    case "$health" in
        *'"model_ready":true'*) break ;;
        *'"model_status":"error"'*) compose logs --tail 50; fail "Модель не загрузилась, журнал выше." ;;
    esac
    sleep 2
done
case "$health" in *'"model_ready":true'*) ;; *) fail "Сервис не ответил за 6 минут: ./run.sh logs" ;; esac

printf '\n\033[1;32m✓ winescanner работает\033[0m\n\n'
echo "  Сайт:       http://127.0.0.1:${PORT}"
echo "  Проверка:   curl -F image=@photo.jpg http://127.0.0.1:${PORT}/v1/eval/predict"
echo "  API:        http://127.0.0.1:${PORT}/docs"
echo "  Журнал:     ./run.sh logs      Остановить: ./run.sh stop"
