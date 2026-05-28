# DistGPU

Платформа для запуска PyTorch-обучения из Jupyter-ноутбуков на удалённых GPU-воркерах: координатор на VPS принимает задачи, планировщик назначает воркеров, логи стримятся в браузер. Поддерживаются одиночный воркер (DDP при `torchrun`), **multi-node FSDP** при `shard_world_size > 1` и задел под **гибридный режим** (pipeline + сжатие активаций на границах стадий).

Подробная архитектура, протокол WebSocket и соглашения по коду описаны в [**AGENTS.MD**](AGENTS.MD).

---

## Требования

- **Координатор**: Python 3.10+, CUDA не обязательна на сервере.
- **Воркер**: NVIDIA GPU, драйвер, **Python с PyTorch (CUDA)** для запуска присланного скрипта обучения.

Зависимости координатора:

```bash
pip install -r requirements.txt
```

В [`requirements.txt`](requirements.txt): FastAPI, uvicorn, websockets, nbformat, PyYAML, PyJWT, passlib (bcrypt) для авторизации пользователей.

---

## Быстрый старт

Клонировать репозиторий и задать корень в `PYTHONPATH`, чтобы импортировался пакет `distgpu`:

```bash
cd share_power   # корень репозитория
export PYTHONPATH="$(pwd)"   # Windows PowerShell: $env:PYTHONPATH = (Get-Location).Path
export WORKER_TOKEN=secret-token-123   # тот же токен у воркера
python -m distgpu.coordinator
```

По умолчанию: HTTP/UI на порту **8000**, WebSocket воркеров на **8765** (см. переменные `HOST`, `PORT`, `WORKER_HOST`, `WORKER_PORT` в [AGENTS.MD](AGENTS.MD)).

**Воркер (один файл для участников):** скопируйте [`worker/worker.py`](worker/worker.py) на машину с GPU и запустите — при первом старте создаётся `.venv` и ставятся `websockets` и PyTorch:

```bash
export SERVER_URL=ws://192.168.1.10:8765/worker
export WORKER_TOKEN=secret-token-123
# опционально: пресет CUDA — cu124 (по умолчанию), cu121, nightly-cu128 и т.д.
export DISTGPU_TORCH=cu124
python worker.py
```

Или с аргументами: `python worker.py --server ws://HOST:8765/worker --token YOUR_TOKEN`

**RTX 5060 / 5070 / 5080 / 5090 (Blackwell, sm_120):** стабильный `cu124` не подходит — worker сам выберет `nightly-cu128`. Если уже ставился `cu124`, переустановите:

```powershell
$env:DISTGPU_TORCH = "nightly-cu128"
python worker.py --reinstall-torch
```

В веб-интерфейсе координатора отображаются число онлайн-воркеров, суммарная VRAM и карточки каждого GPU.

**Нативный C++-воркер** (без Python для клиента): `worker/native/` — см. [AGENTS.MD](AGENTS.MD).

Откройте в браузере `http://<публичный-IP-VPS>:8000` (не `127.0.0.1`, если сидите не на самом сервере), загрузите `.ipynb` и при необходимости укажите `shard_world_size` для FSDP-группы.

### Запуск на VPS (если веб не открывается)

1. **Координатор слушает все интерфейсы** (по умолчанию `HOST=0.0.0.0`). Не задавайте `HOST=127.0.0.1` на VPS.
2. **На сервере** проверьте, что процесс жив и порт открыт:
   ```bash
   curl -s http://127.0.0.1:8000/api/health
   ss -tlnp | grep 8000
   ```
   Ожидается `{"ok":true,...}` и строка с `0.0.0.0:8000` или `*:8000`.
3. **С вашего ПК** (подставьте IP VPS):
   ```bash
   curl -s http://ВАШ_IP:8000/api/health
   ```
   Если на VPS работает, а с ПК — нет, проблема в **firewall** или **security group** облака: откройте входящие **TCP 8000** (веб) и **8765** (воркеры).
4. **ufw** (Ubuntu):
   ```bash
   sudo ufw allow 8000/tcp
   sudo ufw allow 8765/tcp
   sudo ufw reload
   ```
5. Запуск из **корня репозитория** с зависимостями: `pip install -r requirements.txt`, затем `export PYTHONPATH="$(pwd)"` и `python -m distgpu.coordinator`.

### Воркер и ноутбуки (устранённые риски)

| Проблема | Что сделано в коде |
|----------|-------------------|
| Обрыв WS во время обучения (`keepalive ping timeout`) | `run_job` в фоне, увеличен `ping_timeout` |
| RTX 5060 / sm_120 | авто-пресет `nightly-cu128`, проверка после установки |
| `num_workers>0` на Windows | в скрипт вставляется безопасный `DataLoader` |
| Нет CUDA при `device=cuda` | явная ошибка до user code |
| Повторный `run_job` при reconnect | игнор, если процесс уже крутится |
| Потеря логов при закрытом WS | `_ws_send` с перехватом ошибок |
| Артефакты задачи | `worker/.distgpu_runtime/jobs/<job_id>/` на **том же диске**, что и `worker.py` (не `C:\Users\...`) |
| Отключение воркера | 30 с на reconnect, затем reschedule / резерв |

**На воркере** для ноутбуков с `torchvision` / CIFAR: нужны интернет (скачивание данных и весов) и `shard_world_size=1`.

**Пользователи:** регистрация и вход в веб-UI. Задачи и логи привязаны к аккаунту (JWT). В `.env` на координаторе задайте `DISTGPU_JWT_SECRET` (длинная случайная строка, ≥16 символов).

```env
DISTGPU_JWT_SECRET=ваша-длинная-случайная-строка
WORKER_TOKEN=secret-token-123
```

**Локальные результаты** (например `output/model.pt`) остаются на машине воркера; в браузер идут только логи.

**Диск воркера:** по умолчанию `worker/.distgpu_runtime/` (рядом с `worker.py`, например на `G:`). Свой путь: `DISTGPU_DATA_ROOT=G:\distgpu_storage`. В git не попадает (см. `.gitignore`). Папки `data/` и `output/` в корне репо тоже игнорируются — если они уже появились, удалите их перед коммитом.

---

## Гибридный режим (pipeline + YAML)

- Конфигурация: [`distgpu/config/hybrid.py`](distgpu/config/hybrid.py), примеры YAML в [`distgpu/configs/`](distgpu/configs/).
- Локальные проверки модулей (нужен установленный **torch**):

  ```bash
  python -m distgpu.pipeline.splitter
  python -m distgpu.compression.subspace
  ```

- При **`DISTGPU_USE_PIPELINE=1`** сгенерированный скрипт после инициализации NCCL вызывает smoke (`setup_and_run_hybrid`) и завершается **до** ячеек ноутбука; обычное обучение без этой переменной не меняется.

- Через API: `POST /api/submit` с полями `pipeline_enabled` и файлом `pipeline_config` (текст YAML) — см. [AGENTS.MD](AGENTS.MD).

---

## Структура репозитория (кратко)

| Путь | Назначение |
|------|------------|
| `distgpu/coordinator.py` | HTTP API, статика, WS логов, lifespan + WS воркеров |
| `distgpu/executor.py` | Ноутбук → Python-скрипт (DDP/FSDP, чекпоинты, ветка pipeline) |
| `distgpu/config/` | `HybridConfig` и вложенные датаклассы |
| `distgpu/pipeline/` | Сплит модели, коммуникация стадий, `HybridParallelExecutor` |
| `distgpu/compression/` | Subspace-компрессор, задел context-parallel |
| `scheduler/` | Очередь задач, диспетчер, health monitor, SQLite |
| `worker/` | Нативный C++-воркер и опциональный `agent.py` |

---

## Лицензия и статус

MVP без крипто-слоя и без аутентификации пользователей; детали ограничений — в разделе «Что не входит в MVP» в [AGENTS.MD](AGENTS.MD).
