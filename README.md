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

В [`requirements.txt`](requirements.txt): FastAPI, uvicorn, websockets, nbformat (конвертация `.ipynb`), PyYAML (гибридный конфиг и `pipeline_config` при submit).

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
