# AGENTS.md — DistGPU

Руководство по архитектуре агентов, компонентов и их взаимодействию для AI-ассистентов и разработчиков, работающих с кодовой базой DistGPU.

---

## Что такое DistGPU

DistGPU — платформа распределённых GPU-вычислений. Пользователи загружают Jupyter-ноутбуки с PyTorch-кодом обучения; система распределяет задачи по воркерам (ПК участников с GPU), выполняет обучение и стримит логи обратно в браузер. Участники, предоставляющие GPU-мощности, в будущем получают вознаграждение.

**Текущий этап**: MVP на NVIDIA CUDA — одиночный воркер (**DDP** при `torchrun` / без группы) или **multi-node FSDP** при `shard_world_size > 1`; опционально **гибридный режим** (pipeline + сжатие активаций на границах стадий) при `DISTGPU_USE_PIPELINE=1` и YAML-конфиге; горячий резерв; SQLite-персистентность задач; без крипто-слоя.

---

## Структура репозитория

```
distgpu/
├── coordinator.py          # точка входа сервера (VPS): HTTP+браузер и фоновый WS для воркеров
├── executor.py             # конвертер .ipynb → скрипт с DDP/FSDP, чекпоинтами и веткой pipeline
├── config/
│   ├── hybrid.py           # датаклассы HybridConfig (FSDP / pipeline / subspace / context_parallel)
│   └── __init__.py
├── configs/                # примеры YAML для гибридного режима (llama, SDXL, long context)
├── pipeline/
│   ├── splitter.py         # разбиение модели по pipeline-стадиям
│   ├── comm.py             # передача активаций/градиентов между стадиями (+ Subspace)
│   ├── executor.py         # HybridParallelExecutor (smoke / задел под полный train loop)
│   ├── script_entry.py     # точка входа при DISTGPU_USE_PIPELINE=1 из сгенерированного скрипта
│   └── monitor.py          # логирование метрик сжатия (опционально)
├── compression/
│   ├── subspace.py         # SubspaceCompressor (активации на границе PP)
│   └── context_parallel.py  # MixtureOfSubspacesCP (заготовка, длинный контекст)
├── static/
│   └── index.html          # веб-интерфейс (vanilla JS), поле shard_world_size для FSDP
├── scheduler/
│   ├── __init__.py
│   ├── job_queue.py        # очередь, статусы, record_training_exit (в т.ч. FSDP-группа)
│   ├── state_store.py      # SQLite: задачи, shard_world_size, логи
│   ├── worker_registry.py  # реестр воркеров, advertise_host, get_idle_n
│   ├── health_monitor.py   # heartbeat; для FSDP — abort группы + reschedule
│   └── dispatcher.py       # одиночная задача + ветка FSDP (N воркеров, без backup)
└── worker/
    ├── native/                     # рекомендуемый воркер: C++ → один бинарник
    │   ├── CMakeLists.txt
    │   ├── src/main.cpp
    │   ├── install_native.sh / install_native.ps1
    │   └── distgpu-worker(/.exe)   # появляется после сборки в каталоге native/
    ├── agent.py                    # опционально: тот же протокол на Python (разработка)
    ├── requirements-base.txt
    ├── verify_env.py
    ├── install_worker.sh / install_worker.ps1
    └── run_worker.sh / run_worker.ps1
```

---

## Агенты и компоненты

### 1. Coordinator (`coordinator.py`)

**Роль**: главный процесс на VPS. Один процесс Python поднимает **два** ASGI-приложения uvicorn: браузерный API и отдельный сокет для воркеров.

**Запуск** (из корня репозитория, с `PYTHONPATH` = корень):

- `python -m distgpu.coordinator` или `python distgpu/coordinator.py`
- либо `uvicorn distgpu.coordinator:app --host 0.0.0.0 --port 8000` — тогда второй порт для воркеров поднимается **из lifespan** того же процесса (не нужен второй процесс).

**Порты** (по умолчанию):

| Переменная   | Значение по умолчанию | Назначение |
|-------------|------------------------|------------|
| `HOST`      | `0.0.0.0`              | bind HTTP/UI |
| `PORT`      | `8000`                 | HTTP, статика, `GET/POST` API, WS логов `/logs/{job_id}` |
| `WORKER_HOST` | то же, что `HOST`   | bind WS воркеров (можно отделить, например только внутренняя сеть) |
| `WORKER_PORT` | `8765`               | **только** WebSocket воркеров: `ws://<HOST>:8765/worker` |

**Ответственность**:
- Принимает HTTP-запросы от браузера (`POST /api/submit`, `GET /api/workers`, `GET /api/jobs/{id}`, `POST /api/jobs/{id}/cancel`)
- На `WORKER_PORT` принимает WebSocket воркеров (`/worker`)
- На `PORT` принимает WebSocket логов (`/logs/{job_id}`)
- Инициализирует и запускает планировщик (`Dispatcher`, `HealthMonitor`) при старте
- Регистрация/отключение воркеров → `WorkerRegistry`
- Сообщения воркера: `log`, `checkpoint`, `job_done`, `pong`

**Не делает**: не решает, кому назначить задачу (кроме отмены по API) — это `Dispatcher` и `HealthMonitor`.

**Переменные окружения**:
- `WORKER_TOKEN` или `TOKEN` — секрет для воркеров (dev: `secret-token-123`, если не задано)
- `HOST`, `PORT` — публичный HTTP/UI
- `WORKER_HOST`, `WORKER_PORT` — приём воркеров
- `LOG_LEVEL` — уровень логов uvicorn (по умолчанию `info`)
- `DISTGPU_DATA_DIR` — каталог для SQLite (`coordinator.sqlite`): задачи, статусы, чекпоинты, логи; по умолчанию `<корень_репо>/.distgpu_data`
- **`DISTGPU_RENDEZVOUS_HOST`** (на **координаторе**, опционально) — fallback для `MASTER_ADDR` в FSDP, если у воркера rank 0 пустой `advertise_host`; должен указывать на хост, где rank 0 принимает TCP на выбранном порту

После рестарта координатора задачи в статусах `assigned` / `running` переводятся в `queued` (соединения с воркерами потеряны); `last_checkpoint` / `last_step` и **`shard_world_size`** сохраняются; диспетчер снова назначит воркеров (в т.ч. FSDP-группу) с `resume: true` при наличии чекпоинта. Текст **`pipeline_config_yaml`** в SQLite не хранится — после рестарта для восстановленной задачи pipeline-конфиг из API нужно передать заново.

---

### 2. Worker (ПК участника)

Участник подключается к `ws://<VPS>:WORKER_PORT/worker`, получает JSON-команды, для `run_job` запускает **локальный Python** с присланным скриптом (PyTorch/CUDA остаются в пользовательском окружении Python). Протокол одинаков для нативного и Python-воркера.

#### 2a. Нативный воркер (рекомендуется) — `worker/native/`

**Зачем**: один скомпилированный файл без интерпретатора для логики клиента; проще доставка; **исходники оркестрации не отдаются** участникам (в репозитории есть C++, в дистрибутиве — только бинарник). Сгенерированный сервером скрипт обучения по-прежнему выполняется в их `python` — это отдельный артефакт задачи, не код платформы.

**Сборка**:
- Linux / macOS: `cd worker/native && chmod +x install_native.sh && ./install_native.sh`
- Windows: `cd worker\native` → `.\install_native.ps1`  
Требования: **CMake 3.16+**, компилятор **C++17**, **Git** (для FetchContent), сеть для загрузки IXWebSocket и nlohmann/json. После сборки бинарник: `worker/native/distgpu-worker` или `distgpu-worker.exe`.

**Запуск** (отдельного «run»-скрипта не нужно):

```bash
export SERVER_URL=ws://127.0.0.1:8765/worker
export WORKER_TOKEN=...
export PYTHON=python3   # опционально: какой python вызывать для обучения
./distgpu-worker
```

**GPU в `register`**: через вывод `nvidia-smi` (имя, memory.total/free в МиБ). Если `nvidia-smi` недоступен — в JSON уходит `available: false`.

**Переменные окружения воркера**: `SERVER_URL`, `WORKER_TOKEN` / `TOKEN`, `PYTHON` (имя/путь к Python для subprocess), `CHECKPOINT_EVERY` (прокидывается в дочерний процесс), **`DISTGPU_ADVERTISE_HOST`** — публичный IP/hostname для `tcp://` rendezvous (rank 0 в FSDP-группе); уходит в `register` как `advertise_host`.

**Переменные для дочернего Python** (как у Python-агента): `JOB_ID`, `CHECKPOINT_DIR` (`~/distgpu_checkpoints`), `CHECKPOINT_EVERY`, при resume — `CHECKPOINT_PATH`, `START_STEP`, `PYTHONUNBUFFERED=1`. При **`run_job.distributed`** дополнительно: `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `DISTGPU_MANUAL_INIT=1`, `DISTGPU_USE_FSDP=1`. При **`run_job.pipeline_enabled`** и **`pipeline_config_yaml`**: `DISTGPU_USE_PIPELINE=1`, `DISTGPU_CONFIG_PATH`, `DISTGPU_PIPELINE_STAGE_IDX` (+ опционально проброс `DISTGPU_SUBSPACE_RANK` / `DISTGPU_CONTEXT_PARALLEL` из окружения процесса воркера). Нативный воркер сбрасывает перечисленные dist/pipeline переменные после завершения subprocess (см. `env_unset_dist` в `main.cpp`).

**Жизненный цикл**: переподключение с паузой 5 с; стабильный `worker_id` в `~/.distgpu_worker_id`.

#### 2b. Python-воркер (опционально) — `worker/agent.py`

Для разработки и отладки протокола без сборки C++. Запуск: `install_worker.*` + `run_worker.*` или `python agent.py`. Нужны `websockets`, `torch` (для чтения VRAM в `register`).

**Команды от сервера** (нативный и Python одинаково):

| Тип | Действие |
|-----|----------|
| `ping` | `pong` |
| `run_job` | записать скрипт во временный `.py`, subprocess `python`, стрим строк в `log`, разбор `__CHECKPOINT__` / `__RESUMED__`; опционально `distributed` (FSDP); опционально `pipeline_enabled` + `pipeline_config_yaml` + `pipeline_stage_idx` (гибридный конфиг на диск + env) |
| `sync` | лог в консоль (standby / active) |
| `cancel_job` | завершить subprocess; при отмене `job_done` с `exit_code` 130 |

**Сообщения на сервер**: `register`, `pong`, `log`, `checkpoint`, `job_done`.

---

### 3. Job Queue (`scheduler/job_queue.py`)

**Роль**: хранилище задач с приоритетной очередью и управлением статусами. Опционально подключается `JobStateStore` (`scheduler/state_store.py`): SQLite в `DISTGPU_DATA_DIR/coordinator.sqlite`, восстановление через `JobQueue.restore_from_store()` при старте координатора.

**Статусы задачи** (машина состояний):

```
QUEUED → ASSIGNED → RUNNING → DONE
                 ↘          ↘
              RESCHEDULED ← FAILED → DEAD (retry = 3)
                 ↓              ↑
              QUEUED      CANCELLED (отмена по API)
```

**Ключевые поля `Job`**:
- `id` — уникальный идентификатор (8 символов UUID)
- `script` — Python-код для выполнения (сгенерирован из .ipynb)
- `owner_id` — идентификатор пользователя, отправившего задачу
- `priority` — приоритет (выше = важнее, по умолчанию `0`)
- `shard_world_size` — `1` = один воркер (DDP/torchrun по окружению); `>1` = FSDP-группа из N воркеров (персистится в SQLite)
- `assigned_worker_id` — при FSDP: воркер **rank 0** (master rendezvous); при одной задаче — основной воркер
- `shard_worker_ids` — порядок воркеров по рангам (только в RAM на время `RUNNING`)
- `backup_worker_id` — резерв (**только** если `shard_world_size == 1`): второй свободный по VRAM; для FSDP-задач не назначается
- `cancel_requested` — пользователь запросил отмену (`POST /api/jobs/{id}/cancel`)
- `retry_count` / `max_retries` — счётчик и лимит перезапусков (по умолчанию `3`)
- `last_checkpoint` / `last_step` — последний известный чекпоинт
- `subscribers` — список WebSocket браузеров, подписанных на логи этой задачи
- `pipeline_enabled` / `pipeline_config_yaml` — если задача отправлена с pipeline-конфигом: YAML целиком уходит в `run_job` каждому рангу; **в SQLite не сохраняются** (схема `state_store` без этих колонок) — после рестарта координатора pipeline-метаданные для восстановленных задач недоступны

**Завершение процессов** (`record_training_exit`): при FSDP координатор ждёт `job_done` от **всех** рангов; до этого воркеры группы остаются `BUSY`; при ненулевом exit на одном ранге остальным рассылается `cancel_job`. При ошибке участника FSDP — `HealthMonitor` отменяет пиров и `reschedule(job)`.

**Правило добавления**: `queue.push(job)` кладёт задачу в `asyncio.PriorityQueue` с ключом `(-priority, created_at)`. Высокий приоритет — раньше из очереди.

---

### 4. Worker Registry (`scheduler/worker_registry.py`)

**Роль**: реестр всех когда-либо подключавшихся воркеров с управлением их состоянием.

**Статусы воркера** (машина состояний):

```
CONNECTING → IDLE ⇄ BUSY
                ↘     ↓
            DISCONNECTED → STALE (30 сек) → REMOVED (5 мин)
                ↓
            CONNECTING (реконнект)
```

**Таймауты**:
- `STALE_TIMEOUT = 30` сек — после этого задача воркера передаётся замене
- `REMOVED_TIMEOUT = 300` сек — после этого воркер удаляется из реестра навсегда

**Важное поведение при реконнекте**: если воркер переподключается со своим `worker_id` пока он в статусе `DISCONNECTED` или `STALE` — реестр восстанавливает `current_job_id` для нового соединения. Это позволяет воркеру узнать о своей активной задаче и запросить синхронизацию.

**Выбор воркера для задачи** (`get_idle(min_vram_mb, exclude_ids=...)`): среди всех `IDLE` воркеров с **живым WebSocket** (`ws is not None`), достаточным VRAM и id не в `exclude_ids`, выбирается тот, у кого **больше всего свободного VRAM**. Диспетчер передаёт в `exclude_ids` воркеров, уже выбранных как `backup_worker_id` у других задач в `RUNNING`.

**Несколько воркеров под FSDP**: `get_idle_n(n, ...)` — последовательно выбирает `n` различных свободных воркеров.

**`advertise_host` у `Worker`**: задаётся при `register` (или сохраняется при реконнекте из прошлого состояния). Для FSDP **rank 0** должен иметь достижимый снаружи адрес: либо `advertise_host` этого воркера, либо **`DISTGPU_RENDEZVOUS_HOST`** на координаторе (fallback, если у rank 0 пусто). Порт rendezvous выбирает диспетчер случайно (`29600–29999`); на rank 0 должен быть открыт входящий TCP.

---

### 5. Health Monitor (`scheduler/health_monitor.py`)

**Роль**: фоновый сервис. Отвечает за обнаружение падений воркеров и восстановление задач.

**Два цикла**:

`_heartbeat_loop()` — каждые 10 секунд отправляет `ping` всем живым воркерам. Если отправка упала с исключением — вызывает `registry.disconnect()` и `_handle_worker_failure()`. Для задач с **`shard_world_size > 1`** немедленная установка `FAILED` по heartbeat **не** делается (см. `_replace_worker` при STALE).

`_stale_check_loop()` — каждые 5 секунд проверяет `DISCONNECTED` воркеров:
- Прошло > 30 сек → статус `STALE`, запускает `_replace_worker()`
- Прошло > 5 мин → статус `REMOVED`

**Логика замены воркера** (`_replace_worker`):
- Если **`shard_world_size > 1`** (FSDP): `cancel_job` всем оставшимся участникам группы, `queue.reschedule(job)`, `release_worker` для упавшего; полная перегруппировка — только после повторного планирования диспетчером.
- Иначе (одиночная задача):
  1. Проверить есть ли уже `backup_worker_id` у задачи → если да и он `IDLE`, использовать его
  2. Иначе — `registry.get_idle(exclude_ids=…)` (резервы **других** задач исключаются, резерв **этой** задачи — допустим)
  3. Если свободных нет — `queue.reschedule(job)` (задача возвращается в очередь)

При назначении замены передаётся `checkpoint_path` и `resume: true` — воркер продолжит с последнего чекпоинта, а не с нуля.

**`_assign_job_to_worker`**: после успешного `run_job` вызывает `assign_backup_for_running_job` **только** если `shard_world_size <= 1`.

---

### 6. Dispatcher (`scheduler/dispatcher.py`)

**Роль**: основной цикл назначения задач. Берёт задачи из очереди; для **`shard_world_size == 1`** назначает одного воркера и **резерв** (`sync` standby); для **`shard_world_size > 1`** — **FSDP-группу**: `get_idle_n`, сортировка по `worker_id` для стабильных рангов, общий `master_addr`/`master_port`, каждому свой `run_job` с `distributed`; **резерв не назначается**.

**Алгоритм (упрощённо)**:
```
while True:
    job = queue.pop()
    if нет задач → ждать new_job_event

    if job.shard_world_size > 1:
        workers = get_idle_n(N, min_vram=_vram_per_rank(job), exclude=reserved_backups)
        if мало воркеров → push job, ждать new_worker_event
        master = advertise_host(rank0) или DISTGPU_RENDEZVOUS_HOST; случайный порт
        каждому воркеру run_job + distributed { rank, world_size, master_addr, master_port, local_rank };
        при `job.pipeline_enabled` и непустом `pipeline_config_yaml` — дополнительно в том же сообщении:
        `pipeline_enabled: true`, `pipeline_config_yaml` (строка), `pipeline_stage_idx` (= rank стадии)
        RUNNING; backup не вызывать
    else:
        worker = get_idle(...); если нет → push, ждать
        run_job одному воркеру; RUNNING; assign_backup_for_running_job
```

**Уведомления**: `dispatcher.notify_new_job()` вызывается из Coordinator при получении новой задачи; `dispatcher.notify_new_worker()` — при регистрации нового воркера.

**Оценка VRAM**: `_estimate_vram(job)` — эвристика по тексту скрипта; для FSDP используется **`_vram_per_rank`** ≈ `estimate // shard_world_size` (грубая нижняя оценка на ранг, не замена явного `required_vram_mb`).

---

### Память GPU и «несколько воркеров на одну задачу»

**DDP (без FSDP):** на каждом процессе — **полная копия модели** на свою долю данных. Два воркера по 8 ГБ **не складывают** VRAM как общую RAM: каждому нужна полная модель в памяти.

**FSDP (`shard_world_size > 1`, реализовано):** параметры/оптимизатор **шардируются** между рангами (`FullyShardedDataParallel`, `FULL_SHARD`); цель — снизить **пиковую** VRAM на карту при той же логической модели. Чекпоинты: полный state на rank 0 через `FSDP.state_dict_type` / `full_optim_state_dict` и симметричная загрузка.

**Ограничение по сети:** FSDP сильно нагружает канал **синхронизацией градиентов/параметров** (NCCL по `tcp://`), а не передачей активаций между слоями на разных машинах. На «домашнем» интернете это часто узкое место **latency + bandwidth**, не только объём данных.

**Пул воркеров:** параллельно могут идти **разные** задачи (`shard_world_size == 1` + backup); либо одна FSDP-задача занимает N воркеров без резерва.

### Гибридный параллелизм (FSDP + pipeline + Subspace)

**Зачем:** FSDP снижает VRAM (шардирование весов), но не убирает узкое место **ширины канала** между воркерами. **Pipeline parallelism** делит слои по процессам; на границах стадий пересылаются активации — их можно сжимать (**Subspace Networks**, идеи NeurIPS 2025: низкоранговая проекция + опционально error feedback).

**Где в коде:**
- Конфиг: `distgpu/config/hybrid.py` (`HybridConfig.from_yaml` / `from_env`), примеры в `distgpu/configs/*.yaml`. Для YAML на координаторе нужен **PyYAML** (`requirements.txt` в корне репозитория).
- Сплит модели: `distgpu/pipeline/splitter.py` (`split_model`, `get_stage_boundaries`, баланс по числу слоёв или по параметрам).
- Сжатие: `distgpu/compression/subspace.py` (`SubspaceCompressor`), опционально `context_parallel.py` (`MixtureOfSubspacesCP` — заготовка под длинный контекст).
- Связь стадий: `distgpu/pipeline/comm.py` (`PipelineCommunicator`, NCCL `send`/`recv` по тегам).
- Оркестрация smoke: `distgpu/pipeline/executor.py` (`HybridParallelExecutor`), вход из скрипта: `distgpu/pipeline/script_entry.py` (`setup_and_run_hybrid`).

**Активация в сгенерированном ноутбук-скрипте:** после инициализации distributed и чекпоинт-хелперов вставляется ветка: если **`DISTGPU_USE_PIPELINE=1`**, вызывается `setup_and_run_hybrid()`, затем **`SystemExit(0)`** — пользовательские ячейки ноутбука **не выполняются** (режим smoke / отладки гибрида). Без этой переменной поведение как раньше: полный user code + FSDP/DDP.

**Переменные окружения дочернего Python** (в дополнение к FSDP-группе): `DISTGPU_USE_PIPELINE`, `DISTGPU_CONFIG_PATH` (путь к YAML), `DISTGPU_PIPELINE_STAGE_IDX`, опционально `DISTGPU_SUBSPACE_RANK`, `DISTGPU_CONTEXT_PARALLEL`, `DISTGPU_PIPELINE_STAGES`, `DISTGPU_D_MODEL`, `DISTGPU_SUBSPACE_ENABLED` — см. `HybridConfig._apply_env_overrides`.

**Воркер `agent.py`:** к `PYTHONPATH` добавляется корень репозитория (родитель каталога `worker/`), чтобы импортировать `distgpu.*` из временного скрипта. **Нативный воркер** этого не делает: при pipeline нужен установленный пакет или выставленный **`PYTHONPATH`** к корню репо.

**Координатор `POST /api/submit`:** кроме `file` и `shard_world_size` можно передать **`pipeline_enabled`** (`1` / `true`) и второй файл формы **`pipeline_config`** (текст YAML); тогда у `Job` выставляются `pipeline_enabled` и `pipeline_config_yaml`, диспетчер дополняет `run_job` для FSDP-группы.

---

### 7. Executor (`distgpu/executor.py`)

**Роль**: конвертирует загруженный `.ipynb` в Python-скрипт с единым блоком **distributed + DDP/FSDP** и опциональной веткой **pipeline**.

**Что делает**:
1. Парсит ноутбук через `nbformat`
2. Извлекает код из всех `code`-ячеек, пропускает Jupyter-магики (`%matplotlib` и т.д.)
3. Вставляет блок инициализации: при `DISTGPU_MANUAL_INIT=1` — `init_process_group(..., init_method="tcp://MASTER_ADDR:MASTER_PORT", ...)`; иначе при наличии `RANK`/`WORLD_SIZE` — классический `init_process_group(backend="nccl")` (как под `torchrun`); в логе суффикс `+PIPELINE`, если задано `DISTGPU_USE_PIPELINE=1`
4. Вставляет хелперы чекпоинтов
5. **Ветка pipeline:** при `DISTGPU_USE_PIPELINE=1` вызывается `distgpu.pipeline.script_entry.setup_and_run_hybrid()`, затем `SystemExit` **до** пользовательского кода из ноутбука
6. Иначе — пользовательский код, затем **`wrap_model`**: при `DISTGPU_USE_FSDP=1` — `FullyShardedDataParallel` (`FULL_SHARD`, `use_orig_params=True`); иначе при инициализированном процесс-группе — **DDP**; иначе модель без обёртки
7. **`get_device`**, `save_checkpoint` / `load_checkpoint` с ветками для FSDP (`FULL_STATE_DICT`, `full_optim_state_dict` / `shard_full_optim_state_dict`)
8. В конец — `cleanup_ddp()` (уничтожение process group)

**Контракт с пользователем** (когда ветка pipeline не срабатывает): пользователь пишет обычный PyTorch-код и вызывает `wrap_model(model)`. Переменные `DEVICE`, `wrap_model`, `save_checkpoint`, `load_checkpoint` доступны глобально.

**Служебные stdout-маркеры**, которые агент перехватывает:
- `__CHECKPOINT__:<path>:<step>` — сигнал о сохранённом чекпоинте
- `__RESUMED__:<step>` — сигнал об успешной загрузке чекпоинта

---

## Протокол WebSocket

### Воркер → Сервер

```jsonc
// Регистрация (advertise_host опционально, для FSDP rank 0 / tcp master)
{"type": "register", "token": "...", "worker_id": "abc123",
 "gpu": {"name": "RTX 3060", "vram_total_mb": 8192, "vram_free_mb": 7800},
 "advertise_host": "203.0.113.5"}

// Лог строки обучения
{"type": "log", "job_id": "job_42", "text": "Epoch 1, loss: 0.342"}

// Новый чекпоинт
{"type": "checkpoint", "job_id": "job_42", "path": "~/checkpoints/job_42_step100.pt", "step": 100}

// Задача завершена
{"type": "job_done", "job_id": "job_42", "exit_code": 0}

// Heartbeat ответ
{"type": "pong"}
```

### Сервер → Воркер

```jsonc
// Запуск задачи
{"type": "run_job", "job_id": "job_42", "script": "...python code..."}

// Запуск задачи с продолжением
{"type": "run_job", "job_id": "job_42", "script": "...", 
 "resume": true, "checkpoint_path": "~/checkpoints/job_42_step100.pt", "start_step": 100}

// Multi-node FSDP (несколько воркеров): одинаковый script, разные rank
{"type": "run_job", "job_id": "job_42", "script": "...",
 "distributed": {
   "rank": 0, "world_size": 2,
   "master_addr": "203.0.113.5", "master_port": 29731,
   "local_rank": 0
 }}
// Воркер выставляет в subprocess: MASTER_ADDR/PORT, RANK, WORLD_SIZE, LOCAL_RANK,
// DISTGPU_MANUAL_INIT=1, DISTGPU_USE_FSDP=1. Rank 0 должен иметь доступный tcp master_port.

// Тот же FSDP-скрипт + pipeline (YAML в теле сообщения; воркер пишет во временный файл):
{"type": "run_job", "job_id": "job_42", "script": "...",
 "distributed": { "rank": 1, "world_size": 3, "master_addr": "...", "master_port": 29731, "local_rank": 0 },
 "pipeline_enabled": true,
 "pipeline_config_yaml": "d_model: 3072\\npipeline:\\n  ...",
 "pipeline_stage_idx": 1 }
// Дополнительно в subprocess: DISTGPU_USE_PIPELINE=1, DISTGPU_CONFIG_PATH, DISTGPU_PIPELINE_STAGE_IDX;
// опционально из окружения воркера: DISTGPU_SUBSPACE_RANK, DISTGPU_CONTEXT_PARALLEL.

// Синхронизация при реконнекте
{"type": "sync", "has_active_job": true, "job_id": "job_42",
 "current_worker_id": "xyz789", "current_step": 350,
 "checkpoint_path": "~/checkpoints/job_42_step300.pt", "role": "standby"}

// Heartbeat
{"type": "ping"}

// Отмена обучения (после POST /api/jobs/{id}/cancel)
{"type": "cancel_job", "job_id": "job_42"}
```

### Браузер → Сервер (HTTP)

```
POST /api/submit              — multipart: `file` (.ipynb), `shard_world_size` (1 = один воркер; >1 = FSDP-группа, столько свободных воркеров с `advertise_host` / `DISTGPU_RENDEZVOUS_HOST`); опционально `pipeline_enabled` (`1`/`true`) и второй файл поля **`pipeline_config`** (текст YAML гибридного конфига)
GET  /api/workers             — список подключённых воркеров с GPU-инфо
GET  /api/jobs/{job_id}       — статус, логи, assigned_worker_id, backup_worker_id, cancel_requested, shard_world_size, shard_worker_ids
POST /api/jobs/{job_id}/cancel — только для `running`; при FSDP — `cancel_job` **всем** `shard_worker_ids`, иначе назначенному воркеру
WS   /logs/{job_id}           — стриминг логов (на порту PORT, как и UI)
```

Воркеры подключаются **только** к `WORKER_HOST:WORKER_PORT` (`/worker`), не к `PORT`.

---

## Сценарии восстановления

### Воркер переподключается быстро (< 30 сек)

1. Воркер разрывает соединение → `registry.disconnect(worker_id)`
2. `health_monitor` видит `DISCONNECTED`, ждёт
3. Воркер переподключается, отправляет `register` со своим `worker_id`
4. `registry.register()` видит что воркер был `DISCONNECTED`, восстанавливает `current_job_id`
5. Сервер отправляет воркеру `sync` с текущим состоянием задачи
6. Воркер загружает чекпоинт и продолжает с последнего шага

**FSDP-задача (`shard_world_size > 1`):** при реконнекте повторный `run_job` **не** отправляется (распределённое состояние не восстанавливается автоматически); только `sync` при необходимости.

### Воркер недоступен > 30 сек (STALE)

1. `health_monitor._stale_check_loop()` переводит воркера в `STALE`
2. Вызывается `_replace_worker()` → находится свободный воркер-замена
3. Замене отправляется `run_job` с `resume: true` и последним `checkpoint_path`
4. Замена продолжает обучение с последнего чекпоинта
5. Когда оригинальный воркер возвращается — сервер отправляет ему `sync` с `role: standby`
6. Воркер загружает чекпоинт от замены и работает в горячем резерве

### Нет свободных воркеров для замены

1. `_replace_worker()` не находит никого свободного
2. `queue.reschedule(job)` — задача возвращается в очередь с `retry_count + 1`
3. Как только появляется свободный воркер — `dispatcher.notify_new_worker()` → задача берётся из очереди

### Задача исчерпала retry (retry = 3)

1. `queue.reschedule()` проверяет `retry_count >= max_retries`
2. Статус задачи → `DEAD`
3. Все подписанные браузеры получают уведомление `job_dead`
4. Задача остаётся в `_jobs` для истории, но больше не планируется

---

## Соглашения по коду

**Async**: весь сетевой код — `async/await`. Блокирующие операции (torch, файловая система) запускать через `asyncio.to_thread()` если они занимают > 100 мс.

**Windows**: в любом файле, запускающем subprocess или использующем asyncio, добавлять:
```python
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
```

**Пути к файлам**: использовать `pathlib.Path` везде, не строки.

**Subprocess**: запускать через `sys.executable` — не хардкодить `"python"` или `"python3"`.

**Логирование**: формат `[Component] Сообщение`, например `[Dispatcher] Задача job_42 → воркер abc123`.

**Сериализация**: все WebSocket-сообщения — JSON. Ключ `type` обязателен в каждом сообщении.

**Нативный воркер (`worker/native`)**: C++17; отправка в WebSocket из колбэка IXWebSocket и из потока обучения — под одним `mutex` с указателем на активный `ix::WebSocket`; не блокировать колбэк длительным `run_job` (обучение — в `std::thread`).

**Токены и секреты**: не хардкодить в коде. Читать из переменных окружения или `.env`-файла.

---

## Зависимости

### Сервер (VPS)

```
fastapi
uvicorn[standard]
websockets
nbformat
pyyaml          # HybridConfig.from_yaml и приём pipeline_config при submit
```

Корневой файл **`requirements.txt`** содержит минимум `pyyaml` для конфигов; остальные зависимости координатора обычно ставятся отдельно (см. README).

### Воркер (ПК участника)

**Нативный (`worker/native`)**: CMake 3.16+, C++17, Git; зависимости тянутся через FetchContent (IXWebSocket, nlohmann/json). Для обучения на машине участника по-прежнему нужен **Python с PyTorch/CUDA** (или CPU), вызываемый бинарником (`PYTHON` / `python` в PATH).

**Python-воркер (`worker/agent.py`, опционально)**:

```
websockets
torch>=2.3.0         # отдельно, с нужной CUDA (см. install_worker.*)
```

Пример PyTorch с CUDA 12.4:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

---

## Дорожка после MVP (приоритеты)

1. **Сделано в коде (база под OOM):** multi-node **FSDP** при `shard_world_size > 1` (`run_job.distributed`, rendezvous, чекпоинты в `executor`).

2. **Сделано заделом (сеть / PP):** модули **pipeline + Subspace** (`distgpu/pipeline/*`, `distgpu/compression/subspace.py`), конфиг **`HybridConfig`**, примеры **`distgpu/configs/`**, проброс из диспетчера и воркера. Полный **train loop** ноутбука в режиме pipeline пока не интегрирован (smoke через `setup_and_run_hybrid` и выход до user code).

3. **Дальше по памяти / сети:** **ZeRO / DeepSpeed**, CPU-offload; явный **`required_vram_mb`** при submit вместо эвристики; тюнинг NCCL (`NCCL_SOCKET_IFNAME` и др.) для геораспределённых воркеров.

4. **Ускорение и узкие каналы:** **multi-node DDP** как отдельный продуктовый режим; **tensor parallel**; доработка **MoS context parallel** и интеграция в attention по флагу; **1F1B** pipeline schedule при большем числе стадий.

5. **Инференс** с передачей активаций по сети — только если появится model-parallel inference; иначе узкие места другие.

---

## Что не входит в MVP

Следующие возможности **ещё** не реализованы полностью или сознательно вне текущего объёма:

- **Полный pipeline-train из ноутбука** — сейчас при `DISTGPU_USE_PIPELINE=1` выполняется только **smoke** (`HybridParallelExecutor`) и выход; нет склейки с пользовательскими ячейками и реальной моделью из .ipynb.
- **Tensor Parallel**, **1F1B** и полноценная **Mixture-of-Subspaces context parallel** в attention — задел в `context_parallel.py`, без патчинга моделей из коробки.
- **Multi-node DDP как отдельный режим** (несколько рангов без FSDP, только ускорение / больший batch) — сейчас multi-node группа в продукте завязана на **FSDP** (`DISTGPU_USE_FSDP=1`); классический DDP между воркерами DistGPU без шардирования весов отдельно не выделен.
- **Крипто-вознаграждения** — биллинг и токены
- **Поддержка AMD / Apple Silicon** — только NVIDIA CUDA
- **Аутентификация пользователей** — нет логина, нет ролей
- **TLS/HTTPS** — в dev используется незашифрованный WS