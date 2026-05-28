/**
 * DistGPU native worker — WebSocket + запуск обучения через subprocess python.
 * Логика оркестрации в бинарнике (без распространения .py воркера).
 */
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

static std::mutex g_ws_send_mtx;
static ix::WebSocket* g_ws_ptr = nullptr;

static void ws_send_text(const std::string& s) {
    std::lock_guard<std::mutex> lk(g_ws_send_mtx);
    if (g_ws_ptr) {
        g_ws_ptr->sendText(s);
    }
}

static std::string getenv_str(const char* key, const std::string& def = {}) {
    const char* v = std::getenv(key);
    return v ? std::string(v) : def;
}

static fs::path home_dir() {
#ifdef _WIN32
    std::string h = getenv_str("USERPROFILE");
    if (h.empty()) {
        h = getenv_str("HOMEDRIVE") + getenv_str("HOMEPATH");
    }
    return fs::path(h.empty() ? "." : h);
#else
    std::string h = getenv_str("HOME");
    return fs::path(h.empty() ? "." : h);
#endif
}

static std::string load_or_create_worker_id() {
    const fs::path f = home_dir() / ".distgpu_worker_id";
    if (std::ifstream in(f); in) {
        std::string w;
        in >> w;
        if (w.size() == 8) {
            bool ok = true;
            for (char c : w) {
                if (!std::isalnum(static_cast<unsigned char>(c))) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                return w;
            }
        }
    }
    static const char* hex = "0123456789abcdef";
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> d(0, 15);
    std::string w;
    for (int i = 0; i < 8; ++i) {
        w.push_back(hex[d(gen)]);
    }
    std::error_code ec;
    fs::create_directories(f.parent_path(), ec);
    if (std::ofstream out(f); out) {
        out << w;
    }
    return w;
}

static json query_gpu_nvidia_smi() {
    json gpu;
#ifdef _WIN32
    const char* cmd =
        "nvidia-smi --query-gpu=name,memory.total,memory.free "
        "--format=csv,noheader,nounits 2>nul";
    FILE* p = _popen(cmd, "r");
#else
    const char* cmd =
        "nvidia-smi --query-gpu=name,memory.total,memory.free "
        "--format=csv,noheader,nounits 2>/dev/null";
    FILE* p = popen(cmd, "r");
#endif
    if (!p) {
        gpu["available"] = false;
        gpu["name"] = nullptr;
        gpu["vram_total_mb"] = 0;
        gpu["vram_free_mb"] = 0;
        return gpu;
    }
    char buf[4096];
    std::string line;
    while (fgets(buf, sizeof buf, p)) {
        line += buf;
    }
#ifdef _WIN32
    _pclose(p);
#else
    pclose(p);
#endif
    while (!line.empty() &&
           (line.back() == '\n' || line.back() == '\r')) {
        line.pop_back();
    }
    if (line.empty()) {
        gpu["available"] = false;
        gpu["name"] = nullptr;
        gpu["vram_total_mb"] = 0;
        gpu["vram_free_mb"] = 0;
        return gpu;
    }
    std::vector<std::string> parts;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
        while (!cell.empty() && cell.front() == ' ') {
            cell.erase(cell.begin());
        }
        while (!cell.empty() && cell.back() == ' ') {
            cell.pop_back();
        }
        parts.push_back(cell);
    }
    if (parts.size() < 3) {
        gpu["available"] = false;
        gpu["name"] = nullptr;
        gpu["vram_total_mb"] = 0;
        gpu["vram_free_mb"] = 0;
        return gpu;
    }
    try {
        int total = std::stoi(parts[1]);
        int freev = std::stoi(parts[2]);
        gpu["available"] = true;
        gpu["name"] = parts[0];
        gpu["vram_total_mb"] = total;
        gpu["vram_free_mb"] = std::max(0, freev);
    } catch (...) {
        gpu["available"] = false;
        gpu["name"] = nullptr;
        gpu["vram_total_mb"] = 0;
        gpu["vram_free_mb"] = 0;
    }
    return gpu;
}

struct TrainingProc {
    std::atomic<bool> user_cancel{false};
#ifdef _WIN32
    HANDLE hProcess = nullptr;
    HANDLE hThread = nullptr;
#else
    pid_t pid = -1;
#endif
    int read_fd = -1;

    void close_read_fd() {
        if (read_fd >= 0) {
#ifdef _WIN32
            _close(read_fd);
#else
            ::close(read_fd);
#endif
            read_fd = -1;
        }
    }

    void terminate_child() {
#ifdef _WIN32
        if (hProcess) {
            TerminateProcess(hProcess, 1);
            WaitForSingleObject(hProcess, 15000);
            CloseHandle(hProcess);
            CloseHandle(hThread);
            hProcess = nullptr;
            hThread = nullptr;
        }
#else
        if (pid > 0) {
            ::kill(pid, SIGTERM);
            int st = 0;
            (void)waitpid(pid, &st, 0);
            pid = -1;
        }
#endif
    }

    ~TrainingProc() {
        terminate_child();
        close_read_fd();
    }

    void process_line(const std::string& job_id, const std::string& line) {
        constexpr const char* ck = "__CHECKPOINT__:";
        if (line.rfind(ck, 0) == 0) {
            std::string rest = line.substr(std::strlen(ck));
            auto pos = rest.rfind(':');
            if (pos != std::string::npos && pos > 0) {
                std::string path = rest.substr(0, pos);
                std::string step_s = rest.substr(pos + 1);
                try {
                    int step = std::stoi(step_s);
                    json j;
                    j["type"] = "checkpoint";
                    j["job_id"] = job_id;
                    j["path"] = path;
                    j["step"] = step;
                    ws_send_text(j.dump());
                } catch (...) {
                }
            }
            return;
        }
        if (line.rfind("__RESUMED__:", 0) == 0) {
            json j;
            j["type"] = "log";
            j["job_id"] = job_id;
            j["text"] = line;
            ws_send_text(j.dump());
            return;
        }
        json j;
        j["type"] = "log";
        j["job_id"] = job_id;
        j["text"] = line;
        ws_send_text(j.dump());
    }

    static void env_unset_dist() {
#ifdef _WIN32
        for (const char* k :
             {"MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK",
              "DISTGPU_MANUAL_INIT", "DISTGPU_USE_FSDP", "DISTGPU_USE_PIPELINE",
              "DISTGPU_CONFIG_PATH", "DISTGPU_PIPELINE_STAGE_IDX",
              "DISTGPU_SUBSPACE_RANK", "DISTGPU_CONTEXT_PARALLEL"}) {
            SetEnvironmentVariableA(k, nullptr);
        }
#else
        for (const char* k :
             {"MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK",
              "DISTGPU_MANUAL_INIT", "DISTGPU_USE_FSDP", "DISTGPU_USE_PIPELINE",
              "DISTGPU_CONFIG_PATH", "DISTGPU_PIPELINE_STAGE_IDX",
              "DISTGPU_SUBSPACE_RANK", "DISTGPU_CONTEXT_PARALLEL"}) {
            unsetenv(k);
        }
#endif
    }

    static void env_set_str(const char* k, const std::string& v) {
#ifdef _WIN32
        SetEnvironmentVariableA(k, v.c_str());
#else
        setenv(k, v.c_str(), 1);
#endif
    }

    int run_python_script(const std::string& job_id, const std::string& script,
                          bool resume, const std::string& ckpt_path, int start_step,
                          const std::string& python_exe, const json& dist,
                          bool pipeline_enabled,
                          const std::string& pipeline_config_yaml,
                          int pipeline_stage_idx) {
        user_cancel = false;
        terminate_child();
        close_read_fd();

        fs::path tmpdir = fs::temp_directory_path();
        std::string rnd;
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> d(0, 15);
            for (int i = 0; i < 6; ++i) {
                rnd.push_back("0123456789abcdef"[d(gen)]);
            }
        }
        fs::path script_path = tmpdir / ("distgpu_" + job_id + "_" + rnd + ".py");
        {
            std::ofstream out(script_path, std::ios::binary);
            if (!out) {
                return 1;
            }
            out << script;
        }

        const fs::path ckpt_dir = home_dir() / "distgpu_checkpoints";
        std::error_code ec;
        fs::create_directories(ckpt_dir, ec);

        env_unset_dist();
        if (dist.is_object() && dist.contains("master_addr") &&
            dist["master_addr"].is_string()) {
            env_set_str("MASTER_ADDR", dist["master_addr"].get<std::string>());
            env_set_str("MASTER_PORT",
                        std::to_string(dist.value("master_port", 0)));
            env_set_str("RANK", std::to_string(dist.value("rank", 0)));
            env_set_str("WORLD_SIZE",
                        std::to_string(dist.value("world_size", 1)));
            env_set_str("LOCAL_RANK",
                        std::to_string(dist.value("local_rank", 0)));
            env_set_str("DISTGPU_MANUAL_INIT", "1");
            env_set_str("DISTGPU_USE_FSDP", "1");
        }

        fs::path pipeline_cfg_on_disk;
        if (pipeline_enabled && !pipeline_config_yaml.empty()) {
            pipeline_cfg_on_disk =
                tmpdir / ("distgpu_pipe_" + job_id + "_" + rnd + ".yaml");
            {
                std::ofstream out(pipeline_cfg_on_disk, std::ios::binary);
                if (!out) {
                    fs::remove(script_path, ec);
                    return 1;
                }
                out << pipeline_config_yaml;
            }
            env_set_str("DISTGPU_USE_PIPELINE", "1");
            env_set_str("DISTGPU_CONFIG_PATH", pipeline_cfg_on_disk.string());
            env_set_str("DISTGPU_PIPELINE_STAGE_IDX",
                        std::to_string(pipeline_stage_idx));
            std::string sr = getenv_str("DISTGPU_SUBSPACE_RANK");
            if (!sr.empty()) {
                env_set_str("DISTGPU_SUBSPACE_RANK", sr);
            }
            std::string cp = getenv_str("DISTGPU_CONTEXT_PARALLEL");
            if (!cp.empty()) {
                env_set_str("DISTGPU_CONTEXT_PARALLEL", cp);
            }
        }

#ifdef _WIN32
        SetEnvironmentVariableA("PYTHONUNBUFFERED", "1");
        SetEnvironmentVariableA("JOB_ID", job_id.c_str());
        SetEnvironmentVariableA("CHECKPOINT_DIR", ckpt_dir.string().c_str());
        SetEnvironmentVariableA("CHECKPOINT_EVERY",
                                getenv_str("CHECKPOINT_EVERY", "100").c_str());
        if (resume && !ckpt_path.empty()) {
            SetEnvironmentVariableA("CHECKPOINT_PATH", ckpt_path.c_str());
            SetEnvironmentVariableA("START_STEP",
                                    std::to_string(start_step).c_str());
        } else {
            SetEnvironmentVariableA("CHECKPOINT_PATH", nullptr);
            SetEnvironmentVariableA("START_STEP", nullptr);
        }

        SECURITY_ATTRIBUTES sa{};
        sa.nLength = sizeof(sa);
        sa.bInheritHandle = TRUE;
        sa.lpSecurityDescriptor = nullptr;
        HANDLE rd = nullptr;
        HANDLE wr = nullptr;
        if (!CreatePipe(&rd, &wr, &sa, 0)) {
            fs::remove(script_path, ec);
            return 1;
        }
        SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);

        STARTUPINFOA si{};
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
        si.wShowWindow = SW_HIDE;
        si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        si.hStdOutput = wr;
        si.hStdError = wr;

        std::string cmd =
            "\"" + python_exe + "\" \"" + script_path.string() + "\"";
        std::vector<char> cmdline(cmd.begin(), cmd.end());
        cmdline.push_back('\0');

        PROCESS_INFORMATION pi{};
        if (!CreateProcessA(nullptr, cmdline.data(), nullptr, nullptr, TRUE,
                            CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
            CloseHandle(wr);
            CloseHandle(rd);
            fs::remove(script_path, ec);
            return 1;
        }
        CloseHandle(wr);
        hProcess = pi.hProcess;
        hThread = pi.hThread;
        read_fd = _open_osfhandle(reinterpret_cast<intptr_t>(rd), _O_RDONLY);
        if (read_fd < 0) {
            CloseHandle(rd);
            terminate_child();
            fs::remove(script_path, ec);
            return 1;
        }
#else
        if (resume && !ckpt_path.empty()) {
            setenv("CHECKPOINT_PATH", ckpt_path.c_str(), 1);
            setenv("START_STEP", std::to_string(start_step).c_str(), 1);
        } else {
            unsetenv("CHECKPOINT_PATH");
            unsetenv("START_STEP");
        }
        setenv("PYTHONUNBUFFERED", "1", 1);
        setenv("JOB_ID", job_id.c_str(), 1);
        setenv("CHECKPOINT_DIR", ckpt_dir.string().c_str(), 1);
        setenv("CHECKPOINT_EVERY", getenv_str("CHECKPOINT_EVERY", "100").c_str(), 1);

        int pipefd[2];
        if (pipe(pipefd) != 0) {
            fs::remove(script_path, ec);
            return 1;
        }
        pid = fork();
        if (pid < 0) {
            ::close(pipefd[0]);
            ::close(pipefd[1]);
            fs::remove(script_path, ec);
            return 1;
        }
        if (pid == 0) {
            ::close(pipefd[0]);
            dup2(pipefd[1], STDOUT_FILENO);
            dup2(pipefd[1], STDERR_FILENO);
            ::close(pipefd[1]);
            execlp(python_exe.c_str(), python_exe.c_str(), script_path.c_str(),
                   (char*)nullptr);
            _exit(127);
        }
        ::close(pipefd[1]);
        read_fd = pipefd[0];
#endif

        FILE* fp = (read_fd >= 0)
#ifdef _WIN32
                       ? _fdopen(read_fd, "rb")
#else
                       ? fdopen(read_fd, "r")
#endif
                       : nullptr;
        if (!fp) {
            terminate_child();
            fs::remove(script_path, ec);
            return 1;
        }
        read_fd = -1;

        std::string line;
        char ch = 0;
        while (fread(&ch, 1, 1, fp) == 1) {
            if (ch == '\r') {
                continue;
            }
            if (ch == '\n') {
                process_line(job_id, line);
                line.clear();
            } else {
                line.push_back(ch);
            }
        }
        if (!line.empty()) {
            process_line(job_id, line);
        }
        fclose(fp);

        int exit_code = 1;
#ifdef _WIN32
        if (hProcess) {
            WaitForSingleObject(hProcess, INFINITE);
            DWORD code = 1;
            GetExitCodeProcess(hProcess, &code);
            exit_code = static_cast<int>(code);
            CloseHandle(hProcess);
            CloseHandle(hThread);
            hProcess = nullptr;
            hThread = nullptr;
        }
#else
        if (pid > 0) {
            int st = 0;
            waitpid(pid, &st, 0);
            if (WIFEXITED(st)) {
                exit_code = WEXITSTATUS(st);
            }
            pid = -1;
        }
#endif
        fs::remove(script_path, ec);
        if (!pipeline_cfg_on_disk.empty()) {
            fs::remove(pipeline_cfg_on_disk, ec);
        }
        if (user_cancel.load()) {
            exit_code = 130;
            user_cancel = false;
        }
        env_unset_dist();
        return exit_code;
    }

    void cancel() {
        user_cancel = true;
        terminate_child();
    }
};

static std::mutex g_train_mtx;
static std::shared_ptr<TrainingProc> g_active_train;

static void handle_message(const std::string& raw);

static void on_ws_message(const ix::WebSocketMessagePtr& msg) {
    if (msg->type == ix::WebSocketMessageType::Message) {
        handle_message(msg->str);
    } else if (msg->type == ix::WebSocketMessageType::Close) {
        std::lock_guard<std::mutex> lk(g_train_mtx);
        if (g_active_train) {
            g_active_train->cancel();
        }
    } else if (msg->type == ix::WebSocketMessageType::Error) {
        std::cerr << "[worker] ws error\n";
    }
}

static void handle_message(const std::string& raw) {
    json msg;
    try {
        msg = json::parse(raw);
    } catch (...) {
        return;
    }
    if (!msg.contains("type")) {
        return;
    }
    std::string type = msg["type"].get<std::string>();
    if (type == "ping") {
        ws_send_text(R"({"type":"pong"})");
        return;
    }
    if (type == "cancel_job") {
        std::lock_guard<std::mutex> lk(g_train_mtx);
        if (g_active_train) {
            g_active_train->cancel();
        }
        return;
    }
    if (type == "sync") {
        std::cerr << "[worker] sync job=" << msg.value("job_id", std::string())
                  << " role=" << msg.value("role", std::string()) << "\n";
        return;
    }
    if (type != "run_job") {
        return;
    }
    std::string job_id = msg["job_id"].get<std::string>();
    std::string script = msg["script"].get<std::string>();
    bool resume = msg.value("resume", false);
    std::string ckpt = msg.value("checkpoint_path", std::string());
    int start_step = msg.value("start_step", 0);
    std::string py = getenv_str("PYTHON", "python");
    json dist = json::object();
    if (msg.contains("distributed") && msg["distributed"].is_object()) {
        dist = msg["distributed"];
    }

    bool pipeline_flag = false;
    if (msg.contains("pipeline_enabled")) {
        const auto& pe = msg["pipeline_enabled"];
        if (pe.is_boolean()) {
            pipeline_flag = pe.get<bool>();
        } else if (pe.is_number()) {
            pipeline_flag = (pe.get<int>() != 0);
        }
    }
    std::string pipeline_yaml;
    if (pipeline_flag && msg.contains("pipeline_config_yaml") &&
        msg["pipeline_config_yaml"].is_string()) {
        pipeline_yaml = msg["pipeline_config_yaml"].get<std::string>();
    }
    const bool pipeline_active = pipeline_flag && !pipeline_yaml.empty();
    int pipe_stage = 0;
    if (dist.is_object()) {
        pipe_stage = dist.value("rank", 0);
    }
    if (msg.contains("pipeline_stage_idx") &&
        msg["pipeline_stage_idx"].is_number_integer()) {
        pipe_stage = msg["pipeline_stage_idx"].get<int>();
    }

    std::thread([job_id, script, resume, ckpt, start_step, py, dist,
                 pipeline_active, pipeline_yaml, pipe_stage]() {
        auto proc = std::make_shared<TrainingProc>();
        {
            std::lock_guard<std::mutex> lk(g_train_mtx);
            g_active_train = proc;
        }
        std::cerr << "[worker] run_job " << job_id << "\n";
        const int code = proc->run_python_script(
            job_id, script, resume, ckpt, start_step, py, dist, pipeline_active,
            pipeline_yaml, pipe_stage);
        json done;
        done["type"] = "job_done";
        done["job_id"] = job_id;
        done["exit_code"] = code;
        ws_send_text(done.dump());
        {
            std::lock_guard<std::mutex> lk(g_train_mtx);
            if (g_active_train == proc) {
                g_active_train.reset();
            }
        }
    }).detach();
}

int main() {
    ix::initNetSystem();

    const std::string url =
        getenv_str("SERVER_URL", "ws://127.0.0.1:8765/worker");
    const std::string token =
        getenv_str("WORKER_TOKEN", getenv_str("TOKEN", "secret-token-123"));
    const std::string worker_id = load_or_create_worker_id();

    while (true) {
        ix::WebSocket ws;
        {
            std::lock_guard<std::mutex> lk(g_ws_send_mtx);
            g_ws_ptr = &ws;
        }
        ws.setUrl(url);
        ws.setOnMessageCallback(on_ws_message);

        ws.start();

        for (int i = 0; i < 200; ++i) {
            if (ws.getReadyState() == ix::ReadyState::Open) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        if (ws.getReadyState() != ix::ReadyState::Open) {
            std::cerr << "[worker] не удалось подключиться, пауза 5с\n";
            ws.stop();
            std::lock_guard<std::mutex> lk(g_ws_send_mtx);
            g_ws_ptr = nullptr;
            std::this_thread::sleep_for(std::chrono::seconds(5));
            continue;
        }

        json reg;
        reg["type"] = "register";
        reg["token"] = token;
        reg["worker_id"] = worker_id;
        reg["gpu"] = query_gpu_nvidia_smi();
        std::string adv = getenv_str("DISTGPU_ADVERTISE_HOST");
        if (!adv.empty()) {
            reg["advertise_host"] = adv;
        }
        ws.sendText(reg.dump());
        std::cerr << "[worker] подключено id=" << worker_id << "\n";

        while (ws.getReadyState() == ix::ReadyState::Open) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        std::cerr << "[worker] соединение закрыто, пауза 5с\n";
        ws.stop();
        {
            std::lock_guard<std::mutex> lk(g_ws_send_mtx);
            g_ws_ptr = nullptr;
        }
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }

    ix::uninitNetSystem();
    return 0;
}
