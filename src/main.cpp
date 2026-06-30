// rkllm-server: minimal HTTP wrapper over the Rockchip rknn-llm runtime.
//
// Designed for the RK3588 / RK3576 NPU (Orange Pi 5, Radxa ROCK 5, etc).
// Exposes a quantized .rkllm model (Qwen, Llama, Phi, Gemma, ...) via HTTP
// so other services on your LAN / Docker cluster can use the NPU without
// linking against librkllmrt directly.
//
// Endpoints:
//   POST /process   {"text":"..."}
//       Opinionated extractor for the bundled system prompt (a vidraçaria
//       quote-extractor by default) — runs adaptive recovery on invalid JSON.
//   POST /generate  {"system_prompt":"...","user_text":"..."}
//       Generic passthrough — caller provides the system prompt, server
//       returns raw output. No validation. Chat template is restored after.
//   GET  /health    -> uptime, decode tok/s avg, model path
//
// Override the system prompt at runtime: ./rkllm_server <model> --prompt-file <path>
//
// Adaptive recovery: rknn-llm v1.2.1 has a KV-cache state issue where shared
// handles degrade after a few turns. When /process gets invalid JSON, the
// handle is destroyed + re-initialized (~2s) and the request retried once.
// See docs/ADAPTIVE_RECOVERY.md.
//
// Build:    see CMakeLists.txt
// Runtime:  librkllmrt.so (from airockchip/rknn-llm v1.2.1) + a .rkllm model
// License:  Apache-2.0

#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "rkllm.h"
#include "httplib.h"
#include "json.hpp"

using nlohmann::json;

// ---------- Globals -----------------------------------------------------------

static LLMHandle g_handle = nullptr;
static std::string g_model_path;
static std::string g_output;
static int g_decode_tokens = 0;
static int g_prefill_tokens = 0;
static double g_decode_time_ms = 0.0;
static double g_prefill_time_ms = 0.0;
static std::mutex g_infer_mutex;
static auto g_started_at = std::chrono::steady_clock::now();
static double g_tps_sum = 0.0;
static long g_tps_n = 0;

// ---------- Signal handler ----------------------------------------------------

void exit_handler(int signal) {
    if (g_handle) { rkllm_destroy(g_handle); g_handle = nullptr; }
    std::cerr << "exiting on signal " << signal << std::endl;
    _exit(signal);
}

// ---------- rkllm callback ----------------------------------------------------

int rkllm_callback(RKLLMResult *result, void *userdata, LLMCallState state) {
    if (state == RKLLM_RUN_NORMAL) {
        if (result->text) g_output += result->text;
    } else if (state == RKLLM_RUN_FINISH) {
        if (result->perf.generate_tokens > 0) {
            g_decode_tokens = result->perf.generate_tokens;
            g_prefill_tokens = result->perf.prefill_tokens;
            g_decode_time_ms = result->perf.generate_time_ms;
            g_prefill_time_ms = result->perf.prefill_time_ms;
        }
    } else if (state == RKLLM_RUN_ERROR) {
        std::cerr << "rkllm RUN_ERROR" << std::endl;
    }
    return 0;
}

// ---------- System prompt -----------------------------------------------------
//
// Default content for the chat-template `system` block. Plain text — no chat
// markers. The wrapper below adds <|im_start|>system\n...\n<|im_end|>\n at init.
// Override at runtime: ./rkllm_server <model> --prompt-file <path>
//
// This default is a Brazilian-Portuguese glass-shop (vidraçaria) quote extractor
// — the original real-world use case. See examples/prompts/ for alternatives.

static const char* DEFAULT_SYS_PROMPT_CONTENT =
"Voce e o orcamentista de uma vidracaria. Receba a mensagem do cliente e responda APENAS com um JSON em uma linha.\n"
"\n"
"Schema OBRIGATORIO:\n"
"{\"intent\":\"quote_request|greeting|followup|other\",\"product\":<obj-ou-null>}\n"
"\n"
"Quando intent=quote_request, o product DEVE ter:\n"
"{\"type\":\"PORTA|JANELA|BOX|FIXO|SACADA|BASCULANTE\",\"width_cm\":<int>,\"height_cm\":<int>,\"sheets\":<int>,\"color\":\"INCOLOR\"}\n"
"\n"
"REGRAS DE DIMENSAO:\n"
"- Sempre em CENTIMETROS (cm)\n"
"- 1m = 100cm. \"1,20\" ou \"1,20m\" = 120cm. \"0,80\" = 80cm.\n"
"- Numeros sem unidade ja SAO em cm e devem ser usados como estao (NAO multiplique).\n"
"- \"60x40\" = 60cm x 40cm. NUNCA inflar pra 600x400.\n"
"\n"
"REGRAS DE TYPE (estrutura, nao tipo de vidro):\n"
"- \"porta\" -> PORTA\n"
"- \"janela\" -> JANELA\n"
"- \"box\" (chuveiro/banheiro) -> BOX\n"
"- \"sacada\"/\"varanda\" -> SACADA\n"
"- \"basculante\" -> BASCULANTE\n"
"- Se for um vidro/espelho sem estrutura aparente -> FIXO\n"
"\n"
"REGRAS DE INTENT:\n"
"- quote_request: cliente pede orcamento com dimensoes ou estrutura clara.\n"
"- greeting: \"oi\", \"bom dia\", \"obrigado\", \"valeu\".\n"
"- followup: \"tem em azul?\", \"qual prazo?\", \"e a cor?\", \"pode mandar?\".\n"
"- other: tudo o que nao se encaixa.\n"
"\n"
"Defaults: sheets=1, color=\"INCOLOR\".\n"
"\n"
"Exemplos:\n"
"\"vidro temperado 1,20x0,80\" => {\"intent\":\"quote_request\",\"product\":{\"type\":\"FIXO\",\"width_cm\":120,\"height_cm\":80,\"sheets\":1,\"color\":\"INCOLOR\"}}\n"
"\"espelho 60x40\" => {\"intent\":\"quote_request\",\"product\":{\"type\":\"FIXO\",\"width_cm\":60,\"height_cm\":40,\"sheets\":1,\"color\":\"INCOLOR\"}}\n"
"\"porta de vidro 2m x 80cm, 10mm\" => {\"intent\":\"quote_request\",\"product\":{\"type\":\"PORTA\",\"width_cm\":200,\"height_cm\":80,\"sheets\":1,\"color\":\"INCOLOR\"}}\n"
"\"box de banheiro 1,80m x 90cm temperado 8mm\" => {\"intent\":\"quote_request\",\"product\":{\"type\":\"BOX\",\"width_cm\":180,\"height_cm\":90,\"sheets\":1,\"color\":\"INCOLOR\"}}\n"
"\"oi bom dia\" => {\"intent\":\"greeting\",\"product\":null}\n"
"\"obrigado\" => {\"intent\":\"greeting\",\"product\":null}\n"
"\"tem em azul?\" => {\"intent\":\"followup\",\"product\":null}\n"
"\"qual o prazo de entrega?\" => {\"intent\":\"followup\",\"product\":null}";

// Populated at startup with the wrapped chat template (default OR --prompt-file).
static std::string g_sys_prompt;

static std::string wrap_chat_template(const std::string& content) {
    return "<|im_start|>system\n" + content + "\n<|im_end|>\n";
}

static std::string load_prompt_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::cerr << "ERROR: cannot read prompt file: " << path << std::endl;
        std::exit(2);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
        s.pop_back();
    return s;
}

// ---------- rkllm lifecycle ---------------------------------------------------

int init_model(const std::string& model_path) {
    RKLLMParam p = rkllm_createDefaultParam();
    p.model_path = model_path.c_str();
    p.top_k = 1;
    p.top_p = 1.0;
    p.temperature = 0.0;
    p.repeat_penalty = 1.0;
    p.max_new_tokens = 120;
    p.max_context_len = 1024;
    p.skip_special_token = true;
    p.extend_param.base_domain_id = 0;
    p.extend_param.embed_flash = 1;
    int ret = rkllm_init(&g_handle, &p, rkllm_callback);
    if (ret != 0) return ret;
    rkllm_set_chat_template(g_handle, g_sys_prompt.c_str(),
                            "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n");
    return 0;
}

void destroy_model() {
    if (g_handle) { rkllm_destroy(g_handle); g_handle = nullptr; }
}

// ---------- inference --------------------------------------------------------

struct InferResult {
    std::string raw_output;
    double wall_ms;
    double decode_tps;
    double prefill_tps;
    int decode_tokens;
    int prefill_tokens;
};

bool run_inference(const std::string& user_text, InferResult& out) {
    g_output.clear();
    g_decode_tokens = 0;
    g_prefill_tokens = 0;
    g_decode_time_ms = 0;
    g_prefill_time_ms = 0;

    RKLLMInput in;
    memset(&in, 0, sizeof(in));
    in.input_type = RKLLM_INPUT_PROMPT;
    in.role = "user";
    in.prompt_input = (char*)user_text.c_str();

    RKLLMInferParam ip;
    memset(&ip, 0, sizeof(ip));
    ip.mode = RKLLM_INFER_GENERATE;
    ip.keep_history = 0;

    auto t0 = std::chrono::steady_clock::now();
    int ret = rkllm_run(g_handle, &in, &ip, nullptr);
    auto t1 = std::chrono::steady_clock::now();
    out.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    out.raw_output = g_output;
    out.decode_tokens = g_decode_tokens;
    out.prefill_tokens = g_prefill_tokens;
    out.decode_tps = (g_decode_time_ms > 0) ? (g_decode_tokens / (g_decode_time_ms / 1000.0)) : 0.0;
    out.prefill_tps = (g_prefill_time_ms > 0) ? (g_prefill_tokens / (g_prefill_time_ms / 1000.0)) : 0.0;
    return ret == 0 && !g_output.empty();
}

// ---------- output validation + parse ----------------------------------------

static bool extract_json_object(const std::string& raw, std::string& json_out) {
    auto start = raw.find('{');
    if (start == std::string::npos) return false;
    int depth = 0;
    bool in_str = false;
    bool esc = false;
    for (size_t i = start; i < raw.size(); ++i) {
        char c = raw[i];
        if (esc) { esc = false; continue; }
        if (c == '\\') { esc = true; continue; }
        if (c == '"') { in_str = !in_str; continue; }
        if (in_str) continue;
        if (c == '{') depth++;
        else if (c == '}') {
            depth--;
            if (depth == 0) {
                json_out = raw.substr(start, i - start + 1);
                return true;
            }
        }
    }
    return false;
}

static bool is_valid_intent_response(const json& j) {
    if (!j.contains("intent") || !j["intent"].is_string()) return false;
    std::string intent = j["intent"];
    if (intent != "quote_request" && intent != "greeting" &&
        intent != "followup" && intent != "other") return false;
    if (intent == "quote_request") {
        if (!j.contains("product") || j["product"].is_null()) return false;
        const auto& p = j["product"];
        if (!p.contains("type") || !p.contains("width_cm") || !p.contains("height_cm")) return false;
        std::string t = p["type"];
        if (t != "PORTA" && t != "JANELA" && t != "BOX" &&
            t != "FIXO" && t != "SACADA" && t != "BASCULANTE") return false;
        int w = p["width_cm"].get<int>();
        int h = p["height_cm"].get<int>();
        if (w < 5 || w > 600 || h < 5 || h > 600) return false;
    }
    return true;
}

// ---------- adaptive recovery ------------------------------------------------

struct ProcessOutcome {
    json parsed;       // valid parsed json
    InferResult perf;
    bool reinit_used;
};

bool process_with_recovery(const std::string& user_text, ProcessOutcome& outcome, std::string& err) {
    std::lock_guard<std::mutex> lock(g_infer_mutex);
    outcome.reinit_used = false;

    InferResult ir;
    bool ok = run_inference(user_text, ir);
    outcome.perf = ir;
    if (!ok) { err = "rkllm_run returned error or empty"; }

    // attempt parse
    std::string js;
    json parsed;
    bool parsed_ok = false;
    if (extract_json_object(ir.raw_output, js)) {
        try { parsed = json::parse(js); parsed_ok = is_valid_intent_response(parsed); }
        catch (...) { parsed_ok = false; }
    }
    if (parsed_ok) {
        outcome.parsed = parsed;
        if (ir.decode_tps > 0) { g_tps_sum += ir.decode_tps; g_tps_n++; }
        return true;
    }

    // recovery: re-init + retry once
    std::cerr << "invalid output, re-initing handle and retrying. raw: " << ir.raw_output.substr(0, 200) << std::endl;
    destroy_model();
    if (init_model(g_model_path) != 0) {
        err = "re-init failed";
        return false;
    }
    InferResult ir2;
    ok = run_inference(user_text, ir2);
    outcome.perf = ir2;
    outcome.reinit_used = true;
    if (!ok) { err = "retry rkllm_run failed"; return false; }
    if (extract_json_object(ir2.raw_output, js)) {
        try {
            parsed = json::parse(js);
            if (is_valid_intent_response(parsed)) {
                outcome.parsed = parsed;
                if (ir2.decode_tps > 0) { g_tps_sum += ir2.decode_tps; g_tps_n++; }
                return true;
            }
        } catch (...) {}
    }
    err = "invalid output after retry. raw: " + ir2.raw_output.substr(0, 200);
    return false;
}

// ---------- HTTP handlers ----------------------------------------------------

void handle_process(const httplib::Request& req, httplib::Response& res) {
    json req_json;
    try { req_json = json::parse(req.body); }
    catch (...) {
        res.status = 400;
        res.set_content("{\"error\":\"invalid JSON body\"}", "application/json");
        return;
    }
    if (!req_json.contains("text") || !req_json["text"].is_string()) {
        res.status = 400;
        res.set_content("{\"error\":\"missing 'text' field\"}", "application/json");
        return;
    }
    std::string text = req_json["text"];
    if (text.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"empty text\"}", "application/json");
        return;
    }
    if (text.size() > 1024) text = text.substr(0, 1024);

    ProcessOutcome outcome;
    std::string err;
    auto t0 = std::chrono::steady_clock::now();
    bool ok = process_with_recovery(text, outcome, err);
    auto t1 = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    json resp;
    resp["total_latency_ms"] = std::round(total_ms);
    resp["decode_tps"] = outcome.perf.decode_tps;
    resp["prefill_tps"] = outcome.perf.prefill_tps;
    resp["decode_tokens"] = outcome.perf.decode_tokens;
    resp["prefill_tokens"] = outcome.perf.prefill_tokens;
    resp["reinit_used"] = outcome.reinit_used;
    resp["raw_output"] = outcome.perf.raw_output;
    if (ok) {
        resp["intent"] = outcome.parsed["intent"];
        resp["product"] = outcome.parsed.value("product", json(nullptr));
        res.status = 200;
    } else {
        resp["intent"] = "other";
        resp["product"] = nullptr;
        resp["error"] = err;
        res.status = 200;  // still 200; orchestrator decides what to do
    }
    res.set_content(resp.dump(), "application/json");
}

// Generic generation: caller provides system_prompt + user_text.
// Restores the default chat template after — /process keeps working unchanged.
void handle_generate(const httplib::Request& req, httplib::Response& res) {
    json req_json;
    try { req_json = json::parse(req.body); }
    catch (...) {
        res.status = 400;
        res.set_content("{\"error\":\"invalid JSON body\"}", "application/json");
        return;
    }
    if (!req_json.contains("system_prompt") || !req_json["system_prompt"].is_string()
        || !req_json.contains("user_text") || !req_json["user_text"].is_string()) {
        res.status = 400;
        res.set_content("{\"error\":\"missing system_prompt or user_text\"}", "application/json");
        return;
    }
    std::string sys = req_json["system_prompt"];
    std::string user_text = req_json["user_text"];
    if (sys.empty() || user_text.empty()) {
        res.status = 400;
        res.set_content("{\"error\":\"empty system_prompt or user_text\"}", "application/json");
        return;
    }
    if (sys.size() > 4096)       sys = sys.substr(0, 4096);
    if (user_text.size() > 2048) user_text = user_text.substr(0, 2048);

    std::lock_guard<std::mutex> lock(g_infer_mutex);

    std::string custom_template = "<|im_start|>system\n" + sys + "\n<|im_end|>\n";
    rkllm_set_chat_template(g_handle, custom_template.c_str(),
                            "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n");

    auto t0 = std::chrono::steady_clock::now();
    InferResult ir;
    bool ok = run_inference(user_text, ir);
    auto t1 = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Restore the default chat template so /process keeps its configured system prompt.
    rkllm_set_chat_template(g_handle, g_sys_prompt.c_str(),
                            "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n");

    json resp;
    resp["ok"] = ok;
    resp["raw_output"] = ir.raw_output;
    resp["total_latency_ms"] = std::round(total_ms);
    resp["decode_tps"] = ir.decode_tps;
    resp["prefill_tps"] = ir.prefill_tps;
    resp["decode_tokens"] = ir.decode_tokens;
    resp["prefill_tokens"] = ir.prefill_tokens;
    res.status = 200;
    res.set_content(resp.dump(), "application/json");
}

void handle_health(const httplib::Request& req, httplib::Response& res) {
    auto now = std::chrono::steady_clock::now();
    double uptime_s = std::chrono::duration<double>(now - g_started_at).count();
    json resp;
    resp["status"] = "ok";
    resp["uptime_s"] = std::round(uptime_s);
    resp["model_path"] = g_model_path;
    resp["decode_tps_avg"] = (g_tps_n > 0) ? (g_tps_sum / g_tps_n) : 0.0;
    resp["inferences_completed"] = (long)g_tps_n;
    res.set_content(resp.dump(), "application/json");
}

// ---------- main -------------------------------------------------------------

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <model.rkllm> [options]\n"
              << "\n"
              << "Required:\n"
              << "  <model.rkllm>          Path to a quantized .rkllm model file\n"
              << "\n"
              << "Options:\n"
              << "  --port <N>             HTTP port to listen on (default: 18080)\n"
              << "  --prompt-file <path>   Plain-text file with the system prompt content\n"
              << "                         (default: bundled vidraçaria example)\n"
              << "  -h, --help             Print this message\n"
              << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    int port = 18080;
    std::string prompt_file_path;

    // First positional arg = model path. Remaining = flags.
    g_model_path = argv[1];
    if (g_model_path == "-h" || g_model_path == "--help") {
        print_usage(argv[0]);
        return 0;
    }

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--port" || arg == "-p") && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (arg == "--prompt-file" && i + 1 < argc) {
            prompt_file_path = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (i == 2 && std::all_of(arg.begin(), arg.end(), ::isdigit)) {
            // Backwards compat: <model> <port> positional
            port = std::atoi(arg.c_str());
        } else {
            std::cerr << "ERROR: unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    std::string sys_content = prompt_file_path.empty()
        ? std::string(DEFAULT_SYS_PROMPT_CONTENT)
        : load_prompt_file(prompt_file_path);
    g_sys_prompt = wrap_chat_template(sys_content);

    signal(SIGINT, exit_handler);
    signal(SIGTERM, exit_handler);

    std::cerr << "rkllm-server starting:\n"
              << "  model:       " << g_model_path << "\n"
              << "  port:        " << port << "\n"
              << "  prompt:      " << (prompt_file_path.empty() ? "<built-in default>" : prompt_file_path) << "\n"
              << "  prompt size: " << sys_content.size() << " bytes\n";

    if (init_model(g_model_path) != 0) {
        std::cerr << "ERROR: rkllm_init failed" << std::endl;
        return 1;
    }
    std::cerr << "model loaded — HTTP server listening on 0.0.0.0:" << port << std::endl;

    httplib::Server svr;
    // serialize inference (rkllm runtime is not thread-safe)
    svr.new_task_queue = [] { return new httplib::ThreadPool(1); };
    svr.Post("/process", handle_process);
    svr.Post("/generate", handle_generate);
    svr.Get("/health", handle_health);
    svr.set_read_timeout(60, 0);
    svr.set_write_timeout(60, 0);

    if (!svr.listen("0.0.0.0", port)) {
        std::cerr << "failed to bind port " << port << std::endl;
        destroy_model();
        return 1;
    }
    destroy_model();
    return 0;
}
