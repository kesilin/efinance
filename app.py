import streamlit as st
import plotly.graph_objects as go
import requests
import json
import re
import time
from pathlib import Path
from utils import *
from fund_analyzer import UniversalFundAnalyzer, generate_gemini_prompt


def parse_change_value(change):
    """Parse change values like '2.56%' safely into float."""
    if isinstance(change, (int, float)):
        return float(change)
    if isinstance(change, str):
        cleaned = change.replace("%", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def build_analysis_payload(analysis):
    """Build a compact analysis payload for API prompt refinement."""
    return {
        "基金基础信息": analysis.get("基金基础信息", {}),
        "实时行情数据": analysis.get("实时行情数据", {}),
        "最近7个交易日单日涨跌明细": analysis.get("最近7个交易日单日涨跌明细", []),
        "波段周期累计表现": analysis.get("波段周期累计表现", {}),
        "波段择时指标": analysis.get("波段择时指标", {}),
        "持仓结构分析": analysis.get("持仓结构分析", {}),
        "基准对比分析": analysis.get("基准对比分析", {}),
    }


PROVIDER_CONFIG = {
    "Gemini": {
        "api_base": "https://generativelanguage.googleapis.com/v1beta",
        "models": [
            "gemini-3-flash",
            "gemma-3-1b-it",
            "gemini-2.5-pro-latest",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "自定义模型",
        ],
        "key_placeholder": "AIza...",
    },
    "DeepSeek": {
        "api_base": "https://api.deepseek.com/v1",
        "models": [
            "deepseek-chat",
            "deepseek-reasoner",
            "自定义模型",
        ],
        "key_placeholder": "sk-...",
    },
    "豆包": {
        "api_base": "https://ark.cn-beijing.volces.com/api/v3",
        "models": [
            "doubao-1-5-lite-32k-250115",
            "doubao-1-5-pro-32k-250115",
            "ep-自定义EndpointID",
            "自定义模型",
        ],
        "key_placeholder": "ark-...",
    },
}

MEMORY_FILE = Path("data") / "investor_memory.jsonl"
LOCAL_SECRETS_FILE = Path(__file__).resolve().parent / ".secrets.local.json"

GEMINI_MODEL_ALIASES = {
    "gemini-3-flash": "gemini-2.0-flash",
    "gemini-3": "gemini-2.0-flash",
}

STRATEGY_PRESET_FILE = Path("data") / "strategy_presets.json"

BASE_ACTION_OPTIONS = ["加仓", "减仓", "退仓", "策略"]
BASE_RISK_OPTIONS = ["保守", "平衡", "激进"]
BASE_ASSET_OPTIONS = ["基金为主", "股票为主", "基金+股票联动"]
BASE_SIGNAL_OPTIONS = ["技术面优先", "基本面优先", "政策+资金面", "综合研判"]

BASE_QUICK_REQUIREMENT_PHRASES = [
    "请明确分批仓位比例，并给出每一笔的触发条件。",
    "请把止损写成可执行规则，包含触发阈值和动作。",
    "请给出主情景/备选情景，并评估成功概率。",
    "请把持有周期控制在7-30天，并标注最佳执行窗口。",
    "请增加政策与行业催化因素的影响说明。",
    "若无法实时联网，请基于已提供资料完成策略判断。",
]


def _dedup_keep_order(items):
    seen = set()
    out = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def load_strategy_presets():
    default_data = {
        "action": [],
        "risk": [],
        "asset": [],
        "signal": [],
        "quick_phrases": [],
    }
    try:
        if not STRATEGY_PRESET_FILE.exists():
            return default_data
        data = json.loads(STRATEGY_PRESET_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return default_data
        for k in default_data.keys():
            if not isinstance(data.get(k), list):
                data[k] = []
            data[k] = _dedup_keep_order(data[k])
        return data
    except Exception:
        return default_data


def save_strategy_presets(data):
    payload = {
        "action": _dedup_keep_order(data.get("action", [])),
        "risk": _dedup_keep_order(data.get("risk", [])),
        "asset": _dedup_keep_order(data.get("asset", [])),
        "signal": _dedup_keep_order(data.get("signal", [])),
        "quick_phrases": _dedup_keep_order(data.get("quick_phrases", [])),
    }
    STRATEGY_PRESET_FILE.parent.mkdir(parents=True, exist_ok=True)
    STRATEGY_PRESET_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_strategy_preset_state():
    if "strategy_presets" not in st.session_state:
        st.session_state["strategy_presets"] = load_strategy_presets()


def _get_options_with_plus(base_options, preset_key):
    ensure_strategy_preset_state()
    custom = st.session_state["strategy_presets"].get(preset_key, [])
    merged = _dedup_keep_order([*base_options, *custom])
    return [*merged, "➕ 新增自定义选项"]


def add_custom_strategy_option(preset_key, value):
    ensure_strategy_preset_state()
    text = str(value or "").strip()
    if not text:
        return False
    current = st.session_state["strategy_presets"].get(preset_key, [])
    st.session_state["strategy_presets"][preset_key] = _dedup_keep_order([*current, text])
    save_strategy_presets(st.session_state["strategy_presets"])
    return True


def remove_custom_strategy_options(preset_key, values):
    ensure_strategy_preset_state()
    current = _dedup_keep_order(st.session_state["strategy_presets"].get(preset_key, []))
    remove_set = {str(v or "").strip() for v in (values or []) if str(v or "").strip()}
    if not remove_set:
        return 0
    kept = [item for item in current if item not in remove_set]
    removed_count = len(current) - len(kept)
    if removed_count > 0:
        st.session_state["strategy_presets"][preset_key] = kept
        save_strategy_presets(st.session_state["strategy_presets"])
    return removed_count


def build_requirement_template(action, risk_style, asset_view, signal_focus):
    focus = {
        "加仓": "分批加仓",
        "减仓": "分批减仓",
        "退仓": "分步退仓",
        "策略": "策略评估",
    }.get(action, "策略评估")

    risk_desc = {
        "保守": "回撤优先，目标稳健，单笔风险控制更严格",
        "平衡": "收益与回撤并重，分批执行，动态调整",
        "激进": "收益优先，可接受更高波动，但必须有明确止损",
    }.get(risk_style, "收益与回撤并重")

    asset_desc = {
        "基金为主": "以基金净值与历史波动为核心，股票信号作为辅助",
        "股票为主": "以行业股票趋势为主，基金作为执行载体",
        "基金+股票联动": "同时评估基金与相关股票/行业指数联动关系",
    }.get(asset_view, "以基金与股票联动关系综合评估")

    signal_desc = {
        "技术面优先": "优先使用短中期趋势、RSI、回撤、量价节奏",
        "基本面优先": "优先考虑行业景气、估值、盈利预期与资金面",
        "政策+资金面": "优先考虑政策催化、事件冲击、资金流向和市场情绪",
        "综合研判": "融合技术面、基本面、政策与资金面多维判断",
    }.get(signal_focus, "融合多维度信号")

    return (
        f"请基于{risk_style}风格的{focus}策略，分析该标的未来7-30天操作可能性。\n"
        f"标的视角：{asset_desc}。\n"
        f"信号偏好：{signal_desc}。\n"
        f"风控原则：{risk_desc}。\n"
        "请输出：1) 是否执行当前策略；2) 关键依据（数据+策略逻辑）；"
        "3) 分步计划（仓位、止盈、止损、时间节点）；4) 策略达成概率（高/中/低）。\n"
        "若无法实时联网检索，请基于已提供政策/资讯材料进行判断。"
    )


def build_investor_snapshot(analysis):
    """Generate compact investor-oriented status summary for quick decision making."""
    perf = analysis.get("波段周期累计表现", {})
    timing = analysis.get("波段择时指标", {})

    p7 = perf.get("近7个交易日", {}).get("区间涨幅", "无数据")
    p30 = perf.get("近30个交易日", {}).get("区间涨幅", "无数据")
    dd30 = perf.get("近30个交易日", {}).get("最大回撤", "无数据")
    rsi = timing.get("RSI(14)", "无数据")

    def _fnum(v):
        try:
            return float(v)
        except Exception:
            return None

    p7f = _fnum(p7)
    p30f = _fnum(p30)
    dd30f = _fnum(dd30)
    rsif = _fnum(rsi)

    momentum = "震荡"
    if p7f is not None and p30f is not None:
        if p7f > 0 and p30f > 0:
            momentum = "偏强"
        elif p7f < 0 and p30f < 0:
            momentum = "偏弱"

    risk = "中"
    if dd30f is not None:
        if dd30f <= -8:
            risk = "高"
        elif dd30f >= -3:
            risk = "低"

    timing_tag = "中性"
    if rsif is not None:
        if rsif >= 70:
            timing_tag = "高位"
        elif rsif <= 30:
            timing_tag = "低位"

    return {
        "趋势状态": momentum,
        "短期风险": risk,
        "择时区间": timing_tag,
        "7日涨幅": p7,
        "30日涨幅": p30,
        "30日回撤": dd30,
    }


def _mask_key(api_key):
    key = str(api_key or "")
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def terminal_log(event, **kwargs):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if kwargs:
        detail = " | " + " ".join(f"{k}={v}" for k, v in kwargs.items())
    else:
        detail = ""
    print(f"[WEB-AI][{ts}] {event}{detail}", flush=True)


def build_http_session(proxy_url="", use_system_proxy=False):
    session = requests.Session()
    if proxy_url:
        session.trust_env = False
        session.proxies.update({"http": proxy_url, "https": proxy_url})
        return session
    if use_system_proxy:
        session.trust_env = True
        return session
    session.trust_env = False
    session.proxies.update({"http": "", "https": ""})
    return session


def get_runtime_network_options():
    proxy_url = st.session_state.get("net_proxy_url", "").strip()
    use_system_proxy = bool(st.session_state.get("net_use_system_proxy", False))
    # Keep behavior deterministic: explicit proxy always wins over system proxy.
    if proxy_url:
        use_system_proxy = False
    return {
        "proxy_url": proxy_url,
        "use_system_proxy": use_system_proxy,
        "connect_timeout": int(st.session_state.get("net_connect_timeout", 6)),
        "read_timeout": int(st.session_state.get("net_read_timeout", 25)),
        "require_country": st.session_state.get("net_require_country", "").strip().upper(),
        "allow_unknown_country": bool(st.session_state.get("net_allow_unknown_country", True)),
    }


def detect_exit_geo(session, timeout_sec=8):
    urls = [
        "https://ipapi.co/json/",
        "https://ipinfo.io/json",
        "https://ipwho.is/",
        "https://api.ip.sb/geoip",
    ]
    for url in urls:
        try:
            resp = session.get(url, timeout=(5, max(6, timeout_sec)))
            if not resp.ok:
                continue
            data = resp.json()
            ip = str(data.get("ip", "未知"))
            country_code = str(
                data.get("country_code")
                or data.get("countryCode")
                or data.get("country")
                or "未知"
            ).upper()
            country_name = str(
                data.get("country_name")
                or data.get("country")
                or data.get("country_name_en")
                or "未知"
            )
            if len(country_code) != 2 and country_code not in ("未知", ""):
                country_code = "未知"
            return {
                "ip": ip,
                "country_code": country_code,
                "country_name": country_name,
                "source": url,
            }
        except Exception:
            continue
    return {
        "ip": "未知",
        "country_code": "未知",
        "country_name": "未知",
        "source": "unavailable",
    }


def validate_exit_country_or_raise(session, options):
    geo = detect_exit_geo(session, timeout_sec=options["connect_timeout"])
    terminal_log(
        "Exit geo",
        ip=geo["ip"],
        country=geo["country_name"],
        cc=geo["country_code"],
        source=geo["source"],
    )
    required = options["require_country"]
    if not required:
        return geo
    unknown = geo["country_code"] in ("未知", "")
    if unknown and options["allow_unknown_country"]:
        terminal_log("Exit country unknown but allowed", require=required)
        return geo
    if geo["country_code"] != required:
        raise RuntimeError("出口地区校验失败：要求 {}，实际 {}".format(required, geo["country_code"]))
    return geo


def get_request_timeout(options):
    return (max(3, options["connect_timeout"]), max(8, options["read_timeout"]))


def get_runtime_gemini_generation_options():
    return {
        "temperature": float(st.session_state.get("gemini_temperature", 0.25)),
        "top_p": float(st.session_state.get("gemini_top_p", 0.9)),
        "top_k": int(st.session_state.get("gemini_top_k", 40)),
        "max_output_tokens": int(st.session_state.get("gemini_max_output_tokens", 4096)),
        "max_retry_models": int(st.session_state.get("gemini_max_retry_models", 5)),
        "rewrite_min_chars": int(st.session_state.get("gemini_rewrite_min_chars", 280)),
        "extra_instruction": str(st.session_state.get("gemini_extra_instruction", "")).strip(),
        "output_font_px": int(st.session_state.get("ui_output_font_px", 16)),
    }


def ensure_memory_file():
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text("", encoding="utf-8")


def save_investor_memory(record):
    ensure_memory_file()
    with MEMORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_recent_memories(limit=8):
    ensure_memory_file()
    lines = MEMORY_FILE.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return []
    items = []
    for line in lines[-limit:]:
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return items


def ensure_local_secrets_file():
    if not LOCAL_SECRETS_FILE.exists():
        LOCAL_SECRETS_FILE.write_text("{}", encoding="utf-8")


def load_local_secrets():
    ensure_local_secrets_file()
    try:
        return json.loads(LOCAL_SECRETS_FILE.read_text(encoding="utf-8") or "{}")
    except Exception:
        return {}


def save_local_secrets(data):
    ensure_local_secrets_file()
    LOCAL_SECRETS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_saved_api_key(provider):
    secrets = load_local_secrets()
    return str(secrets.get("api_keys", {}).get(provider, ""))


def save_api_key(provider, api_key):
    secrets = load_local_secrets()
    if "api_keys" not in secrets:
        secrets["api_keys"] = {}
    secrets["api_keys"][provider] = api_key
    save_local_secrets(secrets)


def clear_api_key(provider):
    secrets = load_local_secrets()
    if "api_keys" in secrets and provider in secrets["api_keys"]:
        del secrets["api_keys"][provider]
    save_local_secrets(secrets)


def resolve_gemini_model_name(model_name):
    return GEMINI_MODEL_ALIASES.get(model_name, model_name)


def call_gemini_api(prompt_text, api_key, model_name, system_text="", session=None, timeout=None):
    options = get_runtime_network_options()
    runtime_session = session or build_http_session(options["proxy_url"], options["use_system_proxy"])
    runtime_timeout = timeout or get_request_timeout(options)
    tune = get_runtime_gemini_generation_options()
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    terminal_log(
        "Gemini request start",
        model=model_name,
        prompt_len=len(prompt_text),
        has_system=bool(system_text),
        key=_mask_key(api_key),
    )
    start_t = time.time()
    contents = [{"role": "user", "parts": [{"text": prompt_text}]}]
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": tune["temperature"],
            "topP": tune["top_p"],
            "topK": tune["top_k"],
            "maxOutputTokens": tune["max_output_tokens"],
        },
    }

    # Gemma family may not support systemInstruction; inline system guidance into user content.
    use_system_instruction = not str(model_name).lower().startswith("gemma-")
    if system_text and use_system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_text}]}
    elif system_text and not use_system_instruction:
        contents[0]["parts"][0]["text"] = f"[SYSTEM]\n{system_text}\n\n[USER]\n{prompt_text}"

    resp = runtime_session.post(endpoint, json=payload, timeout=runtime_timeout)
    terminal_log("Gemini response", model=model_name, status=resp.status_code, elapsed=f"{time.time()-start_t:.2f}s")
    resp.raise_for_status()
    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError("Gemini API未返回候选结果")
    parts = candidates[0].get("content", {}).get("parts", [])
    content = "\n".join(p.get("text", "") for p in parts if p.get("text"))
    if not content.strip():
        raise ValueError("Gemini API返回为空")
    return content.strip()


def call_gemini_with_retry(prompt_text, api_key, model_name, zh_only=False):
    options = get_runtime_network_options()
    tune = get_runtime_gemini_generation_options()
    request_session = build_http_session(options["proxy_url"], options["use_system_proxy"])
    request_timeout = get_request_timeout(options)
    terminal_log(
        "Network options",
        proxy=(options["proxy_url"] or "none"),
        use_system_proxy=options["use_system_proxy"],
        connect_timeout=options["connect_timeout"],
        read_timeout=options["read_timeout"],
        mode=("explicit" if options["proxy_url"] else ("system" if options["use_system_proxy"] else "direct")),
        require_country=(options["require_country"] or "none"),
        allow_unknown_country=options["allow_unknown_country"],
        temperature=tune["temperature"],
        top_p=tune["top_p"],
        max_output_tokens=tune["max_output_tokens"],
    )
    validate_exit_country_or_raise(request_session, options)

    primary = resolve_gemini_model_name(model_name)
    queue = [primary]
    for cand in ["gemma-3-1b-it", "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro-latest"]:
        if cand not in queue:
            queue.append(cand)

    system_text = ""
    if zh_only:
        system_text = "仅使用简体中文回答。保留数字、百分比、日期、基金代码原样。"
    if tune["extra_instruction"]:
        system_text = (system_text + "\n" + tune["extra_instruction"]).strip()

    last_err = None
    used_model = primary
    max_retry_models = max(1, min(6, tune["max_retry_models"]))
    terminal_log("Gemini retry queue", primary=primary, queue=queue[:max_retry_models], zh_only=zh_only)
    for cand in queue[:max_retry_models]:
        try:
            terminal_log("Gemini attempt", model=cand)
            text = call_gemini_api(
                prompt_text,
                api_key=api_key,
                model_name=cand,
                system_text=system_text,
                session=request_session,
                timeout=request_timeout,
            )
            terminal_log("Gemini attempt success", model=cand, answer_len=len(text))
            return text, cand
        except requests.exceptions.HTTPError as exc:
            used_model = cand
            code = exc.response.status_code if exc.response is not None else -1
            last_err = exc
            terminal_log("Gemini HTTPError", model=cand, code=code)
            if code in (403, 404, 429):
                continue
            raise
        except Exception as exc:
            last_err = exc
            used_model = cand
            terminal_log("Gemini Exception", model=cand, error_type=type(exc).__name__, error=repr(exc)[:320])
            continue

    if last_err:
        raise last_err
    raise RuntimeError(f"Gemini调用失败，最后尝试模型：{used_model}")


def is_fund_wave_request(text):
    src = str(text or "")
    keys = ["波段", "基金", "核心规则", "止盈", "止损"]
    return sum(1 for k in keys if k in src) >= 3


def is_fund_wave_answer_valid(text):
    val = str(text or "")
    min_chars = int(st.session_state.get("gemini_rewrite_min_chars", 280))
    if len(val.strip()) < min_chars:
        return False
    options = ["适合申购", "暂时不建议申购", "完全不建议申购"]
    sections = ["明确结论", "核心理由", "波段操作计划"]
    return any(o in val for o in options) and all(s in val for s in sections)


def rewrite_fund_wave_answer(provider, api_key, model_name, api_base, source_question, draft_text, zh_only=True):
    min_chars = int(st.session_state.get("gemini_rewrite_min_chars", 280))
    prompt = (
        "请将下面草稿改写为严格三段格式，不要输出额外内容。\n"
        "1. 明确结论：只能是 适合申购 / 暂时不建议申购 / 完全不建议申购 之一\n"
        "2. 核心理由：给出3条，每条必须结合最近7天单日涨跌、累计周期表现、择时指标、赛道逻辑、交易成本\n"
        "3. 波段操作计划：必须给出止盈位、止损位、建议持有周期、最佳赎回时间节点，并写清触发条件\n"
        f"要求：中文输出，不少于{min_chars}字；数字、百分比、价格位保留。\n\n"
        f"[原问题]\n{source_question}\n\n[草稿]\n{draft_text}"
    )
    if provider == "Gemini":
        fixed, _ = call_gemini_with_retry(prompt, api_key=api_key, model_name=model_name, zh_only=zh_only)
        return fixed
    fixed = call_openai_compatible_chat(
        messages=[
            {"role": "system", "content": "你是中文基金投顾编辑器，只输出最终改写文本。"},
            {"role": "user", "content": prompt},
        ],
        api_key=api_key,
        model_name=model_name,
        api_base=api_base,
        temperature=0.0,
        timeout=45,
    )
    return fixed


def gemini_preflight_check(api_key, model_name):
    """Fast preflight check to avoid long waiting when Gemini is unreachable or blocked."""
    resolved_model = resolve_gemini_model_name(model_name)
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{resolved_model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": "请回复OK"}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 8,
        },
    }
    options = get_runtime_network_options()
    session = build_http_session(options["proxy_url"], options["use_system_proxy"])
    validate_exit_country_or_raise(session, options)

    try:
        terminal_log("Gemini preflight start", model=resolved_model, requested=model_name, key=_mask_key(api_key))
        start_t = time.time()
        resp = session.post(endpoint, json=payload, timeout=get_request_timeout(options))
        terminal_log("Gemini preflight response", model=resolved_model, status=resp.status_code, elapsed=f"{time.time()-start_t:.2f}s")
        if resp.status_code in (404, 429):
            # Keep going: runtime has fallback model retry logic.
            return True, f"Gemini预检提示（HTTP {resp.status_code}），将继续并自动回退模型"
        if resp.status_code >= 400:
            return False, f"Gemini预检失败（HTTP {resp.status_code}）"
        return True, "Gemini预检通过"
    except requests.exceptions.ConnectTimeout:
        terminal_log("Gemini preflight timeout", model=resolved_model, kind="connect")
        return False, "Gemini连接超时：当前网络可能不可达"
    except requests.exceptions.ReadTimeout:
        terminal_log("Gemini preflight timeout", model=resolved_model, kind="read")
        return False, "Gemini响应超时：网络质量或出口策略异常"
    except Exception as e:
        terminal_log("Gemini preflight exception", model=resolved_model, error_type=type(e).__name__, error=repr(e)[:320])
        return False, f"Gemini预检异常：{e}"


def do_gemini_preflight_with_ui(api_key, model_name, progress_hint, hint_placeholder_empty=True):
    """执行Gemini预检并显示UI提示，返回是否通过"""
    if progress_hint:
        progress_hint.info("Gemini连通性预检中...")
    ok, msg = gemini_preflight_check(api_key.strip(), model_name.strip())
    if not ok:
        if progress_hint:
            progress_hint.empty()
        return False
    if progress_hint:
        progress_hint.success(msg)
    return True



def detect_network_context():
    """Best-effort network context detection for Gemini compliance reminder."""
    context = {
        "public_ip": "未知",
        "country": "未知",
        "country_code": "未知",
        "gemini_reachable": False,
        "network_note": "未检测",
    }

    # Detect public IP and geolocation (best effort).
    try:
        geo_resp = requests.get("https://ipapi.co/json/", timeout=8)
        if geo_resp.ok:
            geo = geo_resp.json()
            context["public_ip"] = str(geo.get("ip", "未知"))
            context["country"] = str(geo.get("country_name", "未知"))
            context["country_code"] = str(geo.get("country_code", "未知"))
    except Exception:
        pass

    # Check whether Gemini endpoint is reachable from current network.
    try:
        ping_resp = requests.get("https://generativelanguage.googleapis.com", timeout=8)
        context["gemini_reachable"] = ping_resp.status_code in [200, 301, 302, 403, 404]
    except Exception:
        context["gemini_reachable"] = False

    if context["country_code"] == "CN":
        context["network_note"] = "当前出口地区疑似中国大陆，调用Gemini前请确认账号与网络合规策略。"
    elif context["country_code"] in ["未知", ""]:
        context["network_note"] = "无法识别出口地区，请谨慎使用Gemini并确认账号合规。"
    else:
        context["network_note"] = f"当前出口地区：{context['country']}。请仍以账号条款与当地政策为准。"

    return context


def call_openai_compatible_chat(messages, api_key, model_name, api_base, temperature=0.3, timeout=45):
    endpoint = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
    }

    terminal_log(
        "OpenAI-compatible request start",
        model=model_name,
        base=api_base,
        msg_count=len(messages),
        key=_mask_key(api_key),
    )
    start_t = time.time()
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    terminal_log("OpenAI-compatible response", model=model_name, status=resp.status_code, elapsed=f"{time.time()-start_t:.2f}s")
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise ValueError("API未返回候选结果")

    content = choices[0].get("message", {}).get("content", "")
    if not content.strip():
        raise ValueError("API返回为空")
    return content.strip()


def refine_prompt_with_openai_compatible(base_prompt, analysis, user_requirement, api_key, model_name, api_base):
    """Refine generated prompt with OpenAI-compatible chat completion API."""
    messages = [
        {
            "role": "system",
            "content": "你是基金投研提示词优化专家。只输出优化后的完整提示词正文，不要解释。",
        },
        {
            "role": "user",
            "content": (
                f"【用户补充要求】\n{user_requirement}\n\n"
                f"【原始提示词】\n{base_prompt}\n\n"
                f"【基金分析数据(JSON)】\n{json.dumps(build_analysis_payload(analysis), ensure_ascii=False)}"
            ),
        },
    ]
    return call_openai_compatible_chat(
        messages=messages,
        api_key=api_key,
        model_name=model_name,
        api_base=api_base,
    )


def merge_user_requirement_into_prompt(base_prompt, user_requirement):
    base = str(base_prompt or "").strip()
    req = str(user_requirement or "").strip()
    if not base:
        return req
    if not req:
        return base

    merged_req = (
        "【我的需求（优先级最高，必须先满足）】\n"
        f"{req}\n\n"
        "【基础输出结构（在满足上方新增需求后继续执行）】"
    )

    if "【我的需求】" in base:
        return base.replace("【我的需求】", merged_req, 1)
    if "我的需求" in base:
        return base.replace("我的需求", merged_req, 1)
    return f"{base}\n\n{merged_req}"


def polish_user_requirement_text(provider, api_key, model_name, api_base, user_requirement):
    raw = str(user_requirement or "").strip()
    if not raw:
        return ""

    polish_prompt = (
        "你是金融分析提示词编辑器。\n"
        "任务：把用户口语化需求改写为“可执行、可验证、结构化”的提示词需求句。\n"
        "要求：\n"
        "1) 只输出改写后的需求文本，不要给基金结论；\n"
        "2) 语义不偏离原需求；\n"
        "3) 若用户提到联网政策分析，但当前无联网检索，请改写为“基于已提供的政策/资讯材料进行分析”；\n"
        "4) 尽量具体，包含时间范围、判断目标（持仓/减仓/继续持有）和依据维度（涨跌数据+政策因素）。\n\n"
        f"【原始需求】\n{raw}"
    )

    try:
        if provider == "Gemini":
            text, _used_model = call_gemini_with_retry(polish_prompt, api_key=api_key, model_name=model_name, zh_only=True)
        else:
            text = call_openai_compatible_chat(
                messages=[
                    {"role": "system", "content": "你是提示词改写器，只输出改写后的需求文本。"},
                    {"role": "user", "content": polish_prompt},
                ],
                api_key=api_key,
                model_name=model_name,
                api_base=api_base,
                temperature=0.2,
                timeout=45,
            )
        cleaned = re.sub(r"\n{2,}", "\n", str(text or "").strip())
        # 防止错误地返回完整投资结论
        if "明确结论" in cleaned and "波段操作计划" in cleaned:
            return raw
        return cleaned or raw
    except Exception:
        return raw


def is_prompt_refine_result_valid(text):
    src = str(text or "").strip()
    if not src:
        return False
    # 必须看起来像提示词模板，而不是直接投资结论
    has_prompt_markers = ("【我的需求" in src) or ("【核心规则" in src) or ("请你扮演" in src)
    if not has_prompt_markers:
        return False
    # 避免返回成“已经分析好的结论”
    bad_markers = ["明确结论：", "适合申购", "暂时不建议申购", "完全不建议申购", "波段操作计划："]
    bad_count = sum(1 for b in bad_markers if b in src)
    return bad_count < 3


def analyze_by_final_prompt(provider, final_prompt, analysis, api_key, model_name, api_base, user_question, zh_only=False):
    terminal_log(
        "Final analysis start",
        provider=provider,
        model=model_name,
        zh_only=zh_only,
        question_len=len(user_question or ""),
    )
    query_text = (
        f"【用户问题】\n{user_question}\n\n"
        f"【请严格遵循的最终提示词】\n{final_prompt}\n\n"
        f"【基金分析数据(JSON)】\n{json.dumps(build_analysis_payload(analysis), ensure_ascii=False)}"
    )

    need_strict_rewrite = is_fund_wave_request(user_question) or is_fund_wave_request(final_prompt)

    if provider == "Gemini":
        answer, used_model = call_gemini_with_retry(
            query_text,
            api_key=api_key,
            model_name=model_name,
            zh_only=zh_only,
        )
        if need_strict_rewrite and not is_fund_wave_answer_valid(answer):
            terminal_log("Final analysis format repair", provider=provider, model=used_model)
            answer = rewrite_fund_wave_answer(
                provider=provider,
                api_key=api_key,
                model_name=used_model,
                api_base=api_base,
                source_question=user_question,
                draft_text=answer,
                zh_only=zh_only,
            )
        cleaned = re.sub(r"\n{3,}", "\n\n", answer).strip()
        terminal_log("Final analysis done", provider=provider, model=used_model, answer_len=len(cleaned))
        return cleaned

    messages = [
        {
            "role": "system",
            "content": "你是专业基金投顾。严格执行用户给定提示词和基金数据，输出可执行结论。",
        },
        {"role": "user", "content": query_text},
    ]
    if zh_only:
        messages.insert(0, {"role": "system", "content": "仅使用简体中文回答。"})

    answer = call_openai_compatible_chat(
        messages=[
            *messages,
        ],
        api_key=api_key,
        model_name=model_name,
        api_base=api_base,
        temperature=0.25,
        timeout=60,
    )
    if need_strict_rewrite and not is_fund_wave_answer_valid(answer):
        terminal_log("Final analysis format repair", provider=provider, model=model_name)
        answer = rewrite_fund_wave_answer(
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            api_base=api_base,
            source_question=user_question,
            draft_text=answer,
            zh_only=zh_only,
        )
    cleaned = re.sub(r"\n{3,}", "\n\n", answer).strip()
    terminal_log("Final analysis done", provider=provider, model=model_name, answer_len=len(cleaned))
    return cleaned


def refine_prompt_by_provider(provider, base_prompt, analysis, user_requirement, api_key, model_name, api_base):
    polished_requirement = polish_user_requirement_text(
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        api_base=api_base,
        user_requirement=user_requirement,
    )
    deterministic_merged = merge_user_requirement_into_prompt(base_prompt, polished_requirement)
    full_prompt = (
        "你是基金投研提示词优化专家。\n"
        "任务：只允许改写‘提示词模板’，不允许给出基金分析结论。\n"
        "输出要求：\n"
        "1) 必须保留提示词结构（核心规则/基金核心数据/我的需求）；\n"
        "2) 将用户补充要求改写后放入【我的需求（优先级最高，必须先满足）】；\n"
        "3) 只输出最终提示词模板正文，不要任何解释和分析结论。\n\n"
        f"【用户补充要求（已规范化）】\n{polished_requirement}\n\n"
        f"【原始提示词】\n{base_prompt}\n\n"
        f"【基金分析数据(JSON)】\n{json.dumps(build_analysis_payload(analysis), ensure_ascii=False)}"
    )

    if provider == "Gemini":
        text, used_model = call_gemini_with_retry(full_prompt, api_key=api_key, model_name=model_name, zh_only=True)
        terminal_log("Prompt refine done", provider=provider, model=used_model, answer_len=len(text))
        if is_prompt_refine_result_valid(text):
            return text
        terminal_log("Prompt refine fallback", provider=provider, reason="invalid-template-output")
        return deterministic_merged

    refined = refine_prompt_with_openai_compatible(
        base_prompt=base_prompt,
        analysis=analysis,
        user_requirement=user_requirement,
        api_key=api_key,
        model_name=model_name,
        api_base=api_base,
    )
    if is_prompt_refine_result_valid(refined):
        return refined
    terminal_log("Prompt refine fallback", provider=provider, reason="invalid-template-output")
    return deterministic_merged


def run_investment_decision(provider, api_key, model_name, api_base, analysis, positions_df, user_goal, risk_style):
    memories = load_recent_memories(limit=8)
    memory_text = json.dumps(memories, ensure_ascii=False)
    position_payload = positions_df.to_dict("records") if positions_df is not None and not positions_df.empty else []
    decision_prompt = (
        "你是买方基金投资顾问，请基于基金数据、持仓与历史记忆给出可执行决策。"
        "输出必须包含：1) 结论(抄底/加仓/持有/减仓/清仓/观望)；2) 置信度(0-100)；"
        "3) 核心依据(3-5条)；4) 具体执行计划(仓位比例、止盈止损、时间窗口)；"
        "5) 风险提示；6) 下一步观察指标。\n\n"
        f"【用户目标】{user_goal}\n"
        f"【风险风格】{risk_style}\n\n"
        f"【基金分析数据】\n{json.dumps(build_analysis_payload(analysis), ensure_ascii=False)}\n\n"
        f"【当前持仓】\n{json.dumps(position_payload, ensure_ascii=False)}\n\n"
        f"【历史记忆(最近8条)】\n{memory_text}"
    )

    if provider == "Gemini":
        text, used_model = call_gemini_with_retry(decision_prompt, api_key=api_key, model_name=model_name, zh_only=True)
        terminal_log("Advisor decision done", provider=provider, model=used_model, answer_len=len(text))
        return text

    return refine_prompt_with_openai_compatible(
        base_prompt="",
        analysis=analysis,
        user_requirement=decision_prompt,
        api_key=api_key,
        model_name=model_name,
        api_base=api_base,
    )


def _extract_json_block(text):
    src = str(text or "").strip()
    if not src:
        return ""
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\}|\[[\s\S]*\])\s*```", src, re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    brace_obj = re.search(r"(\{[\s\S]*\})", src)
    if brace_obj:
        return brace_obj.group(1).strip()
    brace_arr = re.search(r"(\[[\s\S]*\])", src)
    if brace_arr:
        return brace_arr.group(1).strip()
    return src


def parse_positions_from_text_with_gemini(raw_text, api_key, model_name):
    prompt = (
        "你是基金持仓结构化提取助手。请把用户文本中的持仓/申购信息提取为JSON。\n"
        "只输出JSON，不要解释。\n"
        "输出格式：\n"
        "{\n"
        "  \"as_of_date\": \"YYYY-MM-DD 或空\",\n"
        "  \"items\": [\n"
        "    {\n"
        "      \"fund_code\": \"6位字符串\",\n"
        "      \"fund_name\": \"基金名称\",\n"
        "      \"hold_cost\": 数值或null,\n"
        "      \"hold_share\": 数值或null,\n"
        "      \"subscribe_amount\": 数值或null,\n"
        "      \"buy_date\": \"YYYY-MM-DD 或空\",\n"
        "      \"notes\": \"备注\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "规则：\n"
        "1) fund_code必须保留前导0；\n"
        "2) 若文本提供了持仓成本和持仓份额，写入hold_cost/hold_share；\n"
        "3) 若只有申购金额，写入subscribe_amount；\n"
        "4) 无法确定字段填null，不要编造。\n\n"
        f"【用户文本】\n{raw_text}"
    )
    text, used_model = call_gemini_with_retry(prompt, api_key=api_key, model_name=model_name, zh_only=True)
    terminal_log("Position parse done", model=used_model, text_len=len(text))
    json_text = _extract_json_block(text)
    data = json.loads(json_text)
    items = data.get("items", []) if isinstance(data, dict) else []
    as_of_date = data.get("as_of_date", "") if isinstance(data, dict) else ""
    return as_of_date, items


def _safe_float(v):
    if v is None:
        return None
    s = str(v).replace(",", "").replace("%", "").strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _safe_buy_date(v):
    s = str(v or "").strip()
    if not s:
        return datetime.now().strftime("%Y-%m-%d")
    for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"]:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return datetime.now().strftime("%Y-%m-%d")


def calc_fee_free_redeem_date(buy_date_text):
    s = str(buy_date_text or "").strip()
    if not s:
        return "无数据"
    for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"]:
        try:
            dt = datetime.strptime(s, fmt)
            return (dt + timedelta(days=7)).strftime("%Y-%m-%d")
        except Exception:
            continue
    return "无数据"


def _as_trade_date_from_text(text, fallback_date=""):
    src = str(text or "").strip()
    m = re.search(r"(\d{4}-\d{2}-\d{2})", src)
    if m:
        return m.group(1)
    m2 = re.search(r"(\d{4}/\d{2}/\d{2})", src)
    if m2:
        return m2.group(1).replace("/", "-")
    return fallback_date or datetime.now().strftime("%Y-%m-%d")


def _normalize_position_row_values(cost, share, subscribe_amount, latest_nav):
    # 纠偏：把“金额/净值”错位的行自动修正
    if cost is not None and subscribe_amount is not None:
        if cost > 50 and 0 < subscribe_amount < 20:
            cost, subscribe_amount = subscribe_amount, cost

    # 纠偏：老数据只有成本与份额时，若成本明显像金额（如500）则视作申购金额
    if subscribe_amount is None and cost is not None and cost > 50 and latest_nav is not None and latest_nav > 0:
        subscribe_amount = float(cost)
        cost = float(latest_nav)
        if share is None or share <= 0:
            share = subscribe_amount / latest_nav

    if (share is None or share <= 0) and subscribe_amount is not None and subscribe_amount > 0 and cost is not None and cost > 0:
        share = subscribe_amount / cost

    if cost is not None and cost > 50 and subscribe_amount is not None and subscribe_amount > 0 and latest_nav is not None and latest_nav > 0:
        cost = float(latest_nav)
        if share is None or share <= 0:
            share = subscribe_amount / latest_nav

    return cost, share, subscribe_amount


def _calc_profit_fields(latest_nav, cost, share, subscribe_amount):
    if latest_nav is None or share is None or share <= 0:
        return None, None, None, None
    market_value = float(latest_nav) * float(share)
    if subscribe_amount is not None and subscribe_amount > 0:
        principal = float(subscribe_amount)
    elif cost is not None and cost > 0:
        principal = float(cost) * float(share)
    else:
        principal = None
    if principal is None or principal <= 0:
        return market_value, None, None, principal
    profit = market_value - principal
    profit_rate = (profit / principal) * 100
    return market_value, profit, profit_rate, principal


def merge_position_editor_changes(base_df, edited_df):
    merged_df = ensure_position_columns(base_df.copy())
    new_edit = edited_df.copy()
    if "基金代码" not in new_edit.columns:
        return merged_df
    new_edit["基金代码"] = new_edit["基金代码"].astype(str).str.zfill(6)

    for _idx, row in new_edit.iterrows():
        code = str(row.get("基金代码", "")).strip().zfill(6)
        if not code.isdigit() or len(code) != 6:
            continue
        if code in merged_df["基金代码"].astype(str).str.zfill(6).values:
            mask = merged_df["基金代码"].astype(str).str.zfill(6) == code
            for col in ["基金名称", "持仓成本", "持仓份额", "申购金额", "买入日期", "备注"]:
                if col in new_edit.columns and col in merged_df.columns:
                    merged_df.loc[mask, col] = row.get(col)
        else:
            new_row = {c: np.nan for c in merged_df.columns}
            for col in ["基金代码", "基金名称", "持仓成本", "持仓份额", "申购金额", "买入日期", "备注"]:
                if col in merged_df.columns:
                    new_row[col] = row.get(col)
            merged_df = pd.concat([merged_df, pd.DataFrame([new_row])], ignore_index=True)

    return merged_df


def upsert_positions_from_ai_items(position_df, items, overwrite_existing=False):
    updated = ensure_position_columns(position_df.copy())
    inserted_count = 0
    for item in items:
        fund_code = str(item.get("fund_code", "")).strip().zfill(6)
        if not fund_code.isdigit() or len(fund_code) != 6:
            continue

        analyzer = UniversalFundAnalyzer(fund_code)
        fund_name = analyzer.fund_name
        latest_nav = analyzer.history_nav['单位净值'].iloc[-1] if not analyzer.history_nav.empty else None
        if latest_nav is None:
            continue

        hold_cost = _safe_float(item.get("hold_cost"))
        hold_share = _safe_float(item.get("hold_share"))
        subscribe_amount = _safe_float(item.get("subscribe_amount"))
        notes = str(item.get("notes", "") or "").strip()

        if hold_cost is not None and hold_share is not None and hold_share > 0:
            cost = float(hold_cost)
            share = float(hold_share)
            subscribe_val = np.nan
        elif subscribe_amount is not None and subscribe_amount > 0:
            cost = float(latest_nav)
            share = float(subscribe_amount) / float(latest_nav) if latest_nav else 0
            subscribe_val = float(subscribe_amount)
        else:
            continue

        if share <= 0:
            continue

        buy_date = _safe_buy_date(item.get("buy_date"))
        _mv, profit, profit_rate, _p = _calc_profit_fields(latest_nav, cost, share, subscribe_val)
        if profit is None:
            profit = 0
        if profit_rate is None:
            profit_rate = 0

        if fund_code in updated['基金代码'].astype(str).str.zfill(6).values:
            if not overwrite_existing:
                continue
            updated.loc[updated['基金代码'].astype(str).str.zfill(6) == fund_code, [
                "基金名称", "持仓成本", "持仓份额", "申购金额", "买入日期", "备注", "最新净值", "浮盈浮亏", "浮盈浮亏比例"
            ]] = [
                fund_name, cost, share, subscribe_val, buy_date, notes, latest_nav, profit, profit_rate
            ]
        else:
            new_row = pd.DataFrame([{
                "基金代码": fund_code,
                "基金名称": fund_name,
                "持仓成本": cost,
                "持仓份额": share,
                "申购金额": subscribe_val,
                "买入日期": buy_date,
                "备注": notes,
                "最新净值": latest_nav,
                "浮盈浮亏": profit,
                "浮盈浮亏比例": profit_rate,
            }])
            updated = pd.concat([updated, new_row], ignore_index=True)
        inserted_count += 1

    return updated, inserted_count


def ensure_position_columns(position_df):
    alias_map = {
        "fund_code": "基金代码",
        "fund_name": "基金名称",
        "hold_cost": "持仓成本",
        "hold_share": "持仓份额",
        "hold_shares": "持仓份额",
        "subscribe_amount": "申购金额",
        "buy_date": "买入日期",
        "notes": "备注",
    }
    for en_col, zh_col in alias_map.items():
        if en_col in position_df.columns and zh_col not in position_df.columns:
            position_df[zh_col] = position_df[en_col]

    required_cols = [
        "基金代码", "基金名称", "持仓成本", "持仓份额", "买入日期", "最新净值", "浮盈浮亏", "浮盈浮亏比例"
    ]
    for col in required_cols:
        if col not in position_df.columns:
            position_df[col] = np.nan
    if "申购金额" not in position_df.columns:
        position_df["申购金额"] = np.nan
    if "备注" not in position_df.columns:
        position_df["备注"] = ""
    if "当日涨跌幅" not in position_df.columns:
        position_df["当日涨跌幅"] = np.nan
    if "当日涨跌日期" not in position_df.columns:
        position_df["当日涨跌日期"] = ""
    return position_df


def refresh_positions_with_latest(position_df, target_fund_code=None):
    updated_df = ensure_position_columns(position_df.copy())
    update_count = 0
    for idx, row in updated_df.iterrows():
        fund_code = str(row.get('基金代码', '')).strip().zfill(6)
        if not fund_code.isdigit() or len(fund_code) != 6:
            continue
        if target_fund_code and fund_code != str(target_fund_code).strip().zfill(6):
            continue
        try:
            analyzer = UniversalFundAnalyzer(fund_code)
            latest_nav = analyzer.history_nav['单位净值'].iloc[-1] if not analyzer.history_nav.empty else None
            if latest_nav is None:
                continue

            cost = _safe_float(row.get('持仓成本'))
            share = _safe_float(row.get('持仓份额'))
            subscribe_amount = _safe_float(row.get('申购金额'))
            cost, share, subscribe_amount = _normalize_position_row_values(cost, share, subscribe_amount, latest_nav)

            _mv, profit, profit_rate, _principal = _calc_profit_fields(latest_nav, cost, share, subscribe_amount)
            updated_df.loc[idx, ["持仓成本", "持仓份额", "申购金额", "最新净值"]] = [cost, share, subscribe_amount, latest_nav]
            if profit is not None and profit_rate is not None:
                updated_df.loc[idx, ["浮盈浮亏", "浮盈浮亏比例"]] = [profit, profit_rate]

            day_change = parse_change_value(analyzer.gsz)
            fallback_trade_date = analyzer.history_nav['日期'].iloc[-1].strftime('%Y-%m-%d') if not analyzer.history_nav.empty else datetime.now().strftime("%Y-%m-%d")
            day_change_time = _as_trade_date_from_text(analyzer.gsz_time, fallback_date=fallback_trade_date)
            updated_df.loc[idx, ["当日涨跌幅"]] = [day_change if day_change is not None else np.nan]
            updated_df.loc[idx, ["当日涨跌日期"]] = [day_change_time]
            update_count += 1
        except Exception as exc:
            terminal_log("Position refresh failed", fund_code=fund_code, error=repr(exc)[:180])
            continue

    return updated_df, update_count

# ========== Streamlit页面全局配置 ==========
st.set_page_config(
    page_title="基金投资决策工作台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 深色主题CSS样式，匹配金融终端风格
st.markdown("""
<style>
    :root {
        --bg-main: #0E1117;
        --bg-panel: #1E2130;
        --bg-panel-2: #242A3A;
        --text-main: #FAFAFA;
        --text-sub: #A4ACB9;
        --up-red: #FF4B4B;
        --down-green: #00FF94;
        --neutral: #E4E7ED;
    }
    .main {
        background-color: var(--bg-main);
        color: var(--text-main);
    }
    .main .block-container {
        max-width: 95rem;
        padding-top: 0.35rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
    }
    .stApp [data-testid="stHeader"] {
        height: 0.2rem;
    }
    .stApp [data-testid="stToolbar"] {
        top: 0.15rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        padding-top: 0.15rem;
        border-bottom: 1px solid #2E364A;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre-wrap;
        background-color: var(--bg-panel);
        border: 1px solid #2E364A;
        border-bottom: none;
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding-top: 7px;
        padding-bottom: 7px;
        padding-left: 14px;
        padding-right: 14px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #29334B;
        border-color: #3C4C6D;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: var(--neutral);
    }
    .stDataFrame {
        background-color: var(--bg-panel);
    }
    .chg-up {
        color: var(--up-red) !important;
        font-weight: 700;
    }
    .chg-down {
        color: var(--down-green) !important;
        font-weight: 700;
    }
    .chg-neutral {
        color: var(--neutral) !important;
        font-weight: 700;
    }
    .summary-card {
        background: linear-gradient(135deg, #1E2130 0%, #252C3F 100%);
        border: 1px solid #2E364A;
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 8px;
    }
    .summary-title {
        color: var(--text-sub);
        font-size: 12px;
        margin-bottom: 4px;
    }
    .summary-value {
        color: var(--text-main);
        font-size: 20px;
        font-weight: 700;
    }
    section[data-testid="stSidebar"] .feature-nav {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 8px 10px;
        margin-bottom: 10px;
        margin-top: 2px;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 0rem;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 0rem;
        margin-top: -0.35rem;
    }
    section[data-testid="stSidebar"] [data-testid="stButton"] {
        margin-top: 0.05rem;
        margin-bottom: 0.45rem;
    }
    section[data-testid="stSidebar"] .feature-nav a {
        color: #c8d2e2;
        text-decoration: none;
        font-size: 0.85rem;
        line-height: 1.7;
    }
    section[data-testid="stSidebar"] .feature-nav a:hover {
        color: #ffffff;
        text-decoration: underline;
    }
    .investor-hero {
        background: radial-gradient(circle at 80% 20%, rgba(89, 130, 255, 0.28), transparent 36%),
                    linear-gradient(135deg, #1a2238 0%, #202b45 55%, #162035 100%);
        border: 1px solid rgba(122, 161, 255, 0.45);
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 12px;
        box-shadow: 0 12px 26px rgba(0, 0, 0, 0.25);
    }
    .hero-title {
        font-size: 1.18rem;
        font-weight: 700;
        color: #e7efff;
        margin-bottom: 4px;
    }
    .hero-sub {
        color: #bfd0f5;
        font-size: 0.95rem;
    }
    .guide-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        margin: 12px 0 6px 0;
    }
    .guide-item {
        background: #171f31;
        border: 1px solid #2a3a5a;
        border-radius: 10px;
        padding: 10px 12px;
    }
    .guide-k {
        color: #7fb3ff;
        font-size: 0.8rem;
        margin-bottom: 3px;
    }
    .guide-v {
        color: #edf2ff;
        font-size: 0.95rem;
        font-weight: 600;
    }
    @media (max-width: 900px) {
        .guide-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)

if st.sidebar.button("⏹ 安全关闭程序", use_container_width=True, key="safe_shutdown_btn"):
    try:
        init_data_folder()
        position_path_shutdown = "data/position.csv"
        snapshot = st.session_state.get("position_editor_snapshot")
        if isinstance(snapshot, pd.DataFrame):
            if os.path.exists(position_path_shutdown):
                base_df_shutdown = pd.read_csv(position_path_shutdown, encoding="utf-8-sig")
            else:
                base_df_shutdown = pd.DataFrame()
            base_df_shutdown = ensure_position_columns(base_df_shutdown)
            merged_shutdown = merge_position_editor_changes(base_df_shutdown, snapshot)
            merged_shutdown, _ = refresh_positions_with_latest(merged_shutdown)
            merged_shutdown.to_csv(position_path_shutdown, index=False, encoding="utf-8-sig")
        elif os.path.exists(position_path_shutdown):
            base_df_shutdown = pd.read_csv(position_path_shutdown, encoding="utf-8-sig")
            base_df_shutdown = ensure_position_columns(base_df_shutdown)
            base_df_shutdown.to_csv(position_path_shutdown, index=False, encoding="utf-8-sig")

        st.sidebar.success("已完成保存检查，程序正在关闭...")
        time.sleep(0.4)
        os._exit(0)
    except Exception as shutdown_err:
        st.sidebar.error(f"关闭失败：{shutdown_err}")

st.sidebar.markdown("### 功能目录")
st.sidebar.caption("轻量导航")
st.sidebar.markdown(
    """
<div class="feature-nav">
<a href="#fund-main">基金一键分析</a><br>
<a href="#fund-stats">关键统计</a><br>
<a href="#prompt-opt">提示词优化</a><br>
<a href="#prompt-final-analysis">AI最终分析</a><br>
<a href="#history-main">历史记录</a><br>
<a href="#position-main">持仓管理</a><br>
<a href="#advisor-main">智能投顾决策</a>
</div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Gemini网络策略")
st.sidebar.checkbox("使用系统代理", value=st.session_state.get("net_use_system_proxy", True), key="net_use_system_proxy")
st.sidebar.text_input(
    "显式代理(可选)",
    value=st.session_state.get("net_proxy_url", "http://127.0.0.1:7890"),
    key="net_proxy_url",
    help="建议显式代理，例如 http://127.0.0.1:7890",
)
st.sidebar.selectbox("要求出口地区", options=["", "US", "SG", "JP", "HK"], index=1, key="net_require_country")
st.sidebar.checkbox("出口未知时允许继续", value=st.session_state.get("net_allow_unknown_country", True), key="net_allow_unknown_country")
st.sidebar.number_input("连接超时(秒)", min_value=3, max_value=30, value=int(st.session_state.get("net_connect_timeout", 6)), key="net_connect_timeout")
st.sidebar.number_input("读取超时(秒)", min_value=8, max_value=120, value=int(st.session_state.get("net_read_timeout", 25)), key="net_read_timeout")
with st.sidebar.expander("⚙️ Gemini调参", expanded=False):
    st.caption("💡 拖动滑块调整或直接输入新值")
    st.caption("参数说明：T=随机性，P=概率覆盖范围，K=候选词数量。稳健建议 T=0.2、P=0.85-0.9、K=20-40。")
    
    # Temperature
    col1, col2 = st.columns([3.5, 1.5])
    with col1:
        st.slider("Temperature", min_value=0.0, max_value=1.0, value=float(st.session_state.get("gemini_temperature", 0.25)), step=0.05, key="gemini_temperature")
    with col2:
        st.caption(f"当前: {st.session_state.get('gemini_temperature', 0.25):.2f}")
    
    # Top P
    col1, col2 = st.columns([3.5, 1.5])
    with col1:
        st.slider("Top P", min_value=0.1, max_value=1.0, value=float(st.session_state.get("gemini_top_p", 0.9)), step=0.05, key="gemini_top_p")
    with col2:
        st.caption(f"当前: {st.session_state.get('gemini_top_p', 0.9):.2f}")
    
    # Top K
    col1, col2 = st.columns([3.5, 1.5])
    with col1:
        st.slider("Top K", min_value=1, max_value=80, value=int(st.session_state.get("gemini_top_k", 40)), step=1, key="gemini_top_k")
    with col2:
        st.caption(f"当前: {st.session_state.get('gemini_top_k', 40)}")
    
    # 最大输出Tokens (max 提高到 16384)
    col1, col2 = st.columns([3.5, 1.5])
    with col1:
        st.slider("最大输出Tokens", min_value=512, max_value=16384, value=int(st.session_state.get("gemini_max_output_tokens", 4096)), step=512, key="gemini_max_output_tokens")
    with col2:
        st.caption(f"当前: {st.session_state.get('gemini_max_output_tokens', 4096)}")
    
    # 回退模型尝试数
    col1, col2 = st.columns([3.5, 1.5])
    with col1:
        st.slider("回退模型尝试数", min_value=1, max_value=6, value=int(st.session_state.get("gemini_max_retry_models", 5)), step=1, key="gemini_max_retry_models")
    with col2:
        st.caption(f"当前: {st.session_state.get('gemini_max_retry_models', 5)}")
    
    # 结果最小字数 (max 提高到 2000)
    col1, col2 = st.columns([3.5, 1.5])
    with col1:
        st.slider("结果最小字数", min_value=120, max_value=2000, value=int(st.session_state.get("gemini_rewrite_min_chars", 280)), step=20, key="gemini_rewrite_min_chars")
    with col2:
        st.caption(f"当前: {st.session_state.get('gemini_rewrite_min_chars', 280)}")
    
    # 结果字号(px)
    col1, col2 = st.columns([3.5, 1.5])
    with col1:
        st.slider("结果字号(px)", min_value=12, max_value=32, value=int(st.session_state.get("ui_output_font_px", 16)), step=1, key="ui_output_font_px")
    with col2:
        st.caption(f"当前: {st.session_state.get('ui_output_font_px', 16)}px")
    
    st.text_area(
        "附加系统指令(可选)",
        value=st.session_state.get("gemini_extra_instruction", ""),
        key="gemini_extra_instruction",
        height=90,
        placeholder="示例：优先输出可执行策略，减少泛泛表述。",
    )

# 初始化数据文件夹
init_data_folder()
ensure_memory_file()

# ========== 页面标签页 ==========
_tabs = st.tabs(["📈 基金一键分析", "📋 历史分析记录", "💰 持仓盈亏管理", "🧠 智能投顾决策"])

# ========== 标签1：基金一键分析（核心功能） ==========
with _tabs[0]:
    st.markdown('<div id="fund-main"></div>', unsafe_allow_html=True)
    st.title("基金7-30天波段分析系统")
    st.markdown(
        """
        <div class="investor-hero">
            <div class="hero-title">面向个人投资者的决策工作流</div>
            <div class="hero-sub">先看数据，再优化提示词，最后执行AI结论，减少主观交易与情绪化决策。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    
    # 输入区域
    col1, col2 = st.columns([3, 1])
    with col1:
        fund_code = st.text_input("请输入6位基金代码（场外C类/场内ETF均可）", max_chars=6, placeholder="例如：020671、562500")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🚀 一键开始分析", use_container_width=True, type="primary")
    
    st.markdown("---")
    
    # 分析逻辑
    if analyze_btn and fund_code:
        with st.spinner("正在获取基金数据，分析中..."):
            # 初始化分析器
            analyzer = UniversalFundAnalyzer(fund_code)
            analysis = analyzer.get_full_analysis()
            # 保存到session_state，页面刷新不丢失
            st.session_state['current_analysis'] = analysis
            st.session_state['current_fund_code'] = fund_code
            # 保存到历史记录
            base_info = analysis['基金基础信息']
            save_history(
                fund_code=fund_code,
                fund_name=base_info['基金名称'],
                latest_nav=base_info['最新净值'],
                nav_date=base_info['净值更新日期']
            )
    
    # 展示分析结果
    if 'current_analysis' in st.session_state:
        analysis = st.session_state['current_analysis']
        base_info = analysis['基金基础信息']
        realtime_data = analysis['实时行情数据']
        history_nav = analysis['历史净值数据']
        
        # 1. 基金基础信息卡片
        st.subheader(f"📊 {base_info['基金名称']}({base_info['基金代码']})")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("最新净值", format_num(base_info['最新净值'], 4))
        with col2:
            change = realtime_data['估算涨跌幅']
            change_float = parse_change_value(change)
            if change_float is not None:
                if change_float > 0:
                    color_cls = "chg-up"
                elif change_float < 0:
                    color_cls = "chg-down"
                else:
                    color_cls = "chg-neutral"
                st.markdown(
                    f"""
                    <div class="summary-card">
                        <div class="summary-title">最新涨跌幅</div>
                        <div class="summary-value {color_cls}">{format_num(change_float, 2)}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.metric("最新涨跌幅", change)
        with col3:
            st.metric("基金类型", base_info['基金类型'])
        with col4:
            st.metric("净值更新日期", base_info['净值更新日期'])

        snapshot = build_investor_snapshot(analysis)
        st.markdown(
            f"""
            <div class="guide-grid">
                <div class="guide-item"><div class="guide-k">趋势状态</div><div class="guide-v">{snapshot['趋势状态']}</div></div>
                <div class="guide-item"><div class="guide-k">短期风险</div><div class="guide-v">{snapshot['短期风险']}</div></div>
                <div class="guide-item"><div class="guide-k">择时区间</div><div class="guide-v">{snapshot['择时区间']}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            f"投资者快照：近7日 {format_num(snapshot['7日涨幅'], 2)}%，"
            f"近30日 {format_num(snapshot['30日涨幅'], 2)}%，"
            f"近30日最大回撤 {format_num(snapshot['30日回撤'], 2)}%。"
        )
        
        st.markdown("---")

        # 额外关键统计（参考基金网站常用概览指标）
        st.markdown('<div id="fund-stats"></div>', unsafe_allow_html=True)
        st.subheader("📌 关键统计概览")
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        if not history_nav.empty and len(history_nav) >= 7:
            nav_series = history_nav['单位净值']
            ret_1d = nav_series.pct_change().iloc[-1] * 100 if len(nav_series) >= 2 else 0
            vol_30d = nav_series.pct_change().tail(30).std() * np.sqrt(252) * 100 if len(nav_series) >= 30 else np.nan
            drawdown_30d = calculate_cumulative_performance(history_nav, 30).get("最大回撤", "无数据")

            with stat_col1:
                st.metric("最近1日涨跌", f"{format_num(ret_1d, 2)}%")
            with stat_col2:
                st.metric("近30日年化波动率", f"{format_num(vol_30d, 2)}%" if not pd.isna(vol_30d) else "无数据")
            with stat_col3:
                st.metric("近30日最大回撤", f"{format_num(drawdown_30d, 2)}%" if drawdown_30d != "无数据" else "无数据")

        link_col1, link_col2 = st.columns(2)
        with link_col1:
            st.markdown(f"🔗 [东方财富基金档案](https://fund.eastmoney.com/{base_info['基金代码']}.html)")
        with link_col2:
            st.markdown(f"🔗 [天天基金净值估算](https://fundgz.1234567.com.cn/js/{base_info['基金代码']}.js)")

        st.markdown("---")
        
        # 2. 可视化曲线
        st.subheader("📈 净值走势与涨跌统计")
        col1, col2 = st.columns([2, 1])
        
        # 近30天净值曲线
        with col1:
            if not history_nav.empty:
                df_30d = history_nav.tail(30).copy()
                # 计算累计涨幅，判断颜色
                total_change_30d = (df_30d['单位净值'].iloc[-1] / df_30d['单位净值'].iloc[0] - 1) * 100
                line_color = "#00FF94" if total_change_30d >= 0 else "#FF4B4B"
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_30d['日期'],
                    y=df_30d['单位净值'],
                    mode='lines',
                    line=dict(color=line_color, width=3),
                    name='单位净值'
                ))
                fig.update_layout(
                    title="近30天净值走势",
                    plot_bgcolor="#1E2130",
                    paper_bgcolor="#0E1117",
                    font=dict(color="#FAFAFA"),
                    xaxis=dict(title="日期", gridcolor="#2D3748"),
                    yaxis=dict(title="单位净值", gridcolor="#2D3748"),
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # 近7天涨跌柱状图
        with col2:
            daily_detail = analysis['最近7个交易日单日涨跌明细']
            if daily_detail:
                df_7d = pd.DataFrame(daily_detail)
                if '日期' in df_7d.columns:
                    df_7d = df_7d.sort_values('日期')
                df_7d['涨跌颜色'] = df_7d['单日涨跌幅'].apply(lambda x: "#00FF94" if x >= 0 else "#FF4B4B")
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_7d['日期'],
                    y=df_7d['单日涨跌幅'],
                    marker=dict(color=df_7d['涨跌颜色']),
                    name='单日涨跌幅(%)'
                ))
                fig.update_layout(
                    title="近7天单日涨跌",
                    plot_bgcolor="#1E2130",
                    paper_bgcolor="#0E1117",
                    font=dict(color="#FAFAFA"),
                    xaxis=dict(title="交易日", gridcolor="#2D3748"),
                    yaxis=dict(title="涨跌幅(%)", gridcolor="#2D3748"),
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # 3. 核心数据表格
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("📅 近7个交易日单日明细")
            daily_df = pd.DataFrame(analysis['最近7个交易日单日涨跌明细'])
            if '序号' in daily_df.columns:
                daily_df = daily_df.drop(columns=['序号'])
            if '日期' in daily_df.columns:
                daily_df = daily_df.sort_values('日期', ascending=False)
            st.dataframe(daily_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("📊 波段周期累计表现")
            cumulative_df = pd.DataFrame(analysis['波段周期累计表现']).T
            st.dataframe(cumulative_df, use_container_width=True)
        
        st.markdown("---")
        
        # 4. 择时指标与持仓分析
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("🎯 波段择时指标")
            timing_df = pd.DataFrame([analysis['波段择时指标']]).T
            st.dataframe(timing_df, use_container_width=True)
        
        with col2:
            st.subheader("📦 持仓结构分析")
            position_df = pd.DataFrame([analysis['持仓结构分析']]).T
            st.dataframe(position_df, use_container_width=True)
        
        st.markdown("---")
        
        # 5. 基准对比分析
        st.subheader("📋 基准对比分析")
        benchmark_df = pd.DataFrame([analysis['基准对比分析']]).T
        st.dataframe(benchmark_df, use_container_width=True)
        
        st.markdown("---")
        
        # 6. Gemini提示词一键复制
        st.subheader("🤖 Gemini波段操作提示词（一键复制）")
        gemini_prompt = generate_gemini_prompt(analysis)
        st.code(gemini_prompt, language="markdown")

        st.download_button(
            "⬇️ 下载本次分析报告(.txt)",
            data=gemini_prompt.encode("utf-8"),
            file_name=f"基金波段分析_{fund_code}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown('<div id="prompt-opt"></div>', unsafe_allow_html=True)
        st.subheader("🧠 API二次改写提示词（按你的额外要求自动优化）")
        st.caption("支持 Gemini / DeepSeek / 豆包。选择渠道后，模型列表会自动切换。")
        st.info("提示：这里的功能是“改写提示词模板”，不是直接做基金结论分析。")

        provider = st.selectbox("选择API渠道", list(PROVIDER_CONFIG.keys()), index=0)
        provider_meta = PROVIDER_CONFIG[provider]
        model_options = provider_meta["models"]
        model_select = st.selectbox("选择模型", model_options, index=0)
        if model_select == "自定义模型":
            model_name = st.text_input("自定义模型名", value="")
        else:
            model_name = model_select

        saved_api_key = get_saved_api_key(provider)
        api_key = st.text_input(
            f"{provider} API Key",
            type="password",
            value=saved_api_key,
            placeholder=provider_meta["key_placeholder"],
        )
        secret_col1, secret_col2 = st.columns(2)
        with secret_col1:
            if st.button("💾 保存当前Key到本地(不进Git)", use_container_width=True, key="save_key_tab1"):
                if api_key.strip():
                    save_api_key(provider, api_key.strip())
                    st.success(f"{provider} Key 已保存到本地安全文件")
                else:
                    st.warning("当前Key为空，未保存")
        with secret_col2:
            if st.button("🧹 清除该渠道本地Key", use_container_width=True, key="clear_key_tab1"):
                clear_api_key(provider)
                st.success(f"{provider} 本地Key已清除")

        st.caption(f"本地密钥文件：{LOCAL_SECRETS_FILE.name}（已加入git忽略）")
        api_base = st.text_input(
            f"{provider} API Base",
            value=provider_meta["api_base"],
            help="默认地址通常可直接使用。若你使用代理网关或私有部署，可改为自定义Base URL。",
        )

        if provider == "Gemini":
            with st.expander("Gemini跨网与合规检测", expanded=False):
                st.caption("当前应用调用的是 Gemini 生成接口，本身不带网页浏览工具；实时信息主要来自你本地抓取的数据源。")
                if st.button("检测当前网络状态", key="gemini_check_btn_tab1"):
                    session = build_http_session(
                        st.session_state.get("net_proxy_url", "").strip(),
                        bool(st.session_state.get("net_use_system_proxy", False)),
                    )
                    geo = detect_exit_geo(session, timeout_sec=int(st.session_state.get("net_connect_timeout", 6)))
                    st.session_state["gemini_net_tab1"] = {
                        "public_ip": geo["ip"],
                        "country": geo["country_name"],
                        "country_code": geo["country_code"],
                        "gemini_reachable": detect_network_context()["gemini_reachable"],
                        "network_note": f"来源: {geo['source']}",
                    }

                net = st.session_state.get("gemini_net_tab1")
                if net:
                    st.caption(
                        f"公网IP: {net['public_ip']} | 出口地区: {net['country']}({net['country_code']}) | "
                        f"Gemini连通性: {'可达' if net['gemini_reachable'] else '不可达/超时'}"
                    )
                    st.caption(
                        f"策略: proxy={'显式' if st.session_state.get('net_proxy_url','').strip() else ('系统' if st.session_state.get('net_use_system_proxy', False) else '关闭')} | "
                        f"require_country={st.session_state.get('net_require_country', '') or '无'} | "
                        f"allow_unknown={st.session_state.get('net_allow_unknown_country', True)}"
                    )
                    if net["country_code"] == "CN":
                        st.warning(net["network_note"])
                    else:
                        st.info(net["network_note"])

            gemini_confirm_tab1 = st.checkbox(
                "我已确认当前网络与账号使用策略合规，允许调用Gemini",
                value=False,
                key="gemini_confirm_tab1",
            )
        else:
            gemini_confirm_tab1 = True

        if "user_requirement_tab1" not in st.session_state:
            st.session_state["user_requirement_tab1"] = ""

        st.markdown("#### 🧩 快速生成策略要求")
        st.caption("保留点选生成能力；你可在每个下拉框最后选择“新增自定义选项”并保存，后续会自动记住。")

        action_options = _get_options_with_plus(BASE_ACTION_OPTIONS, "action")
        risk_options = _get_options_with_plus(BASE_RISK_OPTIONS, "risk")
        asset_options = _get_options_with_plus(BASE_ASSET_OPTIONS, "asset")
        signal_options = _get_options_with_plus(BASE_SIGNAL_OPTIONS, "signal")

        maker_col1, maker_col2 = st.columns(2)
        with maker_col1:
            action_choice = st.selectbox("操作方向", action_options, key="req_action_tab1")
            risk_choice = st.selectbox("策略风格", risk_options, key="req_risk_tab1")
        with maker_col2:
            asset_choice = st.selectbox("标的视角", asset_options, key="req_asset_tab1")
            signal_choice = st.selectbox("信号侧重", signal_options, key="req_signal_tab1")

        plus_map = [
            ("action", "操作方向", action_choice, "new_action_text_tab1", "save_new_action_tab1"),
            ("risk", "策略风格", risk_choice, "new_risk_text_tab1", "save_new_risk_tab1"),
            ("asset", "标的视角", asset_choice, "new_asset_text_tab1", "save_new_asset_tab1"),
            ("signal", "信号侧重", signal_choice, "new_signal_text_tab1", "save_new_signal_tab1"),
        ]
        for preset_key, title, selected, input_key, btn_key in plus_map:
            if selected == "➕ 新增自定义选项":
                c1, c2 = st.columns([4, 1])
                with c1:
                    new_text = st.text_input(f"新增{title}", key=input_key, placeholder=f"输入要新增的{title}")
                with c2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("保存", key=btn_key, use_container_width=True):
                        if add_custom_strategy_option(preset_key, new_text):
                            st.success(f"已保存：{new_text.strip()}")
                            st.experimental_rerun()
                        else:
                            st.warning("请输入有效内容后再保存")

        effective_action = action_choice if action_choice != "➕ 新增自定义选项" else BASE_ACTION_OPTIONS[0]
        effective_risk = risk_choice if risk_choice != "➕ 新增自定义选项" else BASE_RISK_OPTIONS[0]
        effective_asset = asset_choice if asset_choice != "➕ 新增自定义选项" else BASE_ASSET_OPTIONS[0]
        effective_signal = signal_choice if signal_choice != "➕ 新增自定义选项" else BASE_SIGNAL_OPTIONS[0]

        if st.button("🪄 生成策略需求草稿", use_container_width=True, key="gen_req_template_tab1"):
            st.session_state["user_requirement_tab1"] = build_requirement_template(
                action=effective_action,
                risk_style=effective_risk,
                asset_view=effective_asset,
                signal_focus=effective_signal,
            )

        st.markdown("##### 快捷补充语句")
        ensure_strategy_preset_state()
        quick_phrases = _dedup_keep_order([
            *BASE_QUICK_REQUIREMENT_PHRASES,
            *st.session_state["strategy_presets"].get("quick_phrases", []),
        ])
        phrase_cols = st.columns(3)
        for idx, phrase in enumerate(quick_phrases):
            with phrase_cols[idx % 3]:
                if st.button(f"➕ 语句{idx+1}", key=f"quick_phrase_{idx}_tab1", use_container_width=True):
                    current = st.session_state.get("user_requirement_tab1", "").strip()
                    st.session_state["user_requirement_tab1"] = f"{current}\n{phrase}".strip() if current else phrase

        q1, q2 = st.columns([4, 1])
        with q1:
            custom_phrase = st.text_input("新增快捷语句", key="new_quick_phrase_tab1", placeholder="输入后保存，下次可直接点选")
        with q2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("保存语句", key="save_new_quick_phrase_tab1", use_container_width=True):
                if add_custom_strategy_option("quick_phrases", custom_phrase):
                    st.success("快捷语句已保存")
                    st.experimental_rerun()
                else:
                    st.warning("请输入有效语句后再保存")

        with st.expander("🗂 管理已保存快捷语句", expanded=False):
            ensure_strategy_preset_state()
            saved_phrases = st.session_state["strategy_presets"].get("quick_phrases", [])
            if saved_phrases:
                st.caption("可删除你自己保存的快捷语句（内置语句不会出现在这里）。")
                del_targets = st.multiselect(
                    "选择要删除的语句",
                    options=saved_phrases,
                    key="delete_quick_phrase_targets_tab1",
                )
                if st.button("🗑 删除选中语句", key="delete_quick_phrase_btn_tab1", use_container_width=True):
                    removed = remove_custom_strategy_options("quick_phrases", del_targets)
                    if removed > 0:
                        st.success(f"已删除 {removed} 条快捷语句")
                        st.experimental_rerun()
                    else:
                        st.warning("请先选择要删除的语句")
            else:
                st.caption("你还没有保存自定义快捷语句。")

            st.markdown("##### 批量新增")
            batch_text = st.text_area(
                "每行一条语句",
                key="batch_new_quick_phrase_tab1",
                placeholder="示例：\n请明确分批仓位比例。\n请给出执行概率与备选情景。",
                height=90,
            )
            if st.button("📥 批量保存语句", key="batch_save_quick_phrase_btn_tab1", use_container_width=True):
                lines = [line.strip() for line in str(batch_text or "").splitlines() if line.strip()]
                added = 0
                for line in lines:
                    if add_custom_strategy_option("quick_phrases", line):
                        added += 1
                if added > 0:
                    st.success(f"已保存 {added} 条语句")
                    st.experimental_rerun()
                else:
                    st.warning("没有可保存的新语句")

        user_requirement = st.text_area(
            "补充你的策略要求",
            key="user_requirement_tab1",
            placeholder=(
                "示例：请按保守策略分析未来7-30天波段机会，"
                "回撤控制在3%以内，给出分批仓位、止盈止损和执行概率；"
                "若无法实时联网，请基于已提供的政策/资讯材料判断。"
            ),
            height=170,
        )

        refine_btn = st.button("✨ 调用API改写提示词", use_container_width=True)
        if refine_btn:
            if not api_key.strip():
                st.error("请先填写API Key")
            elif not user_requirement.strip():
                st.error("请先填写补充策略要求")
            elif not model_name.strip():
                st.error("请先选择或填写模型名")
            else:
                progress_hint = st.empty()
                preflight_ok = True
                if provider == "Gemini":
                    preflight_ok = do_gemini_preflight_with_ui(api_key.strip(), model_name.strip(), progress_hint)
                    if not preflight_ok:
                        st.error("Gemini预检未通过，请检查网络或模型权限")

                if preflight_ok:
                    with st.spinner("正在调用API优化提示词..."):
                        try:
                            progress_hint.info("步骤2/2：模型生成中，请稍候...")
                            refined_prompt = refine_prompt_by_provider(
                                provider=provider,
                                base_prompt=gemini_prompt,
                                analysis=analysis,
                                user_requirement=st.session_state.get("user_requirement_tab1", ""),
                                api_key=api_key.strip(),
                                model_name=model_name.strip(),
                                api_base=api_base.strip(),
                            )
                            st.session_state["refined_prompt"] = refined_prompt
                            st.success("✅ 提示词已按你的要求优化完成")
                            progress_hint.empty()
                        except Exception as e:
                            progress_hint.empty()
                            st.error(f"API调用失败：{e}")

        if "refined_prompt" in st.session_state:
            st.markdown("#### 📝 优化后提示词")
            st.code(st.session_state["refined_prompt"], language="markdown")
            st.download_button(
                "⬇️ 下载优化后提示词(.txt)",
                data=st.session_state["refined_prompt"].encode("utf-8"),
                file_name=f"基金波段分析_优化提示词_{fund_code}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown('<div id="prompt-final-analysis"></div>', unsafe_allow_html=True)
        st.subheader("📣 使用最终提示词直接分析基金数据")
        final_prompt_text = st.session_state.get("refined_prompt", gemini_prompt)
        if "refined_prompt" in st.session_state:
            st.caption("当前使用：优化后提示词")
        else:
            st.caption("当前使用：原始提示词（未二次优化）")
        st.info("本模块将直接按“最新版本提示词”执行分析，不再需要额外问题输入。")
        analysis_question = "请严格按最终提示词模板输出完整分析结论，并给出可执行的波段操作建议。"
        zh_only_output = st.checkbox("仅输出中文", value=True, key="final_analysis_zh_only")
        run_final_btn = st.button("🧠 基于最终提示词生成分析结论", use_container_width=True, key="run_final_analysis")
        if run_final_btn:
            if not api_key.strip():
                st.error("请先填写API Key")
            elif not model_name.strip():
                st.error("请先选择或填写模型名")
            elif provider == "Gemini" and not gemini_confirm_tab1:
                st.error("请先完成Gemini合规确认勾选")
            else:
                run_hint = st.empty()
                preflight_ok = True
                if provider == "Gemini":
                    preflight_ok = do_gemini_preflight_with_ui(api_key.strip(), model_name.strip(), run_hint)
                    if not preflight_ok:
                        st.error("Gemini预检未通过，请检查网络/代理设置后重试")

                if preflight_ok:
                    with st.spinner("正在基于最终提示词生成分析..."):
                        try:
                            run_hint.info("步骤2/2：模型分析中，请稍候...")
                            final_result = analyze_by_final_prompt(
                                provider=provider,
                                final_prompt=final_prompt_text,
                                analysis=analysis,
                                api_key=api_key.strip(),
                                model_name=model_name.strip(),
                                api_base=api_base.strip(),
                                user_question=analysis_question,
                                zh_only=zh_only_output,
                            )
                            st.session_state["final_prompt_result"] = final_result
                            run_hint.empty()
                        except Exception as e:
                            run_hint.empty()
                            st.error(f"分析失败：{e}")

        if "final_prompt_result" in st.session_state:
            st.markdown("#### ✅ 分析结果")
            font_px = int(st.session_state.get("ui_output_font_px", 16))
            st.markdown(
                f"<div style='font-size:{font_px}px; line-height:1.8'>{st.session_state['final_prompt_result']}</div>",
                unsafe_allow_html=True,
            )
            st.download_button(
                "⬇️ 下载分析结果(.txt)",
                data=st.session_state["final_prompt_result"].encode("utf-8"),
                file_name=f"基金AI分析结果_{fund_code}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True,
                key="download_final_result",
            )

        st.markdown("---")
        st.subheader("💬 沟通分析（连续提问）")
        st.caption("基于当前最终提示词进行连续问答；可勾选仅中文输出。")
        chat_question = st.text_area(
            "继续提问",
            placeholder="示例：如果我准备分三笔建仓，请按7-30天周期给我分批计划。",
            height=80,
            key="dialog_question_tab1",
        )
        dialog_zh_only = st.checkbox("沟通分析仅中文输出", value=True, key="dialog_zh_only_tab1")
        ask_dialog_btn = st.button("📨 发送沟通问题", use_container_width=True, key="dialog_ask_btn_tab1")
        if ask_dialog_btn:
            if not api_key.strip():
                st.error("请先填写或保存API Key")
            elif not model_name.strip():
                st.error("请先选择模型")
            elif not chat_question.strip():
                st.error("请先输入沟通问题")
            elif provider == "Gemini" and not gemini_confirm_tab1:
                st.error("请先完成Gemini合规确认勾选")
            else:
                try:
                    chat_answer = analyze_by_final_prompt(
                        provider=provider,
                        final_prompt=final_prompt_text,
                        analysis=analysis,
                        api_key=api_key.strip(),
                        model_name=model_name.strip(),
                        api_base=api_base.strip(),
                        user_question=chat_question.strip(),
                        zh_only=dialog_zh_only,
                    )
                    history_key = "dialog_history_tab1"
                    if history_key not in st.session_state:
                        st.session_state[history_key] = []
                    st.session_state[history_key].append({"q": chat_question.strip(), "a": chat_answer})
                    st.success("沟通分析已完成")
                except Exception as e:
                    st.error(f"沟通分析失败：{e}")

        dialog_history = st.session_state.get("dialog_history_tab1", [])
        if dialog_history:
            st.markdown("#### 🗂️ 沟通记录")
            font_px = int(st.session_state.get("ui_output_font_px", 16))
            for idx, item in enumerate(dialog_history[-6:], 1):
                st.markdown(f"**Q{idx}:** {item['q']}")
                st.markdown(
                    f"<div style='font-size:{font_px}px; line-height:1.8'>{item['a']}</div>",
                    unsafe_allow_html=True,
                )
        
        # 保存报告到本地
        filename = f"基金波段分析报告/基金波段分析_{fund_code}_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(gemini_prompt)
        st.success(f"✅ 分析报告已保存到：{filename}")

# ========== 标签2：历史分析记录 ==========
with _tabs[1]:
    st.markdown('<div id="history-main"></div>', unsafe_allow_html=True)
    st.title("历史分析记录")
    st.markdown("---")
    
    history_df = load_history()
    if not history_df.empty:
        # 确保基金代码以字符串形式显示（6位无逗号）
        history_df['基金代码'] = history_df['基金代码'].astype(str).str.zfill(6)

        action_col1, action_col2 = st.columns([2, 3])
        with action_col1:
            if st.button("🔄 一键更新全部持仓到最新净值", use_container_width=True, key="history_refresh_positions_btn"):
                position_path = "data/position.csv"
                if os.path.exists(position_path):
                    pos_df = pd.read_csv(position_path, encoding="utf-8-sig")
                    pos_df, cnt = refresh_positions_with_latest(pos_df)
                    pos_df.to_csv(position_path, index=False, encoding="utf-8-sig")
                    st.success(f"✅ 持仓已更新：{cnt} 条")
                    st.experimental_rerun()
                else:
                    st.warning("未找到持仓文件，请先在持仓页新增记录")
        with action_col2:
            st.caption("说明：会刷新最新净值、浮盈浮亏、浮盈浮亏比例、当日涨跌幅。")
        
        # 展示历史记录表格
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.markdown("---")
        
        # 操作区域：重新分析 + 删除
        st.subheader("📋 管理分析记录")
        for idx, row in history_df.iterrows():
            col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 0.5])
            fund_code = str(row['基金代码']).zfill(6)
            fund_name = row['基金名称']
            with col1:
                st.write(f"**{fund_code}** - {fund_name}")
            with col2:
                if st.button("🔄 重新分析", key=f"reanalyze_{fund_code}_{idx}", use_container_width=True):
                    st.session_state['current_fund_code'] = fund_code
                    with st.spinner("正在重新获取数据..."):
                        analyzer = UniversalFundAnalyzer(fund_code)
                        analysis = analyzer.get_full_analysis()
                        st.session_state['current_analysis'] = analysis
                        # 更新历史记录
                        base_info = analysis['基金基础信息']
                        save_history(
                            fund_code=fund_code,
                            fund_name=base_info['基金名称'],
                            latest_nav=base_info['最新净值'],
                            nav_date=base_info['净值更新日期']
                        )
                        position_path = "data/position.csv"
                        if os.path.exists(position_path):
                            pos_df = pd.read_csv(position_path, encoding="utf-8-sig")
                            pos_df, _ = refresh_positions_with_latest(pos_df, target_fund_code=fund_code)
                            pos_df.to_csv(position_path, index=False, encoding="utf-8-sig")
                        st.success("✅ 数据更新完成（历史+持仓已同步）")
                        st.experimental_rerun()
            with col3:
                pass  # 占位符
            with col4:
                if st.button("🗑️", key=f"delete_{fund_code}_{idx}", use_container_width=True, help="删除此记录"):
                    delete_history(fund_code)
                    st.success("✅ 记录已删除")
                    st.experimental_rerun()
    else:
        st.info("暂无历史分析记录，请先在「基金一键分析」页面分析基金")

# ========== 标签3：持仓盈亏管理 ==========
with _tabs[2]:
    st.markdown('<div id="position-main"></div>', unsafe_allow_html=True)
    st.title("持仓盈亏管理")
    st.markdown("---")
    
    # 加载持仓数据
    position_path = "data/position.csv"
    position_df = pd.read_csv(position_path, encoding="utf-8-sig")
    position_df = ensure_position_columns(position_df)
    if not position_df.empty:
        position_df['基金代码'] = position_df['基金代码'].astype(str).str.zfill(6)
    
    # 1. 新增持仓表单
    with st.expander("➕ 新增/修改持仓", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            add_fund_code = st.text_input("基金代码", max_chars=6, placeholder="6位基金代码")
        with col2:
            add_cost = st.number_input("持仓成本", min_value=0.0001, step=0.0001, format="%.4f")
        with col3:
            add_share = st.number_input("持仓份额", min_value=1, step=1)
        with col4:
            add_buy_date = st.date_input("买入日期", value=datetime.now())
        
        add_btn = st.button("✅ 保存持仓", use_container_width=True, type="primary")
        if add_btn and add_fund_code:
            # 获取基金名称和最新净值
            with st.spinner("正在获取基金最新数据..."):
                analyzer = UniversalFundAnalyzer(add_fund_code)
                fund_name = analyzer.fund_name
                latest_nav = analyzer.history_nav['单位净值'].iloc[-1] if not analyzer.history_nav.empty else add_cost
                # 计算浮盈浮亏
                profit = (latest_nav - add_cost) * add_share
                profit_rate = (latest_nav / add_cost - 1) * 100
                
                # 保存到持仓文件
                if add_fund_code in position_df['基金代码'].values:
                    # 更新已有持仓
                    position_df.loc[position_df['基金代码'] == add_fund_code, [
                        "基金名称", "持仓成本", "持仓份额", "买入日期", "最新净值", "浮盈浮亏", "浮盈浮亏比例"
                    ]] = [
                        fund_name, add_cost, add_share, add_buy_date.strftime("%Y-%m-%d"), latest_nav, profit, profit_rate
                    ]
                else:
                    # 新增持仓
                    new_row = pd.DataFrame([{
                        "基金代码": add_fund_code,
                        "基金名称": fund_name,
                        "持仓成本": add_cost,
                        "持仓份额": add_share,
                        "买入日期": add_buy_date.strftime("%Y-%m-%d"),
                        "最新净值": latest_nav,
                        "浮盈浮亏": profit,
                        "浮盈浮亏比例": profit_rate
                    }])
                    position_df = pd.concat([position_df, new_row], ignore_index=True)
                position_df.to_csv(position_path, index=False, encoding="utf-8-sig")
                st.success("✅ 持仓保存成功")
                st.experimental_rerun()

    with st.expander("🤖 AI批量填充持仓（粘贴自然语言）", expanded=False):
        st.caption("粘贴你的持仓描述（如微信群总结/笔记），Gemini会自动识别并生成可导入持仓。")
        ai_text = st.text_area(
            "粘贴持仓文本",
            height=220,
            placeholder="示例：国泰黄金ETF联接C(004253) 持仓成本4.3250 持仓份额545.58；华安恒生科技C(015283) 申购金额500元...",
            key="ai_positions_raw_text",
        )

        ai_col1, ai_col2 = st.columns([2, 2])
        with ai_col1:
            ai_model = st.selectbox(
                "Gemini识别模型",
                options=["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro-latest", "gemma-3-1b-it"],
                index=0,
                key="ai_position_parse_model",
            )
        with ai_col2:
            ai_api_key = st.text_input(
                "Gemini API Key",
                type="password",
                value=get_saved_api_key("Gemini"),
                placeholder=PROVIDER_CONFIG["Gemini"]["key_placeholder"],
                key="ai_position_parse_api_key",
            )

        parse_btn = st.button("🧠 Gemini识别并生成预览", use_container_width=True, key="ai_parse_positions_btn")
        if parse_btn:
            if not ai_text.strip():
                st.error("请先粘贴持仓文本")
            elif not ai_api_key.strip():
                st.error("请先填写Gemini API Key")
            else:
                with st.spinner("Gemini正在识别文本并结构化..."):
                    try:
                        as_of_date, items = parse_positions_from_text_with_gemini(
                            raw_text=ai_text.strip(),
                            api_key=ai_api_key.strip(),
                            model_name=ai_model,
                        )
                        if not items:
                            st.warning("未识别到可导入记录，请补充更明确的基金代码/成本/份额/申购金额。")
                        else:
                            st.session_state["ai_position_parse_preview"] = items
                            st.session_state["ai_position_parse_as_of"] = as_of_date
                            st.success(f"✅ 识别完成，共 {len(items)} 条")
                    except Exception as e:
                        st.error(f"识别失败：{e}")

        ai_preview = st.session_state.get("ai_position_parse_preview", [])
        if ai_preview:
            st.markdown("#### 识别预览")
            as_of = st.session_state.get("ai_position_parse_as_of", "")
            if as_of:
                st.caption(f"识别基准日期：{as_of}")
            preview_df = pd.DataFrame(ai_preview)
            preview_df = preview_df.rename(
                columns={
                    "fund_code": "基金代码",
                    "fund_name": "基金名称",
                    "hold_cost": "持仓成本",
                    "hold_share": "持仓份额",
                    "subscribe_amount": "申购金额",
                    "buy_date": "买入日期",
                    "notes": "备注",
                }
            )
            st.dataframe(preview_df, use_container_width=True, hide_index=True)

            import_mode = st.radio(
                "导入模式",
                options=["仅新增（同代码跳过）", "覆盖已有同代码"],
                index=0,
                horizontal=True,
                key="ai_import_mode",
            )

            import_btn = st.button("✅ 写入持仓", use_container_width=True, key="ai_import_positions_btn")
            if import_btn:
                with st.spinner("正在写入持仓..."):
                    try:
                        overwrite_existing = import_mode == "覆盖已有同代码"
                        new_df, ok_count = upsert_positions_from_ai_items(
                            position_df,
                            ai_preview,
                            overwrite_existing=overwrite_existing,
                        )
                        if ok_count <= 0:
                            st.warning("没有可写入的数据（可能是同代码已存在且当前模式为仅新增）。")
                        else:
                            new_df.to_csv(position_path, index=False, encoding="utf-8-sig")
                            st.success(f"✅ 已写入 {ok_count} 条持仓记录")
                            st.experimental_rerun()
                    except Exception as e:
                        st.error(f"写入失败：{e}")
    
    st.markdown("---")
    
    # 2. 盈亏总览
    if not position_df.empty:
        # 刷新最新净值
        refresh_btn = st.button("🔄 刷新所有持仓最新净值", use_container_width=True)
        if refresh_btn:
            with st.spinner("正在刷新持仓数据..."):
                position_df, cnt = refresh_positions_with_latest(position_df)
                position_df.to_csv(position_path, index=False, encoding="utf-8-sig")
                st.success(f"✅ 持仓数据刷新完成（{cnt}条）")
        
        # 总盈亏计算
        total_principal = 0.0
        total_market = 0.0
        for _idx, _row in position_df.iterrows():
            _cost = _safe_float(_row.get('持仓成本'))
            _share = _safe_float(_row.get('持仓份额'))
            _sub_amt = _safe_float(_row.get('申购金额'))
            _nav = _safe_float(_row.get('最新净值'))
            if _share is None or _share <= 0 or _nav is None or _nav <= 0:
                continue
            _cost, _share, _sub_amt = _normalize_position_row_values(_cost, _share, _sub_amt, _nav)
            _mv, _profit, _pr, _principal = _calc_profit_fields(_nav, _cost, _share, _sub_amt)
            if _mv is not None:
                total_market += float(_mv)
            if _principal is not None and _principal > 0:
                total_principal += float(_principal)

        total_profit = total_market - total_principal
        total_profit_rate = (total_profit / total_principal) * 100 if total_principal > 0 else 0
        
        # 当日/昨日盈亏判断
        now = datetime.now()
        current_hour = now.hour
        # 交易日15点前显示昨日数据，15点后显示当日数据
        if current_hour < 15:
            profit_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
            profit_title = f"📅 昨日（{profit_date}）持仓总盈亏"
        else:
            profit_date = now.strftime("%Y-%m-%d")
            profit_title = f"📅 今日（{profit_date}）持仓总盈亏"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                profit_title,
                f"¥ {format_num(total_profit, 2)}",
                f"{format_num(total_profit_rate, 2)}%",
                delta_color="inverse",
            )
        with col2:
            st.metric("持仓总本金", f"¥ {format_num(total_principal, 2)}")
        with col3:
            st.metric("持仓总市值", f"¥ {format_num(total_market, 2)}")
        
        st.markdown("---")
        
        # 持仓明细表格
        st.subheader("📋 持仓明细")
        show_df = position_df.copy()
        if "买入日期" in show_df.columns:
            show_df["无手续费退仓时间"] = show_df["买入日期"].apply(calc_fee_free_redeem_date)
        for col in ["浮盈浮亏比例", "当日涨跌幅"]:
            if col in show_df.columns:
                show_df[col] = show_df[col].apply(lambda x: f"{format_num(x, 2)}%" if pd.notna(x) else "无数据")
        desired_order = [
            "基金代码", "基金名称", "持仓成本", "持仓份额", "申购金额", "买入日期", "备注",
            "最新净值", "浮盈浮亏", "浮盈浮亏比例", "当日涨跌幅", "当日涨跌日期", "无手续费退仓时间"
        ]
        show_order = [c for c in desired_order if c in show_df.columns] + [c for c in show_df.columns if c not in desired_order]
        st.dataframe(show_df[show_order], use_container_width=True, hide_index=True)

        st.markdown("#### 📝 手动编辑持仓明细")
        editable_cols = [
            "基金代码", "基金名称", "持仓成本", "持仓份额", "申购金额", "买入日期", "备注"
        ]
        edit_df = position_df[[c for c in editable_cols if c in position_df.columns]].copy()
        edit_df["基金代码"] = edit_df["基金代码"].astype(str).str.zfill(6)

        if hasattr(st, "data_editor"):
            edited_df = st.data_editor(edit_df, use_container_width=True, hide_index=True, num_rows="dynamic", key="position_editor")
        else:
            edited_df = st.experimental_data_editor(edit_df, use_container_width=True, num_rows="dynamic", key="position_editor")
        st.session_state["position_editor_snapshot"] = edited_df.copy()

        save_edit_btn = st.button("💾 保存明细修改", use_container_width=True, key="save_position_editor_btn")
        if save_edit_btn:
            try:
                merged_df = merge_position_editor_changes(position_df.copy(), edited_df)
                merged_df, _ = refresh_positions_with_latest(merged_df)
                merged_df.to_csv(position_path, index=False, encoding="utf-8-sig")
                st.success("✅ 持仓明细已保存")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"保存失败：{e}")
        
        # 删除持仓
        st.markdown("---")
        delete_fund = ""
        del_position_btn = False
        col1, col2 = st.columns([3, 1])
        with col1:
            delete_fund = st.selectbox("选择要删除的持仓", options=position_df['基金代码'].values, format_func=lambda x: f"{x} - {position_df[position_df['基金代码']==x]['基金名称'].iloc[0]}")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            del_position_btn = st.button("🗑️ 删除该持仓", use_container_width=True, type="secondary")
        if del_position_btn and delete_fund:
            position_df = position_df[position_df['基金代码'] != delete_fund]
            position_df.to_csv(position_path, index=False, encoding="utf-8-sig")
            st.success("✅ 持仓已删除")
            st.experimental_rerun()
    else:
        st.info("暂无持仓数据，请先新增持仓")

# ========== 标签4：智能投顾决策（记忆增强） ==========
with _tabs[3]:
    st.markdown('<div id="advisor-main"></div>', unsafe_allow_html=True)
    st.title("智能投顾决策中心")
    st.markdown("---")
    st.caption("结合基金实时分析 + 你的持仓 + 历史问答记忆，给出可执行的仓位与交易建议。")

    adv_col1, adv_col2 = st.columns([2, 1])
    with adv_col1:
        advisor_fund_code = st.text_input(
            "查询基金代码（6位）",
            value=st.session_state.get("current_fund_code", ""),
            max_chars=6,
            placeholder="例如：020671",
            key="advisor_fund_code",
        )
    with adv_col2:
        risk_style = st.selectbox("风险偏好", ["保守", "稳健", "平衡", "积极"], index=1, key="advisor_risk_style")

    user_goal = st.text_area(
        "你的问题与决策目标",
        placeholder="示例：我当前持有较重，最近回撤较大，请给我接下来10个交易日是否抄底/减仓/观望的具体执行策略。",
        height=100,
        key="advisor_user_goal",
    )

    st.markdown("### AI引擎设置")
    provider_adv = st.selectbox("选择API渠道", list(PROVIDER_CONFIG.keys()), index=0, key="advisor_provider")
    provider_adv_meta = PROVIDER_CONFIG[provider_adv]
    model_adv_select = st.selectbox("选择模型", provider_adv_meta["models"], index=0, key="advisor_model_select")
    if model_adv_select == "自定义模型":
        model_adv_name = st.text_input("自定义模型名", value="", key="advisor_model_custom")
    else:
        model_adv_name = model_adv_select

    api_key_adv = st.text_input(
        f"{provider_adv} API Key",
        type="password",
        value=get_saved_api_key(provider_adv),
        placeholder=provider_adv_meta["key_placeholder"],
        key="advisor_api_key",
    )
    adv_secret_col1, adv_secret_col2 = st.columns(2)
    with adv_secret_col1:
        if st.button("💾 保存当前Key到本地(不进Git)", use_container_width=True, key="save_key_tab5"):
            if api_key_adv.strip():
                save_api_key(provider_adv, api_key_adv.strip())
                st.success(f"{provider_adv} Key 已保存到本地安全文件")
            else:
                st.warning("当前Key为空，未保存")
    with adv_secret_col2:
        if st.button("🧹 清除该渠道本地Key", use_container_width=True, key="clear_key_tab5"):
            clear_api_key(provider_adv)
            st.success(f"{provider_adv} 本地Key已清除")
    st.caption(f"本地密钥文件：{LOCAL_SECRETS_FILE.name}（已加入git忽略）")
    api_base_adv = st.text_input(
        f"{provider_adv} API Base",
        value=provider_adv_meta["api_base"],
        key="advisor_api_base",
    )

    if provider_adv == "Gemini":
        with st.expander("Gemini跨网与合规检测", expanded=False):
            if st.button("检测当前网络状态", key="gemini_check_btn_tab5"):
                session_adv = build_http_session(
                    st.session_state.get("net_proxy_url", "").strip(),
                    bool(st.session_state.get("net_use_system_proxy", False)),
                )
                geo_adv = detect_exit_geo(session_adv, timeout_sec=int(st.session_state.get("net_connect_timeout", 6)))
                st.session_state["gemini_net_tab5"] = {
                    "public_ip": geo_adv["ip"],
                    "country": geo_adv["country_name"],
                    "country_code": geo_adv["country_code"],
                    "gemini_reachable": detect_network_context()["gemini_reachable"],
                    "network_note": f"来源: {geo_adv['source']}",
                }

            net_adv = st.session_state.get("gemini_net_tab5")
            if net_adv:
                st.caption(
                    f"公网IP: {net_adv['public_ip']} | 出口地区: {net_adv['country']}({net_adv['country_code']}) | "
                    f"Gemini连通性: {'可达' if net_adv['gemini_reachable'] else '不可达/超时'}"
                )
                st.caption(
                    f"策略: proxy={'显式' if st.session_state.get('net_proxy_url','').strip() else ('系统' if st.session_state.get('net_use_system_proxy', False) else '关闭')} | "
                    f"require_country={st.session_state.get('net_require_country', '') or '无'} | "
                    f"allow_unknown={st.session_state.get('net_allow_unknown_country', True)}"
                )
                if net_adv["country_code"] == "CN":
                    st.warning(net_adv["network_note"])
                else:
                    st.info(net_adv["network_note"])

        gemini_confirm_tab5 = st.checkbox(
            "我已确认当前网络与账号使用策略合规，允许调用Gemini",
            value=False,
            key="gemini_confirm_tab5",
        )
    else:
        gemini_confirm_tab5 = True

    analyze_decision_btn = st.button("🚀 生成持仓决策建议", use_container_width=True, type="primary", key="advisor_run")

    if analyze_decision_btn:
        if not advisor_fund_code or len(advisor_fund_code.strip()) != 6:
            st.error("请输入有效的6位基金代码")
        elif not user_goal.strip():
            st.error("请先输入你的问题与决策目标")
        elif not api_key_adv.strip():
            st.error("请先填写API Key")
        elif not model_adv_name.strip():
            st.error("请先选择或填写模型")
        elif provider_adv == "Gemini" and not gemini_confirm_tab5:
            st.error("请先完成Gemini合规确认勾选")
        else:
            advisor_hint = st.empty()
            preflight_ok_adv = True
            if provider_adv == "Gemini":
                preflight_ok_adv = do_gemini_preflight_with_ui(api_key_adv.strip(), model_adv_name.strip(), advisor_hint)
                if not preflight_ok_adv:
                    st.error("Gemini预检未通过，请检查网络/代理设置后重试")

            if preflight_ok_adv:
                with st.spinner("正在抓取基金数据并生成个性化决策..."):
                    try:
                        advisor_hint.info("步骤2/3：正在抓取基金与持仓数据...")
                        advisor_analyzer = UniversalFundAnalyzer(advisor_fund_code.strip())
                        advisor_analysis = advisor_analyzer.get_full_analysis()

                        position_path = "data/position.csv"
                        advisor_positions = pd.read_csv(position_path, encoding="utf-8-sig") if os.path.exists(position_path) else pd.DataFrame()
                        if not advisor_positions.empty:
                            advisor_positions = advisor_positions[advisor_positions["基金代码"].astype(str).str.zfill(6) == advisor_fund_code.strip()]

                        advisor_hint.info("步骤3/3：模型决策中，请稍候...")
                        decision_text = run_investment_decision(
                            provider=provider_adv,
                            api_key=api_key_adv.strip(),
                            model_name=model_adv_name.strip(),
                            api_base=api_base_adv.strip(),
                            analysis=advisor_analysis,
                            positions_df=advisor_positions,
                            user_goal=user_goal.strip(),
                            risk_style=risk_style,
                        )

                        st.session_state["advisor_decision"] = decision_text
                        st.session_state["advisor_analysis_cache"] = advisor_analysis

                        save_investor_memory(
                            {
                                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "provider": provider_adv,
                                "model": model_adv_name.strip(),
                                "fund_code": advisor_fund_code.strip(),
                                "risk_style": risk_style,
                                "user_goal": user_goal.strip(),
                                "decision": decision_text,
                            }
                        )

                        st.success("✅ 决策建议已生成并写入记忆")
                        advisor_hint.empty()
                    except Exception as e:
                        advisor_hint.empty()
                        st.error(f"生成失败：{e}")

    if "advisor_decision" in st.session_state:
        st.markdown("---")
        st.subheader("📌 AI决策建议")
        font_px = int(st.session_state.get("ui_output_font_px", 16))
        st.markdown(
            f"<div style='font-size:{font_px}px; line-height:1.8'>{st.session_state['advisor_decision']}</div>",
            unsafe_allow_html=True,
        )
        st.download_button(
            "⬇️ 下载决策建议(.txt)",
            data=st.session_state["advisor_decision"].encode("utf-8"),
            file_name=f"智能投顾决策_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
            key="advisor_download",
        )

    st.markdown("---")
    st.subheader("🧾 最近记忆（用于连续决策）")
    memories = load_recent_memories(limit=10)
    if memories:
        memory_df = pd.DataFrame(memories)
        show_cols = [c for c in ["time", "fund_code", "risk_style", "provider", "model", "user_goal"] if c in memory_df.columns]
        st.dataframe(memory_df[show_cols], use_container_width=True, hide_index=True)
    else:
        st.info("暂无记忆记录。先在本页生成一次决策建议后会自动保存。")