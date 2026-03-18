"""Simple API chat sandbox for Gemini / DeepSeek / Doubao.

Usage examples:
  python api_chat_sandbox.py --provider gemini --model gemini-2.5-pro
  python api_chat_sandbox.py --provider deepseek --model deepseek-chat
  python api_chat_sandbox.py --provider doubao --model ep-xxxx
    python api_chat_sandbox.py --provider gemini --model gemini-3-flash --translate-zh

API keys are read from environment variables by default:
  GEMINI_API_KEY, DEEPSEEK_API_KEY, DOUBAO_API_KEY

For Gemini, this script can use official SDK first:
    pip install google-genai
"""

import argparse
import getpass
import json
import os
import re
import sys
import time

import requests


PROVIDERS = {
    "gemini": {
        "default_model": "gemini-3-flash",
        "api_base": "https://generativelanguage.googleapis.com/v1beta",
        "env_key": "GEMINI_API_KEY",
    },
    "deepseek": {
        "default_model": "deepseek-chat",
        "api_base": "https://api.deepseek.com/v1",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "doubao": {
        "default_model": "doubao-1-5-lite-32k-250115",
        "api_base": "https://ark.cn-beijing.volces.com/api/v3",
        "env_key": "DOUBAO_API_KEY",
    },
}

GEMINI_MODEL_PRESETS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemma-3-1b-it",
    "gemini-2.5-flash",
    "gemini-2.5-pro-latest",
]

GEMINI_MODEL_ALIASES = {
    # Common user aliases that may not exist in v1beta generateContent.
    "gemini-3-flash": "gemini-2.0-flash",
    "gemini-3": "gemini-2.0-flash",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Simple multi-provider chat sandbox")
    parser.add_argument("--provider", choices=sorted(PROVIDERS.keys()), default="gemini")
    parser.add_argument("--model", default=None, help="Model name (or endpoint id for doubao)")
    parser.add_argument(
        "--fallback-model",
        default=None,
        help="Fallback model if primary model returns 429/403/404",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List built-in recommended model presets and exit",
    )
    parser.add_argument("--api-base", default=None, help="Override API base URL")
    parser.add_argument("--api-key", default=None, help="API key (prefer env vars for security)")
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--timeout", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument(
        "--single-turn",
        action="store_true",
        help="Do not keep chat history. Each question is processed independently.",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Optional text file path for one-shot analysis prompt.",
    )
    parser.add_argument(
        "--use-system-proxy",
        action="store_true",
        help="Use system proxy settings (default: disabled)",
    )
    parser.add_argument(
        "--proxy",
        default=None,
        help="Explicit proxy URL, e.g. http://127.0.0.1:7890 (overrides system proxy)",
    )
    parser.add_argument(
        "--gemini-transport",
        choices=["auto", "sdk", "http"],
        default="auto",
        help="Gemini transport mode: official SDK or raw HTTP",
    )
    parser.add_argument(
        "--require-country",
        default="",
        help="Optional ISO country code requirement for exit IP, e.g. US. If not matched, program exits.",
    )
    parser.add_argument(
        "--allow-unknown-country",
        action="store_true",
        help="Allow execution when country lookup is unavailable (country=unknown)",
    )
    parser.add_argument(
        "--translate-zh",
        action="store_true",
        help="Translate final assistant response into Simplified Chinese before output",
    )
    parser.add_argument(
        "--translate-mode",
        choices=["native", "secondary"],
        default="native",
        help=(
            "Translation mode for --translate-zh: native=ask model to answer Chinese directly (one call), "
            "secondary=translate after answer (two calls)."
        ),
    )
    return parser.parse_args()


def resolve_gemini_transport(requested_mode):
    """Resolve transport mode with Python version compatibility.

    google-genai requires Python >= 3.9.
    """
    if requested_mode in ("sdk", "http"):
        if requested_mode == "sdk" and sys.version_info < (3, 9):
            print("[WARN] SDK mode requires Python >= 3.9. Falling back to HTTP mode.")
            return "http"
        return requested_mode

    # auto mode
    if sys.version_info < (3, 9):
        return "http"
    return "sdk"


def build_http_session(args):
    """Build requests session with deterministic proxy behavior."""
    session = requests.Session()

    if args.proxy:
        session.trust_env = False
        session.proxies.update({
            "http": args.proxy,
            "https": args.proxy,
        })
        return session

    if args.use_system_proxy:
        session.trust_env = True
        return session

    # Default: disable proxy and env inheritance.
    session.trust_env = False
    session.proxies.update({"http": "", "https": ""})
    return session


def preflight_endpoint(session, provider, api_base, timeout):
    """Fast connectivity check before waiting on first user question."""
    target = "https://generativelanguage.googleapis.com" if provider == "gemini" else api_base
    try:
        resp = session.get(target, timeout=(5, min(8, timeout)))
        return True, "HTTP {}".format(resp.status_code)
    except Exception as exc:
        return False, str(exc)


def detect_exit_geo(session, timeout):
    """Best-effort geolocation lookup for current egress IP."""
    urls = [
        "https://ipapi.co/json/",
        "https://ipinfo.io/json",
        "https://ipwho.is/",
        "https://api.ip.sb/geoip",
    ]
    for url in urls:
        try:
            resp = session.get(url, timeout=(5, min(10, timeout)))
            if not resp.ok:
                continue
            data = resp.json()
            ip = str(data.get("ip", "未知"))
            # Different providers use different keys.
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

            # Normalize some provider values.
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


def call_gemini(session, messages, api_key, model_name, timeout, temperature, max_output_tokens):
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}".format(
        model_name, api_key
    )

    # Convert chat history into Gemini-native message format.
    system_text = ""
    contents = []
    for msg in messages:
        role = msg.get("role", "user").lower()
        content = msg.get("content", "")
        if role == "system":
            system_text = content
            continue
        gemini_role = "model" if role in ("assistant", "model") else "user"
        contents.append({"role": gemini_role, "parts": [{"text": content}]})

    if not contents:
        contents = [{"role": "user", "parts": [{"text": "Hello"}]}]

    # Some Gemma endpoints do not support system/developer instruction.
    use_system_instruction = not str(model_name).lower().startswith("gemma-")
    if system_text and not use_system_instruction:
        # Inline system guidance into the first user turn for compatibility.
        injected = False
        for item in contents:
            if item.get("role") == "user":
                first = item.get("parts", [{}])[0]
                old_text = first.get("text", "")
                first["text"] = "[SYSTEM]\n{}\n\n[USER]\n{}".format(system_text, old_text)
                injected = True
                break
        if not injected:
            contents.insert(0, {"role": "user", "parts": [{"text": "[SYSTEM]\n{}".format(system_text)}]})

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    if system_text and use_system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_text}]}

    response = session.post(endpoint, json=payload, timeout=(8, timeout))
    response.raise_for_status()
    data = response.json()

    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError("Gemini returned no candidates")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "\n".join(p.get("text", "") for p in parts if p.get("text"))
    if not text.strip():
        raise ValueError("Gemini returned empty content")
    return text.strip()


def list_gemini_generate_models(session, api_key, timeout):
    """List Gemini models that support generateContent in v1beta."""
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models?key={}".format(api_key)
    response = session.get(endpoint, timeout=(8, timeout))
    response.raise_for_status()
    data = response.json()

    models = []
    for m in data.get("models", []):
        name = str(m.get("name", ""))
        short_name = name.split("/")[-1] if "/" in name else name
        methods = m.get("supportedGenerationMethods", []) or []
        if "generateContent" in methods and short_name:
            models.append(short_name)
    return models


def resolve_gemini_model(session, api_key, requested_model, timeout):
    """Resolve user model name to an actually available generateContent model."""
    requested = (requested_model or "").strip()
    if not requested:
        return requested, ""

    alias_target = GEMINI_MODEL_ALIASES.get(requested, requested)
    try:
        available = list_gemini_generate_models(session, api_key, timeout)
    except Exception:
        # If model listing fails, still apply alias mapping as a best effort.
        if alias_target != requested:
            return alias_target, "alias"
        return requested, ""

    if requested in available:
        return requested, ""
    if alias_target in available:
        return alias_target, "alias"

    preferred = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
    ]
    for cand in preferred:
        if cand in available:
            return cand, "auto"

    # Last resort: keep requested model unchanged if nothing matched.
    return requested, ""


def suggest_alternate_gemini_model(session, api_key, current_model, timeout):
    """Suggest an alternate Gemini model to recover from 429/404/403."""
    try:
        available = list_gemini_generate_models(session, api_key, timeout)
    except Exception:
        available = []

    candidates = [
        "gemma-3-1b-it",
        "gemini-1.5-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
    ]
    for cand in candidates:
        if cand != current_model and ((not available) or (cand in available)):
            return cand

    if available:
        for cand in available:
            if cand != current_model:
                return cand
    return ""


def build_gemini_retry_queue(session, api_key, current_model, fallback_model, timeout):
    """Build a bounded model retry queue for one user question."""
    queue = []

    def add_unique(name):
        if name and name not in queue:
            queue.append(name)

    add_unique(current_model)
    add_unique(fallback_model)

    try:
        available = list_gemini_generate_models(session, api_key, timeout)
    except Exception:
        available = []

    preferred = [
        "gemma-3-1b-it",
        "gemini-1.5-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
    ]
    for cand in preferred:
        if (not available) or (cand in available):
            add_unique(cand)

    for cand in available:
        add_unique(cand)

    # Keep retries bounded to avoid long loops.
    return queue[:6]


def is_fund_wave_request(user_input):
    text = str(user_input or "")
    keys = ["波段", "核心规则", "基金", "止盈", "止损"]
    score = sum(1 for k in keys if k in text)
    return score >= 3


def is_fund_wave_answer_valid(answer_text):
    text = str(answer_text or "")
    if not text.strip():
        return False

    # Must include one explicit conclusion option.
    options = ["适合申购", "暂时不建议申购", "完全不建议申购"]
    if not any(opt in text for opt in options):
        return False

    # Must contain the three required sections.
    required_sections = ["明确结论", "核心理由", "波段操作计划"]
    if not all(sec in text for sec in required_sections):
        return False

    # Avoid very short/empty boilerplate plans.
    if len(text) < 120:
        return False

    return True


def rewrite_fund_wave_answer(
    session,
    provider,
    api_key,
    model_name,
    api_base,
    gemini_transport,
    timeout,
    source_question,
    draft_answer,
):
    """Rewrite model output into strict 3-part Chinese fund-wave format."""
    prompt = (
        "请将下面草稿改写为严格格式，不要添加任何多余内容。\n"
        "输出必须且仅包含以下三部分：\n"
        "1. 明确结论：只能是 适合申购 / 暂时不建议申购 / 完全不建议申购 之一\n"
        "2. 核心理由：最多3条，必须同时覆盖两类依据：\n"
        "   - 数据面：近7天单日涨跌、累计周期表现、择时指标、赛道逻辑、交易成本\n"
        "   - 事件面：当日相关政策/时事对该基金赛道的影响（方向、强度、持续性）\n"
        "3. 波段操作计划：给出止盈位、止损位、建议持有周期、最佳赎回时间节点与触发条件\n"
        "补充约束：\n"
        "- 保留原草稿核心观点，不可凭空新增与题目无关结论；\n"
        "- 若草稿未提供充分当日政策/时事信息，必须明确写出“事件信息不足”及潜在偏差；\n"
        "- 输出中文、简洁、数据口径一致，不要出现模糊措辞，确保可执行。\n\n"
        "[原问题]\n{}\n\n[草稿]\n{}".format(source_question, draft_answer)
    )
    rewrite_messages = [
        {"role": "system", "content": "你是严谨的中文基金投顾编辑器，只做格式化改写。"},
        {"role": "user", "content": prompt},
    ]

    if provider == "gemini":
        if gemini_transport == "sdk":
            return call_gemini_sdk(rewrite_messages, api_key, model_name, timeout)
        return call_gemini(
            session,
            rewrite_messages,
            api_key,
            model_name,
            timeout,
            temperature=0.0,
            max_output_tokens=2048,
        )

    return call_openai_compatible(
        session,
        rewrite_messages,
        api_key,
        model_name,
        api_base,
        timeout,
        temperature=0.0,
        max_output_tokens=2048,
    )


def call_gemini_sdk(messages, api_key, model_name, timeout):
    """Call Gemini via official google-genai SDK."""
    try:
        from google import genai
    except Exception as exc:
        raise RuntimeError(
            "google-genai not installed. Run: pip install google-genai"
        ) from exc

    # Flatten multi-turn history into a single prompt for stable behavior.
    text_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        text_parts.append("[{}]\n{}".format(role, content))

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents="\n\n".join(text_parts),
    )

    text = getattr(response, "text", "")
    if text and text.strip():
        return text.strip()

    # Some SDK/model combos return candidates/parts instead of response.text.
    try:
        candidates = getattr(response, "candidates", []) or []
        if candidates:
            parts = getattr(candidates[0].content, "parts", [])
            merged = "\n".join(getattr(p, "text", "") for p in parts if getattr(p, "text", ""))
            if merged.strip():
                return merged.strip()
    except Exception:
        pass

    raise ValueError("Gemini SDK returned empty content")


def call_openai_compatible(session, messages, api_key, model_name, api_base, timeout, temperature, max_output_tokens):
    endpoint = api_base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": "Bearer {}".format(api_key),
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }

    response = session.post(endpoint, headers=headers, json=payload, timeout=(8, timeout))
    response.raise_for_status()
    data = response.json()

    choices = data.get("choices", [])
    if not choices:
        raise ValueError("Provider returned no choices: {}".format(json.dumps(data, ensure_ascii=False)[:300]))

    content = choices[0].get("message", {}).get("content", "")
    if not content.strip():
        raise ValueError("Provider returned empty content")
    return content.strip()


def translate_to_zh(
    session,
    source_text,
    provider,
    api_key,
    model_name,
    api_base,
    gemini_transport,
    timeout,
):
    """Translate any model output to concise Simplified Chinese."""
    translation_messages = [
        {
            "role": "system",
            "content": (
                "You are a professional translator. "
                "Translate the given content into natural Simplified Chinese. "
                "Keep numbers, percentages, dates, and tickers exactly unchanged. "
                "Do not add explanations, only output translated Chinese text."
            ),
        },
        {"role": "user", "content": source_text},
    ]

    if provider == "gemini":
        if gemini_transport == "sdk":
            return call_gemini_sdk(translation_messages, api_key, model_name, timeout)
        return call_gemini(
            session,
            translation_messages,
            api_key,
            model_name,
            timeout,
            temperature=0.0,
            max_output_tokens=4096,
        )

    return call_openai_compatible(
        session,
        translation_messages,
        api_key,
        model_name,
        api_base,
        timeout,
        temperature=0.0,
        max_output_tokens=4096,
    )


def main():
    args = parse_args()

    if args.list_models:
        print("Recommended Gemini model presets:")
        for m in GEMINI_MODEL_PRESETS:
            print(" - {}".format(m))
        print("\nYou can still pass any custom model via --model.")
        return

    # Keep process env deterministic for child requests.
    if not args.use_system_proxy:
        os.environ["HTTP_PROXY"] = ""
        os.environ["HTTPS_PROXY"] = ""
        os.environ["http_proxy"] = ""
        os.environ["https_proxy"] = ""
        os.environ["ALL_PROXY"] = ""
        os.environ["all_proxy"] = ""
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"

    session = build_http_session(args)

    meta = PROVIDERS[args.provider]

    model_name = args.model or meta["default_model"]
    api_base = args.api_base or meta["api_base"]

    api_key = args.api_key
    if not api_key:
        api_key = os.environ.get(meta["env_key"], "")

    if not api_key:
        api_key = getpass.getpass("Enter {}: ".format(meta["env_key"]))

    if not api_key:
        print("No API key provided. Exit.")
        return

    print("Provider: {} | Model: {}".format(args.provider, model_name))
    if args.proxy:
        print("Proxy mode: explicit ({})".format(args.proxy))
    else:
        print("System proxy: {}".format("enabled" if args.use_system_proxy else "disabled"))
    if args.provider == "gemini":
        gemini_transport = resolve_gemini_transport(args.gemini_transport)
        resolved_model, resolve_mode = resolve_gemini_model(session, api_key, model_name, args.timeout)
        if resolved_model != model_name:
            if resolve_mode == "alias":
                print("[INFO] Gemini model alias mapped: {} -> {}".format(model_name, resolved_model))
            else:
                print("[INFO] Gemini model auto-switched: {} -> {}".format(model_name, resolved_model))
            model_name = resolved_model
        print("Gemini transport: {}".format(gemini_transport))
    else:
        gemini_transport = "http"

    geo = detect_exit_geo(session, args.timeout)
    print(
        "Exit IP: {} | Country: {} ({}) | Source: {}".format(
            geo["ip"], geo["country_name"], geo["country_code"], geo["source"]
        )
    )

    required_country = args.require_country.strip().upper()
    if required_country:
        unknown_geo = geo["country_code"] in ("未知", "")
        if unknown_geo and args.allow_unknown_country:
            print("[WARN] Exit country unavailable, but allowed by --allow-unknown-country.")
        elif geo["country_code"] != required_country:
            print(
                "[BLOCKED] Exit country check failed: require {} but got {}.\n"
                "Please adjust your proxy route and retry.".format(required_country, geo["country_code"])
            )
            return

    ok, detail = preflight_endpoint(session, args.provider, api_base, args.timeout)
    if ok:
        print("Preflight: OK ({})".format(detail))
    else:
        print("Preflight: FAILED ({})".format(detail))
        print("Hint: if you must use proxy, prefer --proxy http://127.0.0.1:7890")

    print("Type 'exit' to quit.\n")

    system_prompt = args.system
    if args.translate_zh and args.translate_mode == "native":
        # Native mode avoids a second API call for translation and saves quota.
        system_prompt = (
            system_prompt
            + "\n\nYou must answer in concise Simplified Chinese only. "
            + "Keep numbers, percentages, dates, and ticker/code values unchanged."
        )

    if args.translate_zh:
        print("Translation mode: {}".format(args.translate_mode))

    messages = [{"role": "system", "content": system_prompt}]

    # Optional one-shot prompt from file.
    if args.input_file:
        try:
            with open(args.input_file, "r", encoding="utf-8") as f:
                file_prompt = f.read().strip()
        except Exception as exc:
            print("Failed to read input file: {}".format(exc))
            return

        if not file_prompt:
            print("Input file is empty. Exit.")
            return

        print("Loaded input file: {} ({} chars)".format(args.input_file, len(file_prompt)))
        user_inputs = [file_prompt]
    else:
        user_inputs = None

    while True:
        if user_inputs is not None:
            if not user_inputs:
                print("One-shot completed.")
                break
            user_input = user_inputs.pop(0)
            print("You> [input-file loaded]")
        else:
            user_input = input("You> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", ":q"):
                print("Bye")
                break

            # Quick helper: @filepath to load a long prompt in one command.
            if user_input.startswith("@"):
                fp = user_input[1:].strip()
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        user_input = f.read().strip()
                    print("Loaded file prompt: {} ({} chars)".format(fp, len(user_input)))
                except Exception as exc:
                    print("Assistant> Failed to load file: {}".format(exc))
                    continue

        if len(user_input) > 12000:
            print("Assistant> Note: long prompt detected ({} chars).".format(len(user_input)))

        current_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}] if args.single_turn else messages + [{"role": "user", "content": user_input}]

        start = time.time()
        print("Assistant> Thinking...", flush=True)

        answer = None
        last_http_code = ""
        tried_models = []
        http_error_stop = False

        if args.provider == "gemini":
            retry_queue = build_gemini_retry_queue(
                session=session,
                api_key=api_key,
                current_model=model_name,
                fallback_model=args.fallback_model,
                timeout=args.timeout,
            )
        else:
            retry_queue = [model_name]

        for idx, candidate_model in enumerate(retry_queue):
            tried_models.append(candidate_model)
            try:
                if args.provider == "gemini":
                    if candidate_model != model_name:
                        print("Assistant> Retry with model: {}".format(candidate_model))
                    if gemini_transport == "sdk":
                        answer = call_gemini_sdk(current_messages, api_key, candidate_model, args.timeout)
                    else:
                        answer = call_gemini(
                            session,
                            current_messages,
                            api_key,
                            candidate_model,
                            args.timeout,
                            args.temperature,
                            args.max_output_tokens,
                        )
                    model_name = candidate_model
                    break

                answer = call_openai_compatible(
                    session,
                    current_messages,
                    api_key,
                    candidate_model,
                    api_base,
                    args.timeout,
                    args.temperature,
                    args.max_output_tokens,
                )
                model_name = candidate_model
                break

            except requests.exceptions.HTTPError as exc:
                code = str(exc.response.status_code) if exc.response is not None else "unknown"
                last_http_code = code
                detail_text = exc.response.text[:800] if exc.response is not None else ""
                print("Assistant> HTTP error [{}] on model {}".format(code, candidate_model))
                if detail_text:
                    print("Details: {}".format(detail_text[:500]))

                # Retry only for model-specific or quota errors.
                can_retry = args.provider == "gemini" and code in ("403", "404", "429")
                has_next = idx < len(retry_queue) - 1
                if can_retry and has_next:
                    continue

                http_error_stop = True
                break

            except requests.exceptions.ConnectTimeout:
                print("Assistant> Connection timeout. Check network/proxy/export route.")
                http_error_stop = True
                break
            except requests.exceptions.ReadTimeout:
                print("Assistant> Read timeout. Try simpler question or lower model load.")
                http_error_stop = True
                break
            except Exception as exc:
                print("Assistant> Error: {}".format(exc))
                http_error_stop = True
                break

        if answer is None:
            if args.provider == "gemini" and last_http_code == "429":
                print(
                    "Hint: all retried Gemini models are quota-limited now. Tried: {}\n"
                    "  1) Wait for quota reset / enable billing in AI Studio\n"
                    "  2) Switch provider to deepseek/doubao in this script\n"
                    "  3) Keep --translate-mode native to minimize API calls".format(
                        ", ".join(tried_models)
                    )
                )
            elif args.provider == "gemini" and last_http_code in ("403", "404"):
                print("Hint: no available Gemini model succeeded. Tried: {}".format(", ".join(tried_models)))

            if user_inputs is not None:
                print("One-shot aborted due to request error.")
                break
            if http_error_stop:
                continue
            continue

        elapsed = time.time() - start
        final_answer = answer
        if args.translate_zh and args.translate_mode == "secondary":
            try:
                final_answer = translate_to_zh(
                    session=session,
                    source_text=answer,
                    provider=args.provider,
                    api_key=api_key,
                    model_name=model_name,
                    api_base=api_base,
                    gemini_transport=gemini_transport,
                    timeout=args.timeout,
                )
            except Exception as exc:
                print("Assistant> Translation failed, fallback to original output: {}".format(exc))

        # Fund-wave strictness guard: rewrite once if result quality/format is weak.
        if is_fund_wave_request(user_input) and not is_fund_wave_answer_valid(final_answer):
            print("Assistant> Output format check failed, rewriting to strict 3-part format...")
            try:
                repaired = rewrite_fund_wave_answer(
                    session=session,
                    provider=args.provider,
                    api_key=api_key,
                    model_name=model_name,
                    api_base=api_base,
                    gemini_transport=gemini_transport,
                    timeout=args.timeout,
                    source_question=user_input,
                    draft_answer=final_answer,
                )
                if repaired and repaired.strip():
                    final_answer = repaired.strip()
            except Exception as exc:
                print("Assistant> Rewrite failed, keep original output: {}".format(exc))

        # Soft-clean repeated blank lines for readability.
        final_answer = re.sub(r"\n{3,}", "\n\n", final_answer).strip()

        print("Assistant ({:.1f}s)> {}\n".format(elapsed, final_answer))
        if not args.single_turn:
            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": final_answer})


if __name__ == "__main__":
    main()
