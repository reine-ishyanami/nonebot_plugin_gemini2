import json
from nonebot import require, logger

require("nonebot_plugin_alconna")
require("nonebot_plugin_uninfo")
require("nonebot_plugin_htmlrender")
require("nonebot_plugin_localstore")
require("nonebot_plugin_apscheduler")
require("nonebot_plugin_waiter")

import os
import traceback
from google.genai.types import (
    ContentListUnion,
    ContentListUnionDict,
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    SafetySetting,
    Tool,
    GoogleSearch,
)
from nonebot.adapters import Bot
from nonebot.plugin import PluginMetadata
from nonebot_plugin_alconna import (
    Alconna,
    Args,
    Audio,
    Image,
    Match,
    Query,
    Reply,
    Subcommand,
    Text,
    UniMessage,
    UniMsg,
    on_alconna,
    Option,
)
from nonebot_plugin_alconna.builtins.extensions.reply import ReplyMergeExtension
from nonebot_plugin_htmlrender import md_to_pic, text_to_pic
from nonebot_plugin_apscheduler import scheduler
from nonebot_plugin_uninfo import UniSession, Session
from nonebot_plugin_waiter import waiter
import nonebot_plugin_localstore as store
from google.genai import Client
from httpx import AsyncClient
from .config import plugin_config


__plugin_meta__ = PluginMetadata(
    name="nonebot_plugin_gemini2",
    description="gemini聊天，支持文本，图片生成",
    usage="gemini gen/chat/search 文本 图片",
)


gemini_cmd = on_alconna(
    Alconna(
        "gemini",
        Subcommand(
            "chat",
            Option("-t|--text", help_text="回复文本时是否输出文本"),
            Option("-c|--conversation", help_text="是否持续对话"),
            Args["problem", str],
            Args["image?", Image],
            help_text="文本生成",
        ),
        Subcommand(
            "gen",
            Option("-t|--text", help_text="回复文本时是否输出文本"),
            Args["problem", str],
            Args["image?", Image],
            help_text="图片生成",
        ),
        Subcommand(
            "search",
            Option("-t|--text", help_text="回复文本时是否输出文本"),
            Args["problem", str],
            Args["image?", Image],
            help_text="通过搜索生成文本",
        ),
        Subcommand(
            "listen",
            Option("-t|--text", help_text="回复文本时是否输出文本"),
            Args["problem", str],
            Args["audio", Audio],
            help_text="分析音频输出文本",
        ),
    ),
    extensions=[ReplyMergeExtension()],
    use_cmd_start=True,
    response_self=True,
)

os.environ["http_proxy"] = plugin_config.proxy
os.environ["https_proxy"] = plugin_config.proxy

_GEMINI_CLIENT = Client(
    api_key=plugin_config.gemini_api_key,
    http_options={"api_version": "v1alpha", "timeout": 120_000, "headers": {"transport": "rest"}},
)

_HTTP_CLIENT = AsyncClient()

saerch_count_file = store.get_config_file("nonebot_plugin_gemini2", "search_count.json")
gen_count_file = store.get_config_file("nonebot_plugin_gemini2", "gen_count.json")
listen_count_file = store.get_config_file("nonebot_plugin_gemini2", "listen_count.json")

if saerch_count_file.exists():
    json_text = saerch_count_file.read_text()
    search_count_dict: dict[str, int] = json.loads(json_text if json_text else "{}")
else:
    search_count_dict = {}

if gen_count_file.exists():
    json_text = gen_count_file.read_text()
    gen_count_dict: dict[str, int] = json.loads(json_text if json_text else "{}")
else:
    gen_count_dict = {}

if listen_count_file.exists():
    json_text = listen_count_file.read_text()
    listen_count_dict: dict[str, int] = json.loads(json_text if json_text else "{}")
else:
    listen_count_dict = {}


@gemini_cmd.assign("chat")
async def handle_gemini_chat(
    problem: Match[str], image: Match[Image], text=Query("chat.text"), conversation=Query("chat.conversation")
):
    problem_text = problem.result
    parts = []
    parts.append(Part.from_text(text=problem_text))
    if image.available:
        image_content = image.result
        url = str(image_content.url)
        data = await _HTTP_CLIENT.get(url)
        parts.append(Part.from_bytes(data=data.content, mime_type="image/jpeg"))

    contents = [{"role": "user", "parts": parts}]

    try:
        current_text, current_msg_id = await chat_handler(contents, not text.available)
    except Exception as _:
        await UniMessage.image(raw=await text_to_pic(traceback.format_exc())).finish()

    if not conversation.available:
        await gemini_cmd.finish()

    await UniMessage.text("你可以回复响应消息内容来持续对话，一轮对话最多7次对话").send()

    @waiter(["message"], keep_session=True)
    async def receive(msg: UniMsg):
        reply = msg.get(Reply)
        if not reply:
            logger.info("receive empty reply, stop conversation")
            return 0
        if current_msg_id == str(reply[0].id):
            return msg
        else:
            logger.info("receive unexpected reply, stop conversation")
            return 0

    async for msg in receive(timeout=120, retry=5, prompt=""):
        if msg == 0:
            await UniMessage.text("对话已结束").finish()
        if msg is not None:
            continue_text = msg[Text, 0].text
            if continue_text.strip():
                contents.append({"role": "model", "parts": [Part.from_text(text=current_text)]})
                contents.append({"role": "user", "parts": [Part.from_text(text=continue_text)]})
                try:
                    current_text, current_msg_id = await chat_handler(contents, not text.available)
                except Exception as _:
                    await UniMessage.image(raw=await text_to_pic(traceback.format_exc())).finish()
            continue
        await UniMessage.text("输入超时，对话结束").finish()
    else:
        await UniMessage.text("对话已结束").finish()


async def chat_handler(contents: ContentListUnion | ContentListUnionDict, send_image: bool) -> tuple[str, str]:
    response = await _GEMINI_CLIENT.aio.models.generate_content(
        model=plugin_config.gemini_model,
        contents=contents,
        config=GenerateContentConfig(
            system_instruction=plugin_config.gemini_prompt,
            response_modalities=["Text"],
            safety_settings=[
                SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.OFF),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.OFF),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.OFF),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.OFF),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY, threshold=HarmBlockThreshold.OFF),
            ],
        ),
    )

    current_msg_id = ""
    current_text = ""
    for part in response.candidates[0].content.parts:  # type: ignore
        if part.text is not None:
            current_text = part.text
            if send_image:
                res = await UniMessage.image(raw=await md_to_pic(part.text)).send()
            else:
                res = await UniMessage.text(part.text).send()
            current_msg_id = str(res.msg_ids.pop()["message_id"])

    return current_text, current_msg_id


@gemini_cmd.assign("gen")
async def handle_gemini_gen(
    bot: Bot,
    problem: Match[str],
    image: Match[Image],
    text=Query("gen.text"),
    session: Session = UniSession(),
):
    if plugin_config.gemini_gen_max_count >= 0:
        user_id = session.user.id
        count = gen_count_dict.get(user_id, 0)
        if count >= plugin_config.gemini_gen_max_count:
            await UniMessage.text("图片生成次数已达上限").finish()
        else:
            await UniMessage.text(f"今日剩余图片生成次数{plugin_config.gemini_gen_max_count - count - 1}").send()
        if user_id not in bot.config.superusers:
            logger.info(f"user {user_id} generate count: {count}")
            gen_count_dict[user_id] = count + 1
            await save_gen_count()
        else:
            logger.info(f"user {user_id} is superuser, not count")
    problem_text = problem.result
    parts = []
    parts.append(Part.from_text(text=problem_text))
    if image.available:
        image_content = image.result
        url = str(image_content.url)
        data = await _HTTP_CLIENT.get(url)
        parts.append(Part.from_bytes(data=data.content, mime_type="image/jpeg"))

    try:
        response = await _GEMINI_CLIENT.aio.models.generate_content(
            model=plugin_config.gemini_gen_model,
            contents=[{"role": "user", "parts": parts}],
            config=GenerateContentConfig(
                response_modalities=["Text", "Image"],
                safety_settings=[
                    SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.OFF),
                    SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.OFF),
                    SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.OFF
                    ),
                    SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.OFF
                    ),
                    SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY, threshold=HarmBlockThreshold.OFF
                    ),
                ],
            ),
        )

        for part in response.candidates[0].content.parts:  # type: ignore
            if part.text is not None:
                if not text.available:
                    await UniMessage.image(raw=await md_to_pic(part.text)).send()
                else:
                    await UniMessage.text(part.text).send()
            if part.inline_data is not None:
                await UniMessage.image(
                    raw=part.inline_data.data,
                    mimetype=part.inline_data.mime_type,
                ).send()
    except Exception as _:
        await UniMessage.image(raw=await text_to_pic(traceback.format_exc())).finish()


@gemini_cmd.assign("search")
async def handle_gemini_search(
    bot: Bot,
    problem: Match[str],
    image: Match[Image],
    text=Query("search.text"),
    session: Session = UniSession(),
):
    if plugin_config.gemini_search_max_count >= 0:
        user_id = session.user.id
        count = search_count_dict.get(user_id, 0)
        if count >= plugin_config.gemini_search_max_count:
            await UniMessage.text("搜索次数已达上限").finish()
        else:
            await UniMessage.text(f"今日剩余搜索次数{plugin_config.gemini_search_max_count - count - 1}").send()
        if user_id not in bot.config.superusers:
            logger.info(f"user {user_id} search count: {count}")
            search_count_dict[user_id] = count + 1
            await save_search_count()
        else:
            logger.info(f"user {user_id} is superuser, not count")
    problem_text = problem.result
    parts = []
    parts.append(Part.from_text(text=problem_text))
    if image.available:
        image_content = image.result
        url = str(image_content.url)
        data = await _HTTP_CLIENT.get(url)
        parts.append(Part.from_bytes(data=data.content, mime_type="image/jpeg"))
    try:
        response = await _GEMINI_CLIENT.aio.models.generate_content(
            model=plugin_config.gemini_model,
            contents=[{"role": "user", "parts": parts}],
            config=GenerateContentConfig(
                system_instruction=plugin_config.gemini_prompt,
                response_modalities=["Text"],
                tools=[Tool(google_search=GoogleSearch())],
                safety_settings=[
                    SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.OFF),
                    SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.OFF),
                    SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.OFF
                    ),
                    SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.OFF
                    ),
                    SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY, threshold=HarmBlockThreshold.OFF
                    ),
                ],
            ),
        )

        for part in response.candidates[0].content.parts:  # type: ignore
            if part.text is not None:
                if not text.available:
                    await UniMessage.image(raw=await md_to_pic(part.text, width=1000)).send()
                else:
                    await UniMessage.text(part.text).send()
    except Exception as _:
        await UniMessage.image(raw=await text_to_pic(traceback.format_exc())).finish()


@gemini_cmd.assign("listen")
async def handle_gemini_listen(
    bot: Bot,
    problem: Match[str],
    audio: Match[Audio],
    text=Query("search.text"),
    session: Session = UniSession(),
):
    if plugin_config.gemini_audio_max_count >= 0:
        user_id = session.user.id
        count = search_count_dict.get(user_id, 0)
        if count >= plugin_config.gemini_audio_max_count:
            await UniMessage.text("聆听次数已达上限").finish()
        else:
            await UniMessage.text(f"今日剩余聆听次数{plugin_config.gemini_audio_max_count - count - 1}").send()
        if user_id not in bot.config.superusers:
            logger.info(f"user {user_id} listen count: {count}")
            search_count_dict[user_id] = count + 1
            await save_search_count()
        else:
            logger.info(f"user {user_id} is superuser, not count")
    problem_text = problem.result
    parts = []
    parts.append(Part.from_text(text=problem_text))
    if audio.available:
        audio_content = audio.result
        url = str(audio_content.url)
        data = await _HTTP_CLIENT.get(url)
        parts.append(Part.from_bytes(data=data.content, mime_type="audio/mp3"))
    try:
        response = await _GEMINI_CLIENT.aio.models.generate_content(
            model=plugin_config.gemini_model,
            contents=[{"role": "user", "parts": parts}],
            config=GenerateContentConfig(
                system_instruction=plugin_config.gemini_prompt,
                response_modalities=["Text"],
                safety_settings=[
                    SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.OFF),
                    SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.OFF),
                    SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.OFF
                    ),
                    SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.OFF
                    ),
                    SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY, threshold=HarmBlockThreshold.OFF
                    ),
                ],
            ),
        )

        for part in response.candidates[0].content.parts:  # type: ignore
            if part.text is not None:
                if not text.available:
                    await UniMessage.image(raw=await md_to_pic(part.text, width=1000)).send()
                else:
                    await UniMessage.text(part.text).send()
    except Exception as _:
        await UniMessage.image(raw=await text_to_pic(traceback.format_exc())).finish()


async def save_search_count():
    saerch_count_file.write_text(json.dumps(search_count_dict))


async def save_gen_count():
    gen_count_file.write_text(json.dumps(gen_count_dict))


@scheduler.scheduled_job(
    "cron",
    hour="0",
    id="reset_gemini_search_count",
)
async def run_every_2_hour():
    saerch_count_file.write_text("{}")
    search_count_dict.clear()
    logger.info("reset search count")
    gen_count_file.write_text("{}")
    gen_count_dict.clear()
    logger.info("reset gen count")
