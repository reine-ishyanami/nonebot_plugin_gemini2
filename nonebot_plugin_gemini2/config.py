from pydantic import BaseModel
from nonebot import get_plugin_config


class Config(BaseModel):

    gemini_api_key: str
    """API KEY"""
    gemini_model: str = "gemini-2.0-flash-exp"
    """API Model"""
    gemini_prompt: str = ""
    """API 提示"""
    gemini_gen_model: str = "gemini-2.0-flash-exp-image-generation"
    """API 生成模型"""
    gemini_search_max_count: int = 3
    """每人每天调用搜索最大次数，超管不受限制"""
    gemini_gen_max_count: int = 3
    """每人每天调用图片生成最大次数，超管不受限制"""
    proxy: str = "http://127.0.0.1:7890"
    """代理地址"""

plugin_config = get_plugin_config(Config)
