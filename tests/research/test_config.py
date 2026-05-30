import os
from unittest.mock import patch

from trendspec.research.config import ResearchSettings


def test_defaults():
    with patch.dict(os.environ, {}, clear=False):
        s = ResearchSettings(_env_file=None)
        assert s.llm_base_url
        assert s.out_dir


def test_env_override():
    with patch.dict(os.environ, {
        "RESEARCH_LLM_BASE_URL": "https://api.deepseek.com/v1",
        "RESEARCH_LLM_MODEL": "deepseek-x",
        "RESEARCH_OUT_DIR": "/tmp/research_out",
    }):
        s = ResearchSettings(_env_file=None)
        assert s.llm_base_url == "https://api.deepseek.com/v1"
        assert s.llm_model == "deepseek-x"
        assert s.out_dir == "/tmp/research_out"
