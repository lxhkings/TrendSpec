"""
TrendSpec logging configuration.

Provides Chinese-first logging with rich console output.
"""

import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Custom theme with Chinese-friendly styling
TRENDSPEC_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "red bold",
        "critical": "red bold reverse",
        "debug": "dim",
        "success": "green",
        "highlight": "magenta",
        "money": "green bold",
        "date": "blue",
        "symbol": "yellow bold",
    }
)


def get_console() -> Console:
    """Get a rich console with TrendSpec theme."""
    return Console(theme=TRENDSPEC_THEME)


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
    rich_format: bool = True,
) -> logging.Logger:
    """
    Set up logging with Chinese messages and rich formatting.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path for logging
        rich_format: Whether to use rich console formatting

    Returns:
        Configured logger instance
    """
    # Get root logger
    logger = logging.getLogger("trendspec")
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler with rich formatting
    if rich_format:
        console = get_console()
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            show_level=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            markup=True,
        )
        console_handler.setLevel(level)
        # Simplified format for rich handler (rich adds its own formatting)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console_handler)
    else:
        # Fallback to standard handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "trendspec") -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Pre-configured log messages in Chinese
class LogMessages:
    """Standard log messages in Chinese."""

    # Database
    DB_CONNECTING = "[info]正在连接数据库[/info] {host}:{port} ..."
    DB_CONNECTED = "[success]数据库连接成功[/success]"
    DB_ERROR = "[error]数据库连接失败[/error]: {error}"
    DB_QUERY = "[debug]执行查询[/debug]: {query}"

    # Data
    DATA_LOADING = "[info]正在加载数据[/info]: {source}"
    DATA_LOADED = "[success]数据加载完成[/success]: {rows} 行, {cols} 列"
    DATA_CACHED = "[info]数据已缓存[/info]: {path}"
    DATA_CACHE_HIT = "[info]命中缓存[/info]: {path}"
    DATA_CACHE_MISS = "[warning]缓存未命中[/warning]: {path}"

    # Backtest
    BACKTEST_START = "[highlight]开始回测[/highlight]: {strategy}"
    BACKTEST_END = "[success]回测完成[/success]: 收益率 {return_pct:.2%}"
    BACKTEST_PROGRESS = "[info]回测进度[/info]: {current}/{total}"

    # Risk
    RISK_WARNING = "[warning]风控警告[/warning]: {message}"
    RISK_BREACH = "[error]风控触发[/error]: {message}"
    DRAWDOWN_HALT = "[critical]回撤超限，暂停交易[/critical]: 回撤 {drawdown:.2%}"

    # System
    INIT_START = "[info]初始化 TrendSpec...[/info]"
    INIT_COMPLETE = "[success]TrendSpec 初始化完成[/success]"
    CONFIG_LOADED = "[info]配置加载完成[/info]"

    # Errors
    GENERIC_ERROR = "[error]错误[/error]: {error}"
    VALIDATION_ERROR = "[error]参数验证失败[/error]: {field} - {message}"
    FILE_NOT_FOUND = "[error]文件未找到[/error]: {path}"

    # Screen
    SCREEN_START = "[highlight]开始筛选[/highlight]: {universe}"
    SCREEN_RESULT = "[success]筛选完成[/success]: 找到 {count} 只股票"


# Initialize default logger on module import
_default_logger: logging.Logger | None = None


def init_default_logger(level: int = logging.INFO, log_file: Path | None = None) -> None:
    """
    Initialize the default logger for TrendSpec.

    Should be called once at application startup.

    Args:
        level: Logging level
        log_file: Optional log file path
    """
    global _default_logger
    _default_logger = setup_logging(level=level, log_file=log_file)


def log(message: str, level: int = logging.INFO) -> None:
    """
    Log a message using the default logger.

    Args:
        message: Log message (supports rich markup)
        level: Logging level
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging()
    _default_logger.log(level, message)
