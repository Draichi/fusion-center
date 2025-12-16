"""
Rich Logging Module for Project Overwatch.

Provides colorful, formatted logging with tables, panels, and markdown support.
"""

import logging
import os
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

# Custom theme for Project Overwatch
OVERWATCH_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "critical": "bold white on red",
        "success": "bold green",
        "tool": "bold magenta",
        "api": "bold blue",
        "data": "dim cyan",
        "highlight": "bold yellow",
        "muted": "dim white",
        "header": "bold cyan",
        "border": "bright_black",
        "agent": "bold green",
        "thinking": "italic dim cyan",
    }
)

# Initialize Rich console with custom theme
console = Console(theme=OVERWATCH_THEME, stderr=True)


class OverwatchLogger:
    """Custom logger with Rich formatting for Project Overwatch."""

    def __init__(self, name: str = "overwatch", level: str | None = None):
        """Initialize the logger with Rich handler."""
        self.console = console
        self.name = name
        self._bar = None

        # Set up Python logging with Rich handler
        log_level = level or os.getenv("LOG_LEVEL", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(message)s",
            datefmt="[%X]",
            handlers=[
                RichHandler(
                    console=self.console,
                    show_time=True,
                    show_path=False,
                    rich_tracebacks=True,
                    tracebacks_show_locals=True,
                    markup=True,
                )
            ],
        )
        self._logger = logging.getLogger(name)
        
    def set_progress_bar(self, bar: Any) -> None:
        """Set the active progress bar instance."""
        self._bar = bar

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message with cyan color."""
        self._logger.info(f"[info]{message}[/info]", **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with yellow color."""
        self._logger.warning(f"[warning]⚠️  {message}[/warning]", **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message with red color."""
        self._logger.error(f"[error]❌ {message}[/error]", **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message with white on red background."""
        self._logger.critical(f"[critical]🚨 {message}[/critical]", **kwargs)

    def success(self, message: str) -> None:
        """Log success message with green color."""
        self.console.print(f"[success]✅ {message}[/success]")

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._logger.debug(f"[muted]{message}[/muted]", **kwargs)

    def agent(self, message: str) -> None:
        """Log agent-related message."""
        if self._bar:
            self._bar.text(f"🤖 {message}")
        self.console.print(f"[agent]🤖 AGENT:[/agent] {message}")

    def thinking(self, message: str) -> None:
        """Log agent thinking/reasoning."""
        if self._bar:
            self._bar.text(f"💭 Thinking...")
        self.console.print(f"[thinking]💭 {message}[/thinking]")

    def tool_call(self, tool_name: str, params: dict[str, Any]) -> None:
        """Log a tool call with formatted parameters."""
        if self._bar:
            self._bar.text(f"🔧 Tool: {tool_name}")
        params_str = ", ".join(f"{k}={v!r}" for k, v in params.items() if v is not None)
        self.console.print(
            f"[tool]🔧 TOOL:[/tool] [bold]{tool_name}[/bold]([data]{params_str}[/data])"
        )

    def api_call(self, service: str, endpoint: str, method: str = "GET") -> None:
        """Log an outgoing API call."""
        self.console.print(
            f"[api]🌐 API {method}:[/api] [bold]{service}[/bold] → [data]{endpoint}[/data]"
        )

    def api_response(
        self, service: str, status_code: int, duration_ms: float | None = None
    ) -> None:
        """Log an API response with status code."""
        if status_code >= 200 and status_code < 300:
            status_style = "success"
            icon = "✅"
        elif status_code >= 400:
            status_style = "error"
            icon = "❌"
        else:
            status_style = "warning"
            icon = "⚠️"

        duration_str = f" ({duration_ms:.0f}ms)" if duration_ms else ""
        self.console.print(
            f"[api]📥 RESPONSE:[/api] [bold]{service}[/bold] "
            f"[{status_style}]{icon} {status_code}[/{status_style}]{duration_str}"
        )

    def panel(
        self,
        content: str,
        title: str = "",
        style: str = "border",
        subtitle: str | None = None,
    ) -> None:
        """Display content in a styled panel."""
        self.console.print(
            Panel(
                content,
                title=f"[header]{title}[/header]" if title else None,
                subtitle=f"[muted]{subtitle}[/muted]" if subtitle else None,
                border_style=style,
                padding=(1, 2),
            )
        )

    def markdown(self, md_content: str) -> None:
        """Render markdown content."""
        self.console.print(Markdown(md_content))

    def table(
        self,
        title: str,
        columns: list[str],
        rows: list[list[Any]],
        show_lines: bool = False,
    ) -> None:
        """Display data in a formatted table."""
        table = Table(
            title=f"[header]{title}[/header]",
            show_header=True,
            header_style="bold cyan",
            border_style="border",
            show_lines=show_lines,
        )

        for col in columns:
            table.add_column(col)

        for row in rows:
            table.add_row(*[str(cell) for cell in row])

        self.console.print(table)

    def result_summary(
        self,
        tool_name: str,
        status: str,
        count: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Display a formatted result summary."""
        if status == "success":
            status_icon = "✅"
            status_style = "success"
        elif status == "error":
            status_icon = "❌"
            status_style = "error"
        else:
            status_icon = "⚠️"
            status_style = "warning"

        # Build summary content
        lines = [
            f"[{status_style}]{status_icon} Status: {status.upper()}[/{status_style}]",
            f"[highlight]📊 Results: {count}[/highlight]",
        ]

        if details:
            lines.append("")
            lines.append("[muted]Details:[/muted]")
            for key, value in details.items():
                lines.append(f"  • {key}: {value}")

        self.panel(
            "\n".join(lines),
            title=f"🔧 {tool_name}",
            subtitle=datetime.now().strftime("%H:%M:%S"),
        )

    def divider(self, title: str = "") -> None:
        """Print a visual divider."""
        if title:
            self.console.rule(f"[header]{title}[/header]", style="border")
        else:
            self.console.rule(style="border")


# Global logger instance
_logger: OverwatchLogger | None = None


def get_logger() -> OverwatchLogger:
    """Get or create the global logger instance."""
    global _logger
    if _logger is None:
        _logger = OverwatchLogger()
    return _logger


# Convenience functions
def log(message: str, level: str = "info") -> None:
    """Log a message at the specified level."""
    logger = get_logger()
    getattr(logger, level, logger.info)(message)


def log_tool_call(tool_name: str, **params: Any) -> None:
    """Log a tool invocation."""
    get_logger().tool_call(tool_name, params)


def log_api_call(service: str, endpoint: str, method: str = "GET") -> None:
    """Log an API request."""
    get_logger().api_call(service, endpoint, method)


def log_api_response(service: str, status_code: int, duration_ms: float | None = None) -> None:
    """Log an API response."""
    get_logger().api_response(service, status_code, duration_ms)


def log_warning(message: str) -> None:
    """Log a warning."""
    get_logger().warning(message)


def log_error(message: str) -> None:
    """Log an error."""
    get_logger().error(message)


def log_result_table(
    title: str,
    columns: list[str],
    rows: list[list[Any]],
) -> None:
    """Display results in a table."""
    get_logger().table(title, columns, rows)


def log_startup_banner(component: str = "server") -> None:
    """Display the startup banner."""
    logger = get_logger()

    if component == "server":
        banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ██████╗ ██╗   ██╗███████╗██████╗ ██╗    ██╗ █████╗████████╗ ║
║  ██╔═══██╗██║   ██║██╔════╝██╔══██╗██║    ██║██╔══██╚══██╔══╝ ║
║  ██║   ██║██║   ██║█████╗  ██████╔╝██║ █╗ ██║███████║  ██║    ║
║  ██║   ██║╚██╗ ██╔╝██╔══╝  ██╔══██╗██║███╗██║██╔══██║  ██║    ║
║  ╚██████╔╝ ╚████╔╝ ███████╗██║  ██║╚███╔███╔╝██║  ██║  ██║    ║
║   ╚═════╝   ╚═══╝  ╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝  ╚═╝    ║
║                                                               ║
║         🌐 OSINT & Geopolitical Intelligence Server 🛰️         ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
        """
    else:  # agent
        banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ██████╗ ██╗   ██╗███████╗██████╗ ██╗    ██╗ █████╗████████╗ ║
║  ██╔═══██╗██║   ██║██╔════╝██╔══██╗██║    ██║██╔══██╚══██╔══╝ ║
║  ██║   ██║██║   ██║█████╗  ██████╔╝██║ █╗ ██║███████║  ██║    ║
║  ██║   ██║╚██╗ ██╔╝██╔══╝  ██╔══██╗██║███╗██║██╔══██║  ██║    ║
║  ╚██████╔╝ ╚████╔╝ ███████╗██║  ██║╚███╔███╔╝██║  ██║  ██║    ║
║   ╚═════╝   ╚═══╝  ╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝  ╚═╝  ╚═╝    ║
║                                                               ║
║            🤖 Autonomous Intelligence Agent 🧠                 ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
        """

    logger.console.print(f"[bold cyan]{banner}[/bold cyan]")


def log_tools_table(tools: list[tuple[str, str, str]]) -> None:
    """Display registered tools in a table.

    Args:
        tools: List of (name, category, description) tuples
    """
    logger = get_logger()

    table = Table(
        title="[header]🔧 Registered Tools[/header]",
        show_header=True,
        header_style="bold cyan",
        border_style="border",
        padding=(0, 1),
    )

    table.add_column("Tool", style="bold magenta")
    table.add_column("Category", style="cyan")
    table.add_column("Description", style="dim")

    for name, category, description in tools:
        table.add_row(name, category, description)

    logger.console.print(table)


def log_config_status(configs: dict[str, tuple[bool, str]]) -> None:
    """Display configuration status.

    Args:
        configs: Dict of config_name -> (is_set, description)
    """
    logger = get_logger()

    table = Table(
        title="[header]⚙️ Configuration Status[/header]",
        show_header=True,
        header_style="bold cyan",
        border_style="border",
    )

    table.add_column("Config", style="bold")
    table.add_column("Status")
    table.add_column("Description", style="dim")

    for name, (is_set, description) in configs.items():
        status = "[success]✅ Set[/success]" if is_set else "[warning]⚠️ Not Set[/warning]"
        table.add_row(name, status, description)

    logger.console.print(table)


def log_agent_step(step: int, action: str, details: str = "") -> None:
    """Log an agent execution step."""
    logger = get_logger()
    logger.console.print(
        f"[agent]🤖 Step {step}:[/agent] [bold]{action}[/bold]"
        + (f" - [data]{details}[/data]" if details else "")
    )


def log_agent_result(success: bool, summary: str) -> None:
    """Log agent execution result."""
    logger = get_logger()
    if success:
        logger.panel(
            f"[success]✅ {summary}[/success]",
            title="🤖 Agent Completed",
            style="green",
        )
    else:
        logger.panel(
            f"[error]❌ {summary}[/error]",
            title="🤖 Agent Failed",
            style="red",
        )

