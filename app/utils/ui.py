from rich.panel import Panel
from rich.table import Table

from app.utils.console import console


def title(text: str):

    console.rule(f"[header]{text}")


def success(text: str):

    console.print(f"✓ {text}", style="success")


def warning(text: str):

    console.print(f"⚠ {text}", style="warning")


def error(text: str):

    console.print(f"✗ {text}", style="error")


def info(text: str):

    console.print(text, style="info")


def panel(title: str, body: str):

    console.print(
        Panel(
            body,
            title=title,
        )
    )


def table(title: str):

    t = Table(title=title)

    t.add_column("Metric")

    t.add_column("Value")

    return t