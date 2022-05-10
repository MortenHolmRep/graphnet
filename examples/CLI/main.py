from typing import Optional
import typer

app = typer.Typer()

@app.command()
def db(name: Optional[str] = None):
    if name:
        typer.values(f"{path}")

    else:
        typer.values(f"path/to/db")

def main():
    config = {
        "db": f"{path}",
    }
    print(config)