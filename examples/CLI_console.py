from CLI_commands import Commands
from CLI_lib import CLI_Train_model
import os
import typer

app = typer.Typer()
mydb = CLI_Train_model()

commands = {
    "examples": Commands(mydb)
}

for name, command in commands.items():
    app.add_typer(command.load_commands(), name=name)

#print dbPath


if __name__ == '__main__':
    app()