from CLI_lib import CLI_Train_model
import typer

app = typer.Typer()

class Commands():
    def __init__(self, params: CLI_Train_model):
      self.__params = params
      self.dbPath = None
      
    def load_commands(self):
        @app.command()
        def train_model(text: str = typer.Argument(
            "default_text", help="path, name and format of database."
        )):
          self.__params.database(text)

        @app.command()
        def convert_i3_to_parquet(text: str = typer.Argument(
            "default_text", help="path, name and format of database."
        )):
          self.__params.database(text)

        return app