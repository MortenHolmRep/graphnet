import typer


class CLI_Train_model:
    def database(
        self,
        text: str = typer.Argument(
            "default_text", help="path, name and format of database."
        ),
    ):
        """
        Give the database path variable
        """
        typer.echo(text)
    
    #def pulsemap(
    #    self,
    #    text: str = typer.Argument(
    #        "pulsemap", help=""
    #    ),
    #):
    #    typer.echo(text)
    
    #def batch(
    #    self,
    #    text: str = typer.Argument(
    #        "batch_size", help="batch size of training"
    #    ),
    #):
    #    typer.echo(text)
    
    #def workers(
    #    self,
    #    text: str = typer.Argument(
    #        "num_workers", help="number of workers"
    #    ),
    #):
    #    typer.echo(text)
    
    def gpu(
        self,
        text: str = typer.Argument(
            "gpus", help="Choose gpu to use [1/2]; default is CPU=None.", envvar=["", "1", "2"]
        ),
    ):
        typer.echo(text)
    
    #def target(
    #    self,
    #    text: str = typer.Argument(
    #        "target", help="reconstruction target; energy, ..."
    #    ),
    #):
    #    typer.echo(text)
    
    #def epochs(
    #    self,
    #    text: str = typer.Argument(
    #        "n_epochs", help="number of epochs to use."
    #    ),
    #):
    #    typer.echo(text)
    
    #def patience(
    #    self,
    #    text: str = typer.Argument(
    #        "patience", help="number of epochs after early stopping trigger"
    #    ),
    #):
    #    typer.echo(text)
    
    #def output(
    #    self,
    #    text: str = typer.Argument(
    #        "output", help="output path."
    #    ),
    #):
    #    typer.echo(text)
