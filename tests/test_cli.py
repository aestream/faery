from click.testing import CliRunner

from faery.cli.cli import cli


def test_cli_input():
    runner = CliRunner()
    result = runner.invoke(cli, "input tests/data/test.csv", catch_exceptions=True)
    assert result.exit_code == 0
    assert result.output == "1,10,10,0\n3,1,1,0\n10,20,20,1\n11,5,5,1\n"
