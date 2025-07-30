from Deployment import sample

def test_hello(capfd):
    sample.hello()
    out, _ = capfd.readouterr()
    assert "Hello" in out
