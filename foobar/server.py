from .bar               import bar
from .commands.intel    import build
from .foo               import foo
from .rules.research    import decide
from prometheus_client  import start_http_server
import flask

application = flask.Flask(__name__)

@application.route('/foo')
def get_foo():
    return foo()

@application.route('/bar')
def get_bar():
    return bar()

@application.route('/command')
def get_command():
    try:
        return build()
    except:
        return 'sleep'

@application.route('/rule')
def get_rule():
    return decide()

def run_server():
    start_http_server(5000)
    application.run(host='0.0.0.0', port=8000)
