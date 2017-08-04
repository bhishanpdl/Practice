import flask, foobar, multiprocessing, pytest, requests


'''
server_process = None
def setup_module(module):
    global server_process
    server_process = multiprocessing.Process(target=foobar.run_server)
    server_process.start()

def teardown_module(module):
    server_process.terminate()

@pytest.fixture
def server(request):
    server_process = multiprocessing.Process(target=foobar.run_server)
    server_process.start()
    def teardown():
        server_process.terminate()
    request.addfinalizer(teardown)
'''

host = 'localhost'
port = 8000
url  = 'http://{}:{}/{{}}'.format(host, port)

def _test_foo():
    response = requests.get(url.format('foo'))
    assert response.status_code == 200
    assert response.content.decode() == 'foo'

def _test_bar():
    response = requests.get(url.format('bar'))
    assert response.status_code == 200
    assert response.content.decode() == 'bar'
