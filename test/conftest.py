import pytest
import os

@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """
    Automatically change the current working directory to the directory 
    containing the test file before each test runs.
    
    This allows tests to load local configuration files (like 'test_config.yaml')
    using relative paths, regardless of where pytest was started from.
    """
    # request.path is the modern way to get the test path (pathlib.Path)
    # request.fspath is the older way (py.path.local)
    test_path = getattr(request, 'path', None) or getattr(request, 'fspath', None)
    if test_path:
        # Get the directory of the test file
        test_dir = test_path.parent if hasattr(test_path, 'parent') else os.path.dirname(str(test_path))
        monkeypatch.chdir(test_dir)
