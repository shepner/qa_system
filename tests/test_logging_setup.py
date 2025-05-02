import os
import logging
import tempfile
import pytest
from pathlib import Path
from qa_system.logging_setup import setup_logging

@pytest.mark.usefixtures("temp_log_file")
def test_logging_creates_log_file(temp_log_file):
    # Remove file if it exists
    if os.path.exists(temp_log_file):
        os.remove(temp_log_file)
    setup_logging(LOG_FILE=temp_log_file, LEVEL="INFO")
    logger = logging.getLogger("qa_system.tests.test_logging_setup")
    logger.info("Test log file creation")
    logging.shutdown()
    assert os.path.exists(temp_log_file)
    with open(temp_log_file) as f:
        content = f.read()
        assert "Test log file creation" in content

@pytest.mark.usefixtures("temp_log_file")
def test_logging_level_respected(temp_log_file):
    setup_logging(LOG_FILE=temp_log_file, LEVEL="ERROR")
    logger = logging.getLogger("qa_system.tests.test_logging_setup")
    logger.info("This should not appear")
    logger.error("This should appear")
    logging.shutdown()
    with open(temp_log_file) as f:
        content = f.read()
        assert "This should appear" in content
        assert "This should not appear" not in content

@pytest.mark.usefixtures("temp_log_file")
def test_log_rotation(temp_log_file):
    # Set a very small maxBytes to force rotation
    from logging.handlers import RotatingFileHandler
    setup_logging(LOG_FILE=temp_log_file, LEVEL="INFO")
    logger = logging.getLogger("qa_system.tests.test_logging_setup")
    # Find the file handler and set maxBytes to a small value
    for handler in logging.root.handlers:
        if isinstance(handler, RotatingFileHandler):
            handler.maxBytes = 100
            handler.backupCount = 2
    # Write enough logs to trigger rotation
    for i in range(50):
        logger.info(f"Log line {i}")
    logging.shutdown()
    # Check that at least one rotated file exists
    rotated_files = list(Path(temp_log_file).parent.glob(Path(temp_log_file).name + '*'))
    assert any(str(f) != temp_log_file for f in rotated_files)

@pytest.mark.usefixtures("temp_log_file")
def test_console_and_file_output(monkeypatch, temp_log_file, capsys):
    setup_logging(LOG_FILE=temp_log_file, LEVEL="INFO")
    logger = logging.getLogger("qa_system.tests.test_logging_setup")
    logger.info("Console and file test")
    logging.shutdown()
    # Check file output
    with open(temp_log_file) as f:
        content = f.read()
        assert "Console and file test" in content
    # Check console output (captured by capsys)
    captured = capsys.readouterr()
    assert "Console and file test" in captured.out or "Console and file test" in captured.err

@pytest.mark.usefixtures("temp_log_file")
def test_debug_level_and_format(temp_log_file):
    setup_logging(LOG_FILE=temp_log_file, LEVEL="DEBUG")
    logger = logging.getLogger("qa_system.tests.test_logging_setup")
    logger.debug("Debug message")
    logging.shutdown()
    with open(temp_log_file) as f:
        content = f.read()
        assert "DEBUG" in content
        assert "Debug message" in content
        # Check format: should include asctime, name, levelname, message
        assert "- qa_system.tests.test_logging_setup - DEBUG - Debug message" in content 