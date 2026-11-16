"""Tests for src/logger.py: one logging set-up for the console and an optional file."""

import logging

import pytest

from src import logger as log_setup


def test_unit_setup_logging_writes_to_the_file_and_creates_its_folder(tmp_path):
    log_file = tmp_path / "logs" / "pipeline.log"
    log_setup.setup_logging("info", log_file)
    logging.getLogger("src.test").info("hello %s", "log")
    for handler in logging.getLogger().handlers:
        handler.flush()
    line = log_file.read_text(encoding="utf-8")
    assert "hello log" in line
    assert "| INFO | src.test |" in line


def test_unit_setup_logging_without_a_file_uses_the_console_only():
    log_setup.setup_logging("WARNING")
    root = logging.getLogger()
    assert root.level == logging.WARNING
    assert [type(handler) for handler in root.handlers] == [logging.StreamHandler]


def test_unit_setup_logging_replaces_earlier_handlers(tmp_path):
    log_setup.setup_logging("INFO", tmp_path / "first.log")
    log_setup.setup_logging("INFO", tmp_path / "second.log")
    files = [handler for handler in logging.getLogger().handlers if isinstance(handler, logging.FileHandler)]
    assert len(files) == 1


def test_unit_setup_logging_raises_on_an_unknown_level():
    with pytest.raises(ValueError, match="logging.level is 'LOUD'"):
        log_setup.setup_logging("LOUD")
