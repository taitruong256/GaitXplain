from datetime import datetime
from pathlib import Path
import logging
import sys
import yaml


def make_run_dir(output_root: str, mode: str) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = root / f'{mode}_{stamp}'
    suffix = 1
    while run_dir.exists():
        run_dir = root / f'{mode}_{stamp}_{suffix}'
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_config_snapshot(config: dict, run_dir: Path) -> None:
    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)


def append_log_line(run_dir: Path, line: str) -> None:
    with open(run_dir / 'log.txt', 'a') as f:
        f.write(line.rstrip() + '\n')


class _TeeStream:
    def __init__(self, terminal_stream, log_file):
        self.terminal_stream = terminal_stream
        self.log_file = log_file

    def write(self, message):
        self.terminal_stream.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal_stream.flush()
        self.log_file.flush()


def setup_run_logging(run_dir: Path) -> logging.Logger:
    log_file = open(run_dir / 'log.txt', 'a', buffering=1)
    sys.stdout = _TeeStream(sys.__stdout__, log_file)
    sys.stderr = _TeeStream(sys.__stderr__, log_file)

    logger = logging.getLogger('gaitxplain')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter('%(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    return logger