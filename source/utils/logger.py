import logging

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
console_handler.setFormatter(formatter)

logger = logging.getLogger("Distar")
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)
