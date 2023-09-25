import logging

logger = logging.getLogger("tarkibi")
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z"))
logger.addHandler(ch)

fh = logging.FileHandler('tarkibi.log')
fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z"))
logger.addHandler(fh)
