import logging 
import sys

class AppLogger():
    @staticmethod
    def setup_logging(level=logging.INFO):
        log_format = '🔔🔔%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        
        logging.basicConfig(
            level=level,
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        
    @staticmethod
    def get_logger(name: str):
        return logging.getLogger(name)
    
    
AppLogger.setup_logging()

logger = AppLogger.get_logger(__name__)