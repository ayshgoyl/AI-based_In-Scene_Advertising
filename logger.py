import logging
import datetime
import os

ALLOWED_CATEGORIES = {
    "Performance",
    "Security",
    "Error",
    "EventFlow",
    "Configuration",
    "Variable"
}

class EnterpriseLogger:
    def __init__(self, log_file_path, job_id):
        self.log_file_path = log_file_path
        self.job_id = job_id
        
        self.logger = logging.getLogger(job_id)
        self.logger.setLevel(logging.DEBUG)
        
        # Don't duplicate handlers if the logger already exists
        if not self.logger.handlers:
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            fh = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            self.logger.addHandler(fh)
            
            # Use console for debug purposes as well
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            self.logger.addHandler(ch)

    def _write_log(self, level_fn, category, message, **kwargs):
        if category not in ALLOWED_CATEGORIES:
            category = "EventFlow" # Default fallback
            
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # Append kwargs to message if present for detailed metric logging
        if kwargs:
            msg_details = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            message = f"{message} | {msg_details}"
            
        log_line = f"[{timestamp}] [{self.job_id}] [{level_fn.__name__.upper()}] [{category}] {message}"
        level_fn(log_line)

    def info(self, category, message, **kwargs):
        self._write_log(self.logger.info, category, message, **kwargs)
        
    def debug(self, category, message, **kwargs):
        self._write_log(self.logger.debug, category, message, **kwargs)
        
    def warning(self, category, message, **kwargs):
        self._write_log(self.logger.warning, category, message, **kwargs)
        
    def error(self, category, message, **kwargs):
        self._write_log(self.logger.error, category, message, **kwargs)
        
    def fatal(self, category, message, **kwargs):
        self._write_log(self.logger.critical, category, message, **kwargs)
