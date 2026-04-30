"""Utilities for capturing terminal output to log files."""


class Tee:
    """A file-like object that writes to both a file and the original stream."""
    
    def __init__(self, file_path: str, original_stream):
        """Initialize Tee with a file path and original stream.
        
        Args:
            file_path: Path to the log file
            original_stream: The original stdout or stderr stream
        """
        self.file = open(file_path, 'a', encoding='utf-8', buffering=1)  # Line buffered
        self.original_stream = original_stream
        self.file_path = file_path
    
    def write(self, text: str) -> int:
        """Write text to both file and original stream."""
        try:
            self.original_stream.write(text)
            self.original_stream.flush()
        except Exception:
            pass  
        
        try:
            self.file.write(text)
            self.file.flush()
        except Exception:
            pass  
        
        return len(text)
    
    def flush(self):
        """Flush both streams."""
        try:
            self.original_stream.flush()
        except Exception:
            pass
        try:
            self.file.flush()
        except Exception:
            pass
    
    def close(self):
        """Close the file."""
        if self.file and not self.file.closed:
            self.file.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
    
    # Make it compatible with file-like operations
    def __getattr__(self, name):
        """Delegate other attributes to the original stream."""
        return getattr(self.original_stream, name)

