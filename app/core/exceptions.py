class DatasetNotFoundError(FileNotFoundError):
    """Raised when a required dataset file is not found."""
    pass

class DatasetNotLoadedError(Exception):
    """Raised when attempting to access an unloaded dataset."""