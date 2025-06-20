import sys


def error_message_detail(error, error_detail: sys):
    """
    Constructs a detailed error message including file name, line number, and error description.

    :param error: Exception object
    :param error_detail: sys module (to extract traceback info)
    :return: Formatted string with error details
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in python script: [{file_name}] at line number [{exc_tb.tb_lineno}] with message: [{str(error)}]"
    return error_message


class CustomException(Exception):
    """
    Custom Exception class for unified error handling across the pipeline.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
