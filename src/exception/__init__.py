import sys
import logging


import sys


def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        return str(error)

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    return (
        f"Error occurred in python script: [{file_name}] "
        f"at line number [{line_number}]: {str(error)}"
    )

class MyException(Exception):
    """
    Custom exception class for handling errors.
    """
    def __init__(self, error_message: str, error_detail: sys):
        """
        Initializes the Exception with a detailed error message.

        :param error_message: A string describing the error.
        :param error_detail: The sys module to access traceback details.
        """
        # Call the base class constructor with the error message
        super().__init__(error_message)

        # Format the detailed error message using the error_message_detail function
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        """
        Returns the string representation of the error message.
        """
        return self.error_message