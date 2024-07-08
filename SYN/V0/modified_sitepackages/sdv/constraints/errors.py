"""Constraint Exceptions."""

import logging

from sdv.errors import log_exc_stacktrace

LOGGER = logging.getLogger(__name__)


class MissingConstraintColumnError(Exception):
    """Error used when constraint is provided a table with missing columns."""

    def __init__(self, missing_columns):
        self.missing_columns = missing_columns


class AggregateConstraintsError(Exception):
    """Error used to represent a list of constraint errors."""

    def __init__(self, errors):
        self.errors = errors
        for error in self.errors:
            log_exc_stacktrace(LOGGER, error)

    def __str__(self):
        return '\n' + '\n\n'.join(map(str, self.errors))


class InvalidFunctionError(Exception):
    """Error used when an invalid function is utilized."""


class FunctionError(Exception):
    """Error used when an a function produces an unexpected error."""


class ConstraintMetadataError(Exception):
    """Error to raise when Metadata is not valid."""
