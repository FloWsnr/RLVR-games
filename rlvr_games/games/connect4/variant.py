"""Standard Connect 4 variant constraints."""

STANDARD_CONNECT4_ROWS = 6
STANDARD_CONNECT4_COLUMNS = 7
STANDARD_CONNECT4_CONNECT_LENGTH = 4


def validate_standard_connect4_dimensions(*, rows: int, columns: int) -> None:
    """Validate that the board shape matches standard Connect 4.

    Parameters
    ----------
    rows : int
        Number of board rows.
    columns : int
        Number of board columns.

    Raises
    ------
    ValueError
        If the board shape is not the standard 6x7 grid.
    """
    if rows != STANDARD_CONNECT4_ROWS or columns != STANDARD_CONNECT4_COLUMNS:
        raise ValueError(
            "Connect 4 only supports the standard 6x7 board with connect_length=4."
        )


def validate_standard_connect4_configuration(
    *,
    rows: int,
    columns: int,
    connect_length: int,
) -> None:
    """Validate that one configuration matches standard Connect 4.

    Parameters
    ----------
    rows : int
        Number of board rows.
    columns : int
        Number of board columns.
    connect_length : int
        Number of contiguous pieces required for a win.

    Raises
    ------
    ValueError
        If the supplied configuration is not the standard 6x7 connect-4
        variant.
    """
    validate_standard_connect4_dimensions(rows=rows, columns=columns)
    if connect_length != STANDARD_CONNECT4_CONNECT_LENGTH:
        raise ValueError(
            "Connect 4 only supports the standard 6x7 board with connect_length=4."
        )


__all__ = [
    "STANDARD_CONNECT4_COLUMNS",
    "STANDARD_CONNECT4_CONNECT_LENGTH",
    "STANDARD_CONNECT4_ROWS",
    "validate_standard_connect4_configuration",
    "validate_standard_connect4_dimensions",
]
