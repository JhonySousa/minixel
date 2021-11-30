"""Module for implementation of the sheet (data)."""

from typing import Dict, NoReturn, Optional,Tuple
from string import ascii_uppercase
from collections import UserDict
import re

from cellformula import CellFormula, CellType


class Sheet(UserDict):
    """Sheet class, where the data will be stored.
    Acts like an dict (is an UserDict), but with
    special attributes and methods for display on the
    screen non-interctively.

    Attributes:
        size (get_only): Computes the number of cols and rows.
        col_sizes (get_only): How many spaces that column needs.
    """

    def __init__(self, **kwargs: CellType) -> NoReturn:
        super().__init__(**kwargs)
        self._size: Optional[Tuple[int, int]] = None
        self._col_sizes: Dict[str, int] = dict()

    @property
    def size(self) -> Tuple[int, int]:
        """Computes the number of cols and rows."""
        if self._size is None or self._size == (0, 0):
            max_row = max_col = 0
            for elements in self.keys():
                row, col = Sheet.coord2index(elements)
                max_row = max((max_row, row + 1))
                max_col = max((max_col, col + 1))
            self._size = (max_row, max_col)
        return self._size

    @property
    def col_sizes(self) -> Dict[str, int]:
        """How many spaces that column needs."""
        if not self._col_sizes:
            for key, value in self.items():
                _, col = Sheet.coord2index(key)
                if isinstance(value, CellFormula):
                    self._col_sizes[col] = max(
                        len(str(value)),
                        self._col_sizes.get(col, 0)
                    )
                elif isinstance(value, int):
                    self._col_sizes[col] = max(
                        len(str(value)),
                        self._col_sizes.get(col, 0)
                    )
                else:
                    self._col_sizes[col] = max(
                        len(value),
                        self._col_sizes.get(col, 0)
                    )
        return self._col_sizes

    def display(self) -> NoReturn:
        """Display the sheet on the screen non-interactvely"""
        if self.size == (0, 0):
            print(
                '''+-------+
                | Empty |
                +-------+
                '''
            )

        index_row_space = len(str(self.size[0])) + 2
        for row in range(self.size[0] + 1):
            print(
                '|\033[90m{0:^{1}}\033[m|'.format(
                    row if row else ' __',
                    index_row_space
                ),
                end=''
            )
            for col in range(self.size[1]):
                if row == 0:
                    print('\033[90m', end='')

                if row:
                    element = str(self.get(Sheet.index2coord(row - 1, col), 0))
                else:
                    element = ascii_uppercase[col]

                print(f'{element:^{self.col_sizes[col] + 2}}\033[m', end='|')
            print('')

    @staticmethod
    def index2coord(row: int, col: int) -> str:
        """Convert matrix index to excel coord.
        E.g:
            >> index2coord(0, 0)
            << 'A1'
            >> index2coord(3, 4)
            << 'D5'

        Args:
            col (int): column
            row (int): row

        Returns:
            str: The excel coord.
        """
        return ascii_uppercase[col] + str(row + 1)

    @ staticmethod
    def coord2index(coord:str) -> Tuple[int, int]:
        """Convert the excel coord to matrix indexes
        E.g:
            >> coord2index('A1')
            << (0, 0)
            >> coord2index('D5')
            << (3, 4)

        Args:
            coord (str): The excel index

        Returns:
            Tuple[int, int]: The indexes.
        """
        row = int(re.match(r'[A-Z](\d+).*', coord).group(1)) - 1
        col = ascii_uppercase.find(coord[0])
        return (row, col)
