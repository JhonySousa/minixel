from typing import Dict, NamedTuple, NoReturn, Optional, Union, Tuple, Callable
from string import ascii_uppercase
from collections import UserDict
from functools import reduce
import argparse
# import curses
import os
import re

from cellformula import CellFormula


Number = Union[int, float]
Equation = NamedTuple('Equation', [
    ('term1', str), ('term2', str),
    ('op', Callable[[Number, Number], Number])
])


CellType = Union[str, int, CellFormula]
SheetType = Dict[str, CellType]


class Sheet(UserDict):
    def __init__(self, __dict: Optional[Dict[str, CellType]]=None, **kwargs: CellType) -> None:
        super().__init__(__dict=__dict, **kwargs)
        self._size: Optional[Tuple[int, int]] = None
        self._col_sizes: Dict[str, int] = dict()

    @property
    def size(self) -> Tuple[int, int]:
        if self._size is None or self._size == (0, 0):
            max_row = max_col = 0
            for elements in self.keys():
                if elements == '__dict':
                    continue
                row, col = coord2index(elements)
                max_row = max((max_row, row + 1))
                max_col = max((max_col, col + 1))
            self._size = (max_row, max_col)
        return self._size

    @property
    def col_sizes(self) -> Dict[str, int]:
        if not self._col_sizes:
            for key, value in self.items():
                if key == '__dict':
                    continue
                _, col = coord2index(key)
                if isinstance(value, CellFormula):
                    self._col_sizes[col] = len(str(value))
                elif isinstance(value, int):
                    self._col_sizes[col] = len(str(value))
                else:
                    self._col_sizes[col] = len(value)
        return self._col_sizes

    def display(self):
        if self.size == (0, 0):
            return '''
            +-------+
            | Empty |
            +-------+
            '''
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
                element = str(self.get(index2coord(row - 1, col), 0)) if row else ascii_uppercase[col]
                print(f'{element:^{self.col_sizes[col] + 2}}\033[m', end='|')
            print('')


def index2coord(row:int, col:int) -> str:
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


def read_csv(path: str) -> Sheet:
    """Read the CSV and generates the Table (sheet).

    Args:
        path (str)

    Returns:
        Sheet
    """
    with open(path, 'r') as csv_file:
        content = csv_file.readlines()
    sheet = Sheet()
    num_pattern = re.compile(r'\-?\d+')
    for row_index, row in enumerate(content):
        for col_index, cell in enumerate(row.split(',')):
            cell = cell.strip()
            if num_pattern.match(cell):
                cell = int(cell)
            elif cell.startswith('='):
                cell = CellFormula(cell)
            sheet[index2coord(row_index, col_index)] = cell
    return sheet


def interactive_mode(sheet: SheetType) -> NoReturn:
    """Display the table and the data in each cell.

    Args:
        sheet (Dict[str, CellType]): The datasheet.

    Returns:
        NoReturn: Dosent return anything
    """
    ...


def main():
    """Main function"""
    parser = argparse.ArgumentParser()
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument(
        '-v', '--verbose',
        help='Show more information.',
        action='store_true'
    )
    display_group.add_argument(
        '-i', '--interactive',
        help='Enter interactive mode.',
        action='store_true'
    )
    parser.add_argument(
        'input',
        help="Input file to read data. If dosent exist, the program will create am empty file",
    )
    parser.add_argument(
        '-o',
        '--output',
        help="Input file to send the data. If dosent exist, the program will create am empty file",
        type=argparse.FileType('w')
    )
    args = parser.parse_args()
    path = args.input
    if not os.path.exists(path):
        sheet = Sheet()
    else:
        sheet = read_csv(path)
    del(sheet['__dict'])

    for el in sheet.values():
        if not isinstance(el, CellFormula):
            continue
        el.compute(sheet)
    if args.verbose:
        sheet.display()


if __name__ == '__main__':
    main()
