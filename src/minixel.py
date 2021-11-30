from typing import NoReturn
import argparse
# import curses
import os
import re

from cellformula import CellFormula
from table import Sheet



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
            sheet[Sheet.index2coord(row_index, col_index)] = cell
    return sheet


def main() -> NoReturn:
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
        help="Input file to read data. If dosent exist," \
        + " the program will create am empty file",
    )
    parser.add_argument(
        '-o',
        '--output',
        help="Input file to send the data. If dosent exist," \
        + " the program will create am empty file",
        type=argparse.FileType('w')
    )
    args = parser.parse_args()
    path = args.input
    if not os.path.exists(path):
        sheet = Sheet()
    else:
        sheet = read_csv(path)

    for el in sheet.values():
        if not isinstance(el, CellFormula):
            continue
        el.compute(sheet)
    if args.verbose:
        sheet.display()
    if args.output:
        try:
            out = [ [''] * sheet.size[1] for _ in range(sheet.size[0])]
            for coord, element in sheet.items():
                index = Sheet.coord2index(coord)
                out[index[0]][index[1]] = str(element)
            for index, lines in enumerate(map(','.join, out)):
                if index:
                    args.output.write('\n')
                args.output.write(lines)
        finally:
            args.output.close()



if __name__ == '__main__':
    main()
