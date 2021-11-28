from typing import Any, List, Dict, NamedTuple, NoReturn, Union, Tuple, Callable
from operator import add, sub, mul, truediv as div
from string import ascii_uppercase
# from collections import deque
# import curses
import sys
import os
import re


Number = Union[int, float]
Equation = NamedTuple('Equation', [
    ('term1', str), ('term2', str),
    ('op', Callable[[Number, Number], Number])
])


# TODO: accept interval computation and match: =SUM(A1:5)/B1
class CellFormula:
    pattern = r"^=<OP>\(?<CE>\)?\(?(?:[\+\-\*\/\^]<OP>\(?<CE>\)*)+$"\
        .replace('<OP>', r'(?:(?:DIV|MUL|SUB|SUM)(?=\([A-Z]\d+(?::\d+){1,2}\)))?')\
            .replace('<CE>', r'(?:[A-Z]\d+(?::\d+){0,2}|\d+)')
    pattern = re.compile(pattern)

    op_dict = {
        'DIV': div,
        'MUL': mul,
        'SUB': sub,
        'SUM': add,
        '^': pow,
        '/': div,
        '*': mul,
        '-': sub,
        '+': add,
    }

    def __init__(self, cell: str) -> None:
        cell = cell.replace(' ', '')
        n_open_brackets = cell.count('(')
        if not self.pattern.match(cell) or n_open_brackets != cell.count(')'):
            return
        self.setence = cell
        self.formula: List[Equation] = list()

        cell = CellFormula.tokerise_setence(cell)

        # TODO: More performance jumping to each '()' pair
        # using RegEx or simple scan instead of stepping through
        # the sentence, character by character?
        for start in range(len(cell) - 1, -1, -1):
            if cell[start] != '(' and start > 0:
                continue
            for end in range(start, len(cell)):
                if cell[end] != ')' and end < len(cell) - 1:
                    continue
                if start > 0:
                    # remove the open brackets
                    sub_equation = CellFormula.parse_tokens(cell[start+1:end])
                else:
                    # if is zero, it is the begginnig..
                    sub_equation = CellFormula.parse_tokens(cell[:end + 1])
                for _ in range(end - start + 1):
                    cell.pop(start)
                cell.insert(start, sub_equation)
                break
        self.formula = cell[0]

    @classmethod
    def tokerise_setence(cls, setence: str) -> List[str]:
        """Given an setence, break it down in to a list.

        Args:
            setence (str)

        Returns:
            List[str]: tokens
        """
        pattern = r'<CE>|([\+\-\*\/\^])|(\(?\)?)|(SUM|SUB|MUL|DIV)\(<CE>\)' \
            .replace('<CE>', r'([A-Z]\d(?::\d){0,2})')
        pattern = re.compile(pattern)
        setence = [el for el in pattern.split(setence) if el and el != '=']
        return setence

    @staticmethod
    def get_interval(term: str):
        """Get the size of the interval.
        For example, the A1:5 will go from A1 to A5.
        And A1:6:2 will go fom A1 to A6, 2 by 2.

        Args:
            term (str): The term

        Yield:
            int: The term.
        """
        start = int(re.match(r'^[A-Z](\d+):\d+.*', term).group(1))
        end = int(re.match(r'^[A-Z]\d+:(\d+).*', term).group(1))
        inc = re.match(r'^[A-Z]\d+:\d+:(\d+)', term)
        inc = 1 if not inc else inc.group(1)
        col = term[0]
        for row_index in range(start, end + 1, inc):
            yield f'{col}{row_index}'

    @classmethod
    def map_operator(cls, term: str, operator: Callable[[str, Any], int]) -> Equation:
        """Given an term that describes an interval, generate that interval (get_interval)
        and map the operator, joining them together and generating the hole equation.

        Args:
            term (str): The term
            operator (Callable[[str, Any], int]): The operation

        Returns:
            Equation: The chain of equations joined.
        """
        term_gen = cls.get_interval(term)
        equation = Equation(next(term_gen), next(term_gen), operator)
        for t in term_gen:
            equation = Equation(t, equation, cls.op_dict[operator])
        return equation

    @classmethod
    def parse_tokens(cls, tokens: List[str]) -> Equation:
        """Given the tokens, generate the hole equation, and also
        do it recusrivly over intervals using the map_operator.

        Args:
            tokens (List[str]): The tokens.

        Returns:
            Equation: The hole equation
        """
        pattern = re.compile(r'[A-Z]\d+')
        for op in cls.op_dict:
            for _ in range(tokens.count(op)):
                index_op = tokens.index(op)

                if len(op) > 1:
                    tokens.insert(
                        index_op + 1,
                        CellFormula.map_operator(tokens.pop(index_op + 1), op)
                    )
                    tokens.pop(index_op)
                    continue

                term1 = tokens[index_op - 1]
                term2 = tokens[index_op + 1]

                if isinstance(term1, str) and not pattern.match(term1):
                    raise ValueError()
                if isinstance(term2, str) and not pattern.match(term2):
                    raise ValueError()

                tokens.pop(index_op - 1) # Delete the term 1
                tokens.pop(index_op - 1) # Delete the operator
                tokens.pop(index_op - 1) # Delete the term 2
                tokens.insert(index_op - 1, Equation(term1, term2, cls.op_dict[op]))
        return tokens[0]

    def compute(self, sheet: Dict[str, Any]) -> int:
        ...


CellType = Union[str, int, CellFormula]
SheetType = Dict[str, CellType]


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
    row = int(coord[-1:]) - 1
    col = ascii_uppercase.find(coord[0])
    return (row, col)


def read_csv(path: str) -> SheetType:
    """Read the CSV and generates the Table (sheet).

    Args:
        path (str)

    Returns:
        Sheet
    """
    with open(path, 'r') as csv_file:
        content = csv_file.readlines()
    sheet = dict()
    for row_index, row in enumerate(content):
        for col_index, cell in enumerate(row.split(',')):
            cell = cell.strip()
            if cell.isnumeric():
                cell = int(cell)
            elif cell.startswith('='):
                cell = CellFormula(cell)
            sheet[index2coord(row_index, col_index)] = cell
    return sheet


def display_headings(col_sizes: List[int]) -> NoReturn:
    """Display on the stdscr the headings using the
    maximum space available.

    Args:
        col_sizes (List[int]): sizes of each column
    """
    ...


def display_table(sheet: SheetType) -> NoReturn:
    """Display the table and the data in each cell.

    Args:
        sheet (Dict[str, CellType]): The datasheet.

    Returns:
        NoReturn: Dosent return anything
    """
    ...


def main():
    """Main function"""
    path = ''
    if len(sys.argv) > 1:
        path = sys.argv[1]
    if not os.path.exists(path):
        sys.exit(1)
    sheet = read_csv(path)
    print(sheet)


if __name__ == '__main__':
    main()
