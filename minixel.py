from typing import Any, List, Dict, NamedTuple, NoReturn, Optional, Union, Tuple, Callable
from operator import add, sub, mul, truediv as div
from string import ascii_uppercase
from collections import UserDict
from functools import reduce
import argparse
# import curses
import os
import re


Number = Union[int, float]
Equation = NamedTuple('Equation', [
    ('term1', str), ('term2', str),
    ('op', Callable[[Number, Number], Number])
])


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

        self.value = None
        self.setence = cell
        self.formula: List[Equation] = list()

        if not self.pattern.match(cell) or n_open_brackets != cell.count(')'):
            raise ValueError(
                'Something was wrong with this cell, check the number of' \
                + 'open/closed backets and if the pattern match.'
            )

        cell = CellFormula.tokerise_setence(cell)

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
        self.formula: Equation = cell[0]

    @classmethod
    def tokerise_setence(cls, setence: str) -> List[str]:
        """Given an setence, break it down in to a list.

        Args:
            setence (str)

        Returns:
            List[str]: tokens
        """
        pattern = re.compile(
            r'(SUM|SUB|DIV|MUL)\(([A-Z]\d+(?::\d+){1,2})\)|' \
            + r'([A-Z]\d+(?::\d+){0,2})|(\d+)|([\+\-\*\/\^])|(\(?\)?)'
        )
        setence = [el for el in pattern.split(setence) if el not in (None, '', '=', ' ')]
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
        inc = 1 if not inc else int(inc.group(1))

        if start > end:
            inc *= -1

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
        try:
            terms = dict()
            terms[0] = next(term_gen)
            terms[1] = next(term_gen)
            equation = Equation(terms[0], terms[1], operator)
        except StopIteration:
            return Equation(terms.get(0, 0), terms.get(1, 0), operator)

        for t in term_gen:
            equation = Equation(t, equation, operator)
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
        if '(' in tokens:
            tokens.remove('(')
        if ')' in tokens:
            tokens.remove(')')
        pattern = re.compile(r'(?:[A-Z]\d+)|\d+')
        for op in cls.op_dict:
            for _ in range(tokens.count(op)):
                index_op = tokens.index(op)

                if len(op) > 1:
                    tokens.insert(
                        index_op + 1,
                        CellFormula.map_operator(
                            tokens.pop(index_op + 1),
                            cls.op_dict[op]
                        )
                    )
                    tokens.pop(index_op)
                    continue

                term1 = tokens[index_op - 1]
                term2 = tokens[index_op + 1]
                term1 = int(term1) if isinstance(term1, str) and term1.isdecimal() else term1
                term2 = int(term2) if isinstance(term2, str) and term2.isdecimal() else term2

                if isinstance(term1, str) and not pattern.match(term1):
                    raise ValueError()
                if isinstance(term2, str) and not pattern.match(term2):
                    raise ValueError()

                tokens.pop(index_op - 1) # Delete the term 1
                tokens.pop(index_op - 1) # Delete the operator
                tokens.pop(index_op - 1) # Delete the term 2
                tokens.insert(index_op - 1, Equation(term1, term2, cls.op_dict[op]))
        return tokens[0]

    def get_value(self, sheet: Dict[str, Any], call_stack=None) -> int:
        """Get the value, and if the value is'nt computed yet, it will be
        computed right on the way.

        Args:
            sheet (Dict[str, Any]): The table sheet
            call_stack (List, optional): The list of call_stack. Defaults to None.

        Returns:
            int: The value (computed or not).
        """
        if not self.value:
            return self.compute(sheet, call_stack)
        return self.value

    def compute(self, sheet: Dict[str, Any], call_stack=None) -> int:
        call_stack = call_stack if call_stack else []
        if self in call_stack:
            return "Cyclic recursion ("

        current_equation = self.formula
        equation_stack = list()
        result = 0
        while True:
            terms: List[Union[str, Equation]] = [
                current_equation.term1,
                current_equation.term2
            ]
            operator = current_equation.op

            for indx in range(2):
                # In case if one of the terms was an Equation...
                if isinstance(terms[indx], Equation):
                    term1, term2 = terms
                    term1 = None if indx == 0 else term1
                    term2 = None if indx == 1 else term2
                    equation_stack.append(Equation(
                        term1, term2, op=current_equation.op
                    ))
                    current_equation = terms[indx]
                    terms.clear()
                    break

                terms[indx] = sheet.get(terms[indx], 0) if isinstance(terms[indx], str) else terms[indx]
                terms[indx] = 0 if terms[indx] == '' else terms[indx]

                # Still str value in the term...
                if isinstance(terms[indx], str):
                    raise ValueError(
                        f'Trying to evaluate formula "{self.setence}",' + \
                        'got an "str" instead of "int": ' + \
                        f'{current_equation.term1 if indx == 0 else current_equation.term2}=' + \
                        f'"{terms[indx]}"'
                    )

                if isinstance(terms[indx], self.__class__):
                    output = terms[indx].get_value(sheet, call_stack + [self])
                    if isinstance(terms[indx], str):
                        if len(call_stack) > 0:
                            return f'{output} {terms[indx]}'
                        else:
                            first_cell = terms[indx].split(' ')[3]
                            raise RecursionError(
                                f'{output} {terms[indx]} <- {first_cell} )'
                            )
                    terms[indx] = output

            if len(terms) > 0:
                result = operator(int(terms[0]), int(terms[1]))
                if len(equation_stack) > 0:
                    terms[0], terms[1], operator = equation_stack.pop(-1)
                    terms[0] = result if terms[0] is None else terms[0]
                    terms[1] = result if terms[1] is None else terms[1]
                    current_equation = Equation(*terms, operator)
                    continue
                else:
                    break
        self.value = result
        return result

    def __repr__(self) -> str:
        if self.value is None:
            return self.setence
        else:
            return '=' + str(self.value)


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
