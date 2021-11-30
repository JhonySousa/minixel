from typing import Iterator, List, NoReturn, Optional, Union, NamedTuple, Callable, Any, Dict
from operator import add, mul, sub, truediv as div
import re


Number = Union[int, float]
Equation = NamedTuple('Equation', [
    ('term1', str), ('term2', str),
    ('op', Callable[[Number, Number], Number])
])
CellCallStack = NamedTuple(
    'CellCallStack', [
        ('cell', 'CellFormula'),
        ('label', Optional[str])
    ]
)


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

    def __init__(self, cell: str) -> NoReturn:
        cell = cell.replace(' ', '')
        self.value = None
        self.formula: List[Equation] = list()
        self.setence = cell
        
        CellFormula._validate_cell(cell)
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
    def _validate_cell(cls, cell:str) -> List[str]:
        open_brackets = cell.count('(')
        clos_brackets = cell.count(')')
        pattern_match = re.match(r'(SUM|SUB|DIV|MUL)(?!\([A-Z]\d+(?::\d+){1,2}\))', cell)
        if pattern_match:
            raise FormulaFormatError(
                cell,
                f"has an interval operator unformated {pattern_match.group(0)}" \
                + f" at {pattern_match.start(0)} position."
            )
        if open_brackets > clos_brackets:
            raise FormulaFormatError(cell, "Unmatched '('.")
        if open_brackets < clos_brackets:
            raise FormulaFormatError(cell, "Unmatched ')'.")
        if not cls.pattern.match(cell):
            raise FormulaFormatError(cell)
        pattern_match = re.search(r'(?<!(?:SUM|SUB|DIV|MUL)\()([A-Z]\d+(?::\d+)+)', cell)
        if pattern_match:
            raise FormulaFormatError(
                cell,
                f'has an interval outside of any interval operator {pattern_match.group(1)}' \
                + f' at {pattern_match.start(1)}.'
            )
        pattern_match = re.search(r'[A-Z][\+\-\*\/\^]', cell)
        if pattern_match:
            raise FormulaFormatError(cell, f'has an col as term of operation at {pattern_match.start(0)}.')

    @classmethod
    def tokerise_setence(cls, setence: str) -> List[str]:
        """Given an setence, break it down in to a list.

        Args:
            setence (str)

        Returns:
            List[str]: tokens
        """
        try:
            pattern = re.compile(
                r'(SUM|SUB|DIV|MUL)\(([A-Z]\d+(?::\d+){1,2})\)|' \
                + r'([A-Z]\d+(?::\d+){0,2})|(\d+)|([\+\-\*\/\^])|(\(?\)?)'
            )
            setence = [el for el in pattern.split(setence) if el not in (None, '', '=', ' ')]
            return setence
        except Exception as err:
            raise TokenrizeFormulaError(setence, 'Troubles with tokerising ->' + err) from None

    @staticmethod
    def get_interval(term: str) -> Iterator[str]:
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

    def compute(
            self, sheet: Dict[str, Any],
            call_stack:Optional[List[CellCallStack]]=None
    ) -> int:
        call_stack = call_stack if call_stack else [CellCallStack(self, None)]
        if [call[0] for call in call_stack].count(self) > 1:
            raise CyclicError([call[1] for call in call_stack])

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

                if isinstance(terms[indx], str):
                    terms[indx] = sheet.get(terms[indx])
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
                    output = terms[indx].get_value(sheet, call_stack + [])
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


class CellFormulaError(Exception):
    """Base class for CellFormula errors."""
    def __init__(self, formula: str, message: str) -> None:
        self.formula = formula
        self.message = message
        super().__init__()

    def __str__(self) -> str:
        return f'This formula ({self.formula}) {self.message}.'

class FormulaFormatError(CellFormulaError):
    """Error that occours when the given formula dosent match.
    """
    def __init__(self, formula:str, message="dosent match with pattern") -> NoReturn:
        super().__init__(formula, message)


class TokenrizeFormulaError(CellFormulaError):
    """This will occour when something goes wrong on the process
    Breaking down the formula (Tokenrizing the formula)."""
    ...


class CyclicError(CellFormulaError):
    def __init__(self, call_stack: List[str]) -> None:
        self.call_stack = call_stack
        self.call_stack.append(self.call_stack[0])
        super().__init__("", "")

    def __str__(self) -> str:
        return 'Cyclic Recursion: ' + ' -> '.join(self.call_stack)
