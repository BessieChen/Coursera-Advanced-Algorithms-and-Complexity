# python3
from sys import stdin
import copy
import numpy as np

VeryBigNumber = 1e9


class Equation:
    def __init__( self, a, b ):
        self.a = a
        self.b = b

class Position:
    def __init__( self, column, row ):
        self.column = column
        self.row = row


def Prime_element( a, used_rows, used_columns ):
    mm = len(a)
    pivot_element = Position(0, 0)
    while used_rows[pivot_element.row]:
        pivot_element.row += 1
    while used_columns[pivot_element.column]:
        pivot_element.column += 1
    while 0 == a[pivot_element.row][pivot_element.column] or used_rows[pivot_element.row]:
        pivot_element.row += 1
        if pivot_element.row > mm - 1:
            return False, None
    return True, pivot_element


def SwapLines( a, b, used_rows, pivot_element ):
    a[pivot_element.column], a[pivot_element.row] = a[pivot_element.row], a[pivot_element.column]
    b[pivot_element.column], b[pivot_element.row] = b[pivot_element.row], b[pivot_element.column]
    used_rows[pivot_element.column], used_rows[pivot_element.row] = used_rows[pivot_element.row], used_rows[
        pivot_element.column]
    pivot_element.row = pivot_element.column


def ProcessPivotElement( a, b, pivot_element ):
    n = len(a)
    mm = len(a[pivot_element.row])
    scale = a[pivot_element.row][pivot_element.column]
    for j in range(mm):
        a[pivot_element.row][j] /= scale
    b[pivot_element.row] /= scale
    for i in range(n):
        if i != pivot_element.row:
            scale = a[i][pivot_element.column]
            for j in range(pivot_element.column, n):
                a[i][j] -= a[pivot_element.row][j] * scale
            b[i] -= b[pivot_element.row] * scale


def MarkPivotElementUsed( pivot_element, used_rows, used_columns ):
    used_rows[pivot_element.row] = True
    used_columns[pivot_element.column] = True


def SolveEquation( equation ):
    a = equation.a
    b = equation.b
    size = len(a)

    used_columns = [False] * size
    used_rows = [False] * size
    for step in range(size):
        solved, pivot_element = Prime_element(a, used_rows, used_columns)
        if not solved:
            return False, None
        SwapLines(a, b, used_rows, pivot_element)
        ProcessPivotElement(a, b, pivot_element)
        MarkPivotElementUsed(pivot_element, used_rows, used_columns)

    return True, b


def addEquations( n, mm, A, b, Big_number ):
    for i in range(mm):
        e = [0.0] * mm
        e[i] = -1.0
        A.append(e)
        b.append(0.0)
    A.append([1.0] * mm)
    b.append(Big_number)


def checkResult( n, mm, A, b, c, result, lastEquation, ans, bestScore ):
    for r in result:
        if r < -1e-3:
            return False, ans, bestScore
    for i in range(n):
        r = 0.0
        for j in range(mm):
            r += A[i][j] * result[j]
        if r > b[i] + 1e-3:
            return False, ans, bestScore
    score = 0.0
    for j in range(mm):
        score += c[j] * result[j]
    if score <= bestScore:
        return False, ans, bestScore
    else:
        if lastEquation:
            return True, 1, score
        else:
            return True, 0, score


def solve_diet_problem( n, mm, A, b, c, Big_number=VeryBigNumber ):
    addEquations(n, mm, A, b, Big_number)
    # print(A, b)
    l = n + mm + 1
    ans = -1
    bestScore = -float('inf')
    bestResult = None
    for x in range(2 ** l):
        usedIndex = [i for i in range(l) if ((x / 2 ** i) % 2) // 1 == 1]
        if len(usedIndex) != mm:
            continue
        lastEquation = False
        if usedIndex[-1] == l - 1:
            lastEquation = True
        As = [A[i] for i in usedIndex]
        bs = [b[i] for i in usedIndex]
        # print(As, bs)
        solved, result = SolveEquation(copy.deepcopy(Equation(As, bs)))
        # print(As, bs, result)
        if solved:
            isAccepted, ans, bestScore = checkResult(n, mm, A, b, c, result, lastEquation, ans, bestScore)
            if isAccepted:
                bestResult = result
    # print(A)
    return [ans, bestResult]


def solve_diet_problem0( n, mm, A, b, c ):
    # addEquations(n, mm, A, b, VeryBigNumber)
    res = linprog(-np.array(c), A, b)
    if 3 == res.status:
        ans = 1
        x = None
    elif 0 == res.status:
        ans = 0
        x = list(res.x)
    else:
        ans = -1
        x = None
    return ans, x


n, mm = list(map(int, stdin.readline().split()))
A = []
for i in range(n):
    A += [list(map(int, stdin.readline().split()))]
b = list(map(int, stdin.readline().split()))
c = list(map(int, stdin.readline().split()))

anst, ansx = solve_diet_problem(n, mm, A, b, c)

if anst == -1:
    print("No solution")
if anst == 0:
    print("Bounded solution")
    print(' '.join(list(map(lambda x: '%.18f' % x, ansx))))
if anst == 1:
    print("Infinity")