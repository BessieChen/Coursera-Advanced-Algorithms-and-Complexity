# python3
from __future__ import print_function
from sys import stdin
import datetime

EPS = 1e-6
PRECISION = 20


def print_matrix( m, title='' ):
    return
    print(title)
    for row in m:
        print(' '.join([str(round(num, 1)).rjust(5) for num in row]))
    print()


def simplex_pivot( table, basic ):

    n = len(table[0]) - 1
    m = len(table)

    assert min(table[-1:][0][:-1]) < 0

    pivot_columns = table[m - 1].index(min(table[m - 1][:-1]))
    lowest = float('Inf')
    pivot_rows = None
    pivot_test = None
    for bb in range(m - 1):
        element = table[bb][pivot_columns]
        if are_equal(element, 0) or element <= 0 - EPS:
            continue
        b_val = table[bb][n]
        pivot_test = b_val / element
        if not are_equal(pivot_test, lowest) and pivot_test - EPS < lowest:
            lowest = pivot_test
            pivot_rows = bb

    if pivot_test == None:
        return 1

    if pivot_rows == None:
        return -1

    pivot_element = (pivot_columns, pivot_rows)
    reduce_column(table, pivot_element)
    basic[pivot_rows] = pivot_columns
    return 0


def reduce_column( table, pivot ):
    pivot_rows = pivot[1]
    pivot_columns = pivot[0]
    pivot_coeff = table[pivot_rows][pivot_columns]

    if pivot_coeff != 1:
        for column, coeff in enumerate(table[pivot_rows]):
            table[pivot_rows][column] = coeff / pivot_coeff
        pivot_coeff = 1

    for row in range(len(table)):
        if row == pivot_rows:
            continue

        factor = (table[row][pivot_columns] / pivot_coeff)
        for column, coeff in enumerate(table[row]):
            if column == pivot_columns:
                table[row][column] = 0
                continue
            table[row][column] = table[row][column] - factor * table[pivot_rows][column]


def are_equal( a, bb ):
    if abs(a - bb) <= EPS:
        return True
    else:
        return False


def initial_tableau( a, bb, c ):
    table = [row.copy() for row in a]
    n = len(table)

    for idx, row in enumerate(table):
        basic_row = [0] * n
        b_val = bb[idx]
        if b_val < 0:
            # change sign, which means surplus variable
            basic_row[idx] = -1
            new_row = [-e for e in row] + basic_row + [-b_val]
        else:
            basic_row[idx] = 1
            new_row = row + basic_row + [b_val]
        table[idx] = new_row

    table.append([-i for i in c] + [0] * (n + 1))

    return table


def perform_simplex( table, basic ):
    while min(table[-1:][0][:-1]) < 0 - EPS:
        step = simplex_pivot(table, basic)
        print_matrix(table, 'intermediate simplex')
        if step in [-1, 1]:
            return step
    return 0


def update_objective_function( z_orig, table, basic ):
    """
    The objective function of the original needs to be
    in terms of the NEW basic variables. The augmented
    variables can be removed.
    """
    temp = [row.copy() for row in table[:-1]]

    # Reduce each column in the original that corresponds to
    # a new basic variable.
    used_rows = []
    basic_row = {}
    # Reduce the columns to provide solution for variables NOT
    # in the list of new basic variables.
    basic_to_reduce = [val for val in basic if z_orig[val] != 0]

    for b_val in sorted(basic):
        print_matrix(temp, 'getting variable {0} in terms of non-basic variables'.format(b_val))
        for row_idx, row in enumerate(temp):
            # print('b_val {0} row[b_val] {1}'.format(b_val, row[b_val]))
            if not are_equal(row[b_val], 0) and row_idx not in used_rows:
                # Partial pivot, move row up if required.
                next_row = used_rows[-1] + 1 if len(used_rows) > 0 else 0
                if next_row != row_idx:
                    temp_row = temp[next_row]
                    temp[next_row] = temp[row_idx]
                    temp[row_idx] = temp_row
                    pivot_rows = next_row
                else:
                    pivot_rows = row_idx
                used_rows.append(pivot_rows)
                basic_row[b_val] = pivot_rows
                reduce_column(temp, (b_val, pivot_rows))
                break
        else:
            assert False, "unable to transition to phase II"

    # Process the original objective function so that
    # it consists of only basic variables.
    z_eq = [0] * len(temp[0])
    for idx, z_coeff in enumerate(z_orig[:-1]):
        if z_coeff != 0 and idx in basic:
            ratio = z_coeff / temp[basic_row[idx]][idx]
            for idx_row, coeff in enumerate(temp[basic_row[idx]][:-1]):
                if idx_row == idx:
                    continue
                z_eq[idx_row] -= ratio * coeff
            z_eq[-1] -= ratio * temp[basic_row[idx]][-1:][0]
        else:
            z_eq[idx] += z_coeff

    return z_eq


def find_row( matrix, column ):
    row = list(filter(lambda x: are_equal(x[1], 1), enumerate([row[column] for row in matrix])))
    assert len(row) == 1
    return row[0][0]


def phase_transition( augmented, n_initial, basic ):
    """
    It is possible that some of the augmented variables are
    now basic variables. These need to be swapped out using
    the transition rule.
    """

    def basic_in_aug():
        return list(filter(lambda x: x >= n_initial, basic)).copy()

    basic_in_augmented = basic_in_aug()
    original_basic = basic.copy()

    for b_val in basic_in_augmented:
        lowest_ratio = float('Inf')
        current_pivot = None
        current_var = 0
        pivot_rows = original_basic.index(b_val)
        for var in range(current_var, n_initial):
            pivot_coeff = augmented[pivot_rows][var]
            if var not in basic and not are_equal(pivot_coeff, 0):
                ratio = augmented[pivot_rows][-1] / pivot_coeff
                if ratio < lowest_ratio:
                    lowest_ratio = ratio
                    current_pivot = (var, pivot_rows)

        assert not current_pivot is None
        reduce_column(augmented, current_pivot)
        basic[pivot_rows] = current_pivot[0]

    assert len(basic_in_aug()) == 0


def augment( table ):
    m = len(table) - 1
    n = len(table[0]) - 1

    # the number of augmented variables will determine the new width of the the table.
    augmented = [None] * (len(table) - 1)

    column_count = [None] * n
    for r_idx, row in enumerate(table):
        for c_idx, e in enumerate(row[:-1]):
            cc = column_count[c_idx]
            if e != 0 and e != 1:
                column_count[c_idx] = -1
            elif e > 0 and (cc is not None and cc >= 0):
                column_count[c_idx] = -1
            elif cc is None and e == 1:
                column_count[c_idx] = r_idx
                # If last on the last row of the matrix.
            if r_idx == len(table[:-1]) - 1 and \
                    column_count[c_idx] is None:
                column_count[c_idx] = -1

    used_equations = set([])

    for idx, bc in enumerate(column_count):
        if bc != -1:
            if not bc in used_equations:
                used_equations.add(bc)
            else:
                column_count[idx] = -1

    num_aug = m - len(list(filter(lambda x: (x >= 0), column_count)))
    var_idx = n
    # New objective function.
    w = [0] * (len(table[0]) + num_aug)
    basic = []

    for idx, cc in enumerate(augmented):
        augmented[idx] = table[idx][:-1].copy() + ([0] * num_aug) + table[idx][-1:].copy()

        if idx in column_count:
            basic.append(column_count.index(idx))
            continue
        else:
            basic.append(var_idx)
            # Update the objective function to include the augmented variables.
            for e_idx, e in enumerate(augmented[idx][:-1]):
                w[e_idx] -= e
            w[-1] -= augmented[idx][-1]
            augmented[idx][var_idx] = 1
            var_idx += 1

    if num_aug == 0:
        augmented.append(table[-1].copy())
        return (basic, augmented)

    augmented.append(w)
    return (basic, augmented)


def indexes( lst, a ):
    result = []
    for i, x in enumerate(lst):
        if are_equal(x, a):
            result.append(i)
    return result


failed_at = ''

def allocate_ads( m, n, a, bb, c ):
    global failed_at

    table = initial_tableau(a, bb, c)
    n_initial = len(table[0]) - 1
    print_matrix(table, 'Initial tableaux')

    assert len(table[0]) - 1 >= len(a) - 1

    # Simplex method.
    # Phase I
    (basic, augmented) = augment(table)
    print_matrix(augmented, 'Augmented matrix')

    res = perform_simplex(augmented, basic)
    if res in [-1, 1]:
        failed_at = 'Performing Simplex on augmented matrix'
        return [res, None]

    print_matrix(augmented, 'Transition to phase II')

    augmented_objective = augmented[-1][-1]
    is_augmented = len(table[0]) != len(augmented[0])
    if is_augmented and not are_equal(augmented_objective, 0):
        failed_at = 'Checking result of Simplex on augmented objective, value was {0}' \
            .format(augmented_objective)
        return [-1, None]

    phase_transition(augmented, n_initial, basic)
    z_eq = update_objective_function(table[-1:][0].copy(), table, basic)

    phase_two = [row[:n_initial] + row[-1:] for row in augmented]
    phase_two[-1] = z_eq

    print_matrix(phase_two, 'Enter Phase II')
    res = perform_simplex(phase_two, basic)
    if res in [-1, 1]:
        failed_at = 'Performing Simplex in Phase II'
        return [res, None]

    result = []
    for i in range(n):
        if i in sorted(basic.copy()):
            result += phase_two[basic.index(i)][-1:]
        else:
            result += [0]
    return [0, result]


def ad_allocation():
    m, n = list(map(int, stdin.readline().split()))
    a = []
    for i in range(m):
        a += [list(map(int, stdin.readline().split()))]
    bb = list(map(int, stdin.readline().split()))
    c = list(map(int, stdin.readline().split()))
    anst, ansx = allocate_ads(m, n, a, bb, c)

    if anst == -1:
        print("No solution")
    if anst == 0:
        print("Bounded solution")
        print(' '.join(list(map(lambda x: '%.18f' % x, ansx))))
    if anst == 1:
        print("Infinity")

    if False and failed_at != '':
        print('Failed at: ' + failed_at)


if __name__ == "__main__":
    ad_allocation()