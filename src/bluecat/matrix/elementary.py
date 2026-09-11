import torch


def rowswap(matrix, source_row, target_row):
    result = matrix.clone()

    temp = result[source_row].clone()
    result[source_row] = result[target_row]
    result[target_row] = temp

    return result

def rowscale(matrix, source_row, scaling_factor):
    result = matrix.clone()
    result[source_row] = scaling_factor * result[source_row]
    return result

def rowreplacement(matrix, first_row, second_row, j, k):
    result = matrix.clone()

    result[second_row] = (
        j * result[first_row] + k * result[second_row]
    )

    return result

def rref(matrix):
    result = matrix.clone().float()

    rows, cols = result.shape
    pivot_row = 0

    for col in range(cols):
        if pivot_row >= rows:
            break

        # Find a nonzero entry in this column
        swap_row = None
        for row in range(pivot_row, rows):
            if result[row, col] != 0:
                swap_row = row
                break

        # If the whole column is zero, move to the next column
        if swap_row is None:
            continue

        # Swap the nonzero row into the pivot position
        if swap_row != pivot_row:
            result = rowswap(result, pivot_row, swap_row)

        # Scale the pivot to 1
        pivot = result[pivot_row, col]

        if pivot != 1:
            result = rowscale(result, pivot_row, 1 / pivot)

        # Make everything below the pivot zero
        for row in range(pivot_row + 1, rows):
            if result[row, col] != 0:
                factor = -result[row, col]
                result = rowreplacement(
                    result,
                    pivot_row,
                    row,
                    factor,
                    1
                )

        pivot_row += 1

    return result

if __name__ == "__main__":
    A = torch.tensor([
        [1., 3., 0., 0., 3.],
        [0., 0., 1., 0., 9.],
        [0., 0., 0., 1., -4.]
    ])

    print("Original matrix:")
    print(A)

    B = rowswap(A, 0, 1)
    print("After swapping R1 and R2:")
    print(B)

    C = rowscale(B, 0, 1/3)
    print("After scaling R1 by 1/3:")
    print(C)

    D = rowreplacement(C, 0, 2, -3, 1)
    print("After R3 = -3R1 + R3:")
    print(D)

    E = rref(A)

    print("RREF result:")
    print(E)