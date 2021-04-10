import math
from functools import reduce
from itertools import combinations, chain
from typing import List, Tuple, Set, Optional, Iterable, Dict

Coord = Tuple[int, int]


class Sudoku:
    def __init__(self, grid: List[List[int]]):
        size = len(grid)
        box_size = int(math.sqrt(size))

        assert size == box_size * box_size

        self.grid = grid
        self.size = size
        self.box_size = box_size

        self.candidates = [[self._find_candidates(i, j) for j in range(self.size)] for i in range(self.size)]

    def _find_candidates(self, i, j) -> Set[int]:
        """
        Computes the set of all numbers influencing the given coordinates. This includes the row, column and box.
        """
        # the value is already set, no possible values
        if self.grid[i][j] != 0:
            options = set()
        else:

            # unique values contained in the neighborhood (row,col,box) of this cell
            neighborhood = set(self.grid[i]) \
                           | set(row[j] for row in self.grid) \
                           | set(self.grid[a][b] for a, b in self._get_box_coords(i, j))

            # remove the already set values from the candidate list
            nums = set(range(1, self.size + 1))
            options = nums.difference(neighborhood)
        return options

    def __repr__(self) -> str:
        out = ''
        for row in self.grid:
            out += str(row) + '\n'
        return out

    def __str__(self) -> str:
        # Line separating boxes horizontally
        sep = '+'.join(['-' * (2 * self.box_size + 1)] * self.box_size) + '\n'

        lines = []
        for i in range(self.size):
            # row cells split into groups according to the box they belong to
            cells = [self.grid[i][j:j + self.box_size] for j in range(0, self.size, self.box_size)]

            # Join the groups with spaces, add space in the front and back
            cells = [' ' + ' '.join(map(str, cell)) + ' ' for cell in cells]

            # join all the boxes of the row
            lines.append('|'.join(cells) + '\n')

        groups = [''.join(lines[i:i + self.box_size]) for i in range(0, self.size, self.box_size)]
        return sep.join(groups)

    def _get_box_coords(self, r: int, c: int) -> List[Coord]:
        indices = []

        # coordinates of the upper left corner
        ci = (r // self.box_size) * self.box_size
        cj = (c // self.box_size) * self.box_size

        for i in range(self.box_size):
            for j in range(self.box_size):
                indices.append((ci + i, cj + j))

        return indices

    def _get_row_coords(self, r: int) -> List[Coord]:
        return [(r, j) for j in range(self.size)]

    def _get_col_coords(self, c: int) -> List[Coord]:
        return [(i, c) for i in range(self.size)]

    def _get_neighbour_coords(self, r: int, c: int) -> List[List[Coord]]:
        return [
            self._get_row_coords(r),
            self._get_col_coords(c),
            self._get_box_coords(r, c),
        ]

    def _has_duplicates(self, search_list: Iterable[int]) -> bool:
        """
        Check for duplicates in the given list
        """
        seen = set()
        for num in search_list:
            if num in seen and num != 0:
                return True
            else:
                seen.add(num)
        return False

    def _has_all_nums(self, search_list: Iterable[int]) -> Tuple[bool, Optional[int]]:
        """
        Check if all numbers exist in the given search list.
        """
        remaining = set(search_list)
        for num in range(1, self.size + 1):
            if num not in remaining:
                return False, num
        return True, None

    def _erase(self, numbers: List[int], indices: List[Coord], retain: List[Coord]) -> List[Coord]:
        """
        Deletes given numbers from all candidate sets for the given indices. Retains the numbers for all candidates
        contained in the retain list.
        """

        num_set = set(numbers)

        deleted = []
        for i, j in indices:
            if (i, j) in retain:
                continue

            if self.candidates[i][j] & num_set:
                self.candidates[i][j].difference_update(num_set)
                deleted.append((i, j))
        return deleted

    def _set_closed_tuple_candidates(self, numbers: List[int], indices: List[Coord]) -> List[Coord]:
        """
        Removes all candidates not contained in numbers from all cells contained in the index list. Returns modified
        positions.
        """
        num_set = set(numbers)

        changed = []
        for i, j in indices:
            # beware triples where the whole triple is not in each box

            new_candidates = self.candidates[i][j] & num_set

            # Only update if the new candidates weren't already part of the candidate set
            if len(self.candidates[i][j]) != len(new_candidates):
                self.candidates[i][j] = new_candidates
                changed.append((i, j))

        return changed

    def _get_candidate_coord_lists(self, indices) -> List[List[Coord]]:
        """
        For each candidate numbers a list is created which contains the cell coordinates of all cells where the number
        is a candidate.
        """
        candidate_cells = [[] for _ in range(self.size + 1)]
        for i, j in indices:
            for num in self.candidates[i][j]:
                candidate_cells[num].append((i, j))
        return candidate_cells

    def _find_closed_tuples(self, indices: List[Coord], tuple_sizes: List[int]) -> List[Coord]:
        """
        Find and eliminate tuples of cells that form a closed system with a fixed set of candidates that must appear in
        these cells. This allows the elimination of all other candidates in these cells.
        """

        tuple_sizes = set(tuple_sizes) & {1, 2, 3}  # allowed sizes are only 1, 2 and 3
        candidate_lists = self._get_candidate_coord_lists(indices)

        # Maps tuple length to list of numbers appearing at this length
        len2candidates: Dict[int, List[int]] = {size: [] for size in tuple_sizes}
        for num, candidate_cells in enumerate(candidate_lists):
            size = len(candidate_cells)
            if size in len2candidates.keys():
                len2candidates[size].append(num)

        # Triples can also be formed from numbers that appear only once
        if 3 in tuple_sizes:
            len2candidates[3] += len2candidates[2]

        closed_tuples: List[Tuple[List[Coord], List[int]]] = []
        for size in tuple_sizes:
            # make every possible combination
            for numbers in list(combinations(len2candidates[size], size)):

                # compute the union of all positions
                candidate_positions = [set(candidate_lists[num]) for num in numbers]
                common_positions = reduce(lambda a, b: a | b, candidate_positions, set())

                if len(common_positions) == size:
                    # unique numbers (pair or triple) found
                    closed_tuples.append((list(common_positions), list(numbers)))

        erased = []
        for tuple_cells, tuple_nums in closed_tuples:
            self._set_closed_tuple_candidates(tuple_nums, tuple_cells)
            erased += self._erase(tuple_nums, indices, tuple_cells)
        return erased

    def _intersection_removal(self, box_cells: List[Coord]) -> List[Coord]:
        """
        Find two or three cells in a box which are the only locations for a candidate number. If they are all in the
        same row or same column, the candidate can be removed from all other locations within this row or column.

        :return: The list of modified cell locations
        """

        modified = []
        for num, candidate_cells in enumerate(self._get_candidate_coord_lists(box_cells)):
            # Select all candidates where only 2 or 3 cells remain in the group
            if len(candidate_cells) not in [2, 3]:
                continue

            row_reduction, col_reduction = True, True
            i0, j0 = candidate_cells[0]
            for i, j in candidate_cells[1:]:
                row_reduction = row_reduction and (i == i0)
                col_reduction = col_reduction and (j == j0)

            if row_reduction:
                line = self._get_row_coords(i0)
                modified += self._erase([num], line, candidate_cells)

            if col_reduction:
                line = self._get_col_coords(j0)
                modified += self._erase([num], line, candidate_cells)

        return modified

    def has_constraint_violations(self) -> Tuple[bool, Optional[str]]:
        """
        Check for constraint violations on the board. Constraints checked for each row, column and box:
        - No duplicate numbers
        - All numbers or can be set according to the possibilities
        """

        row_coords = [(f'row {r}', self._get_row_coords(r)) for r in range(self.size)]
        col_coords = [(f'col {c}', self._get_col_coords(c)) for c in range(self.size)]
        box_coords = [(f'box {r, c}', self._get_box_coords(r, c))
                      for r in range(0, self.size, self.box_size)
                      for c in range(0, self.size, self.box_size)]

        for group_list in [row_coords, col_coords, box_coords]:
            for k, (group_desc, coords) in enumerate(group_list):

                group_numbers = [self.grid[i][j] for i, j in coords]

                if self._has_duplicates(group_numbers):
                    return False, f'Duplicate values in {group_desc}'

                for i, j in coords:
                    group_numbers += self.candidates[i][j]

                possible, missing_num = self._has_all_nums(group_numbers)
                if not possible:
                    return False, f'Cannot place {missing_num:d} in {group_desc}'

        return True, None

    def set_position(self, r: int, c: int, num: int) -> None:
        """
        Place a number in the given cell and propagate the changes to all affected neighbors, filling and propagating
         recursively if necessary.
        """

        # set the number and clear all candidates for this cell
        self.grid[r][c] = num
        self.candidates[r][c] = set()

        # Remove the number from all candidate sets affected by the change. Collect the changed indices in 'erased'.
        initial_neighbors: List[Coord] = list(set(chain(*self._get_neighbour_coords(r, c))))
        erased: List[Coord] = [(r, c)] + self._erase([num], initial_neighbors, [])

        while erased:
            i, j = erased.pop()

            neighbors: List[List[Coord]] = self._get_neighbour_coords(i, j)
            for cell_coords in neighbors:
                erased += self._find_closed_tuples(cell_coords, [1, 2, 3])

            erased += self._intersection_removal(self._get_box_coords(i, j))

    def solve_trivial_cells(self) -> None:
        """
        Solves simple cells that can be set immediately with simple reasoning.
        """

        rows = [self._get_row_coords(i) for i in range(self.size)]
        cols = [self._get_col_coords(j) for j in range(self.size)]
        boxes = [self._get_box_coords(i0, j0)
                 for j0 in range(0, self.size, self.box_size)
                 for i0 in range(0, self.size, self.box_size)]

        for group_cells in rows + cols + boxes:
            self._find_closed_tuples(group_cells, tuple_sizes=[1, 2])

        for box_cells in boxes:
            self._intersection_removal(box_cells)
