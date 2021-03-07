import random
from typing import ForwardRef

from PIL import Image, ImageDraw


def sample(xs):
    return random.sample(xs, 1)[0]


class Cell(object):
    __slots__ = ["row", "column", "links", "north", "south", "west", "east"]

    def __init__(self, row, column):
        self.row = row
        self.column = column
        self.links = set()

    def __repr__(self):
        return f"Cell({self.row}, {self.column})"

    def __hash__(self):
        return hash((self.row, self.column))

    def __eq__(self, other):
        if isinstance(other, Cell):
            return (self.row, self.column) == (other.row, other.column)
        return False

    def link(self, other, reverse=True):
        self.links.add(other)
        if reverse:
            other.link(self, reverse=False)

    def is_linked(self, other):
        return other in self.links

    def unlink(self, other, reverse=True):
        self.links.remove(other)
        if reverse:
            other.unlink(self, reverse=False)

    def neighbors(self):
        xs = [self.north, self.south, self.west, self.east]
        return list(filter(lambda x: not x is None, xs))

    def distances(self):
        """Constructs a new `Distances` dictionary using `self`.

        Creates dictionary that contains all the distances in
        the grid as calculated with refernce to self.
        """
        d = Distances(self)
        frontier = [self]
        while frontier:
            new_frontier = []
            for cell in frontier:
                for linked in cell.links:
                    if linked in d:
                        continue
                    d[linked] = d[cell] + 1
                    new_frontier.append(linked)
            frontier = new_frontier
        return d


class Grid(object):
    __slots__ = ["rows", "columns", "grid", "root"]

    def __init__(self, rows, columns, rootproc=None):
        self.rows = rows
        self.columns = columns
        self.grid = self._prepare_grid()
        self._configure_cells()
        if rootproc:
            self.root = rootproc(self)
        else:
            self.root = self[0, 0]

    def __getitem__(self, key):
        row, col = key
        if row < 0 or row >= self.rows:
            return None
        if col < 0 or col >= self.columns:
            return None
        return self.grid[row][col]

    def __str__(self):
        output = ["+" + "----+" * self.columns + "\n"]
        for row in self.each_row():
            top = ["|"]
            bottom = ["+"]
            for cell in row:
                body = f" {self._contents_of(cell)} "
                if cell and cell.is_linked(cell.east):
                    east_boundary = " "
                else:
                    east_boundary = "|"
                top.append(body)
                top.append(east_boundary)
                if cell and cell.is_linked(cell.south):
                    south_boundary = "    "
                else:
                    south_boundary = "----"
                corner = "+"
                bottom.append(south_boundary)
                bottom.append(corner)
            top.append("\n")
            bottom.append("\n")
            output.append("".join(top))
            output.append("".join(bottom))
        return "".join(output)

    def each_row(self):
        for row in self.grid:
            yield row

    def each_cell(self):
        for row in self.grid:
            for cell in row:
                yield cell

    def random_cell(self):
        row = sample(self.grid)
        return sample(row)

    def size(self):
        return self.rows * self.columns

    def root_distances(self):
        return self.root.distances()

    def deadends(self):
        return filter(lambda x: len(x.links) == 1, self.each_cell())

    def render(self, **kwargs):
        cell_size = kwargs.get("cell_size", 10)
        modes = kwargs.get("modes", ["backgrounds", "walls"])
        img_width = cell_size * self.columns
        img_height = cell_size * self.rows
        background = (255, 255, 255)
        wall = (0, 0, 0)
        img = Image.new("RGB", (img_width + 1, img_height + 1), background)
        draw = ImageDraw.Draw(img)
        for mode in modes:
            for cell in self.each_cell():
                x1 = cell.column * cell_size
                y1 = cell.row * cell_size
                x2 = (cell.column + 1) * cell_size
                y2 = (cell.row + 1) * cell_size
                if mode == "backgrounds":
                    color = self._background_color_for(cell)
                    draw.rectangle([(x1, y1), (x2, y2)], fill=color)
                elif mode == "walls":
                    if not cell.north:
                        draw.line([(x1, y1), (x2, y1)], fill=wall)
                    if not cell.west:
                        draw.line([(x1, y1), (x1, y2)], fill=wall)
                    if not cell.is_linked(cell.east):
                        draw.line([(x2, y1), (x2, y2)], fill=wall)
                    if not cell.is_linked(cell.south):
                        draw.line([(x1, y2), (x2, y2)], fill=wall)
        return img

    def _prepare_grid(self):
        def create_row(r):
            return [Cell(r, c) for c in range(self.columns)]

        return [create_row(r) for r in range(self.rows)]

    def _configure_cells(self):
        for cell in self.each_cell():
            if not cell:
                continue
            row, col = cell.row, cell.column
            cell.north = self[row - 1, col]
            cell.south = self[row + 1, col]
            cell.west = self[row, col - 1]
            cell.east = self[row, col + 1]

    def _contents_of(self, cell):
        return "  "

    def _background_color_for(self, cell):
        return 255, 255, 255


class DistanceGrid(Grid):
    def __init__(self, wrappee, distances=None):
        distances = distances or wrappee.root_distances()
        self.rows = wrappee.rows
        self.columns = wrappee.columns
        self.grid = wrappee.grid
        self.root = wrappee.root
        self.distances = distances

    def _contents_of(self, cell):
        if self.distances and cell in self.distances:
            return f"{self.distances[cell]:02x}"
        else:
            return super()._contents_of(cell)


class ColoredGrid(Grid):
    def __init__(self, wrappee, distances=None):
        distances = distances or wrappee.root_distances()
        self.rows = wrappee.rows
        self.columns = wrappee.columns
        self.grid = wrappee.grid
        self.distances = distances
        self.root = wrappee.root
        _, self.maximum = distances.max()

    def _background_color_for(self, cell):
        if not cell in self.distances:
            return "purple"
        distance = self.distances[cell]
        intensity = (self.maximum - distance) / self.maximum
        dark = round(255 * intensity)
        bright = 128 + round(127 * intensity)
        return dark, bright, dark


class Distances(object):
    __slots__ = ["root", "cells"]

    def __init__(self, root):
        self.root = root
        self.cells = {root: 0}

    def __getitem__(self, key):
        return self.cells[key]

    def __setitem__(self, key, value):
        self.cells[key] = value

    def __iter__(self):
        return iter(self.cells)

    def __str__(self):
        return str(self.cells)

    def max(self):
        max_distance = 0
        max_cell = self.root
        for cell, distance in self.cells.items():
            if distance > max_distance:
                max_cell = cell
                max_distance = distance
        return max_cell, max_distance

    def path_to(self, goal):
        current = goal
        path = Distances(self.root)
        path[current] = self.cells[current]
        while current != self.root:
            for neighbor in current.links:
                if self.cells[neighbor] < self.cells[current]:
                    path[neighbor] = self.cells[neighbor]
                    current = neighbor
                    break
        return path


class Mask(object):
    __slots__ = ["rows", "columns", "bits"]

    def __init__(self, rows, columns):
        self.rows, self.columns = rows, columns
        self.bits = [[True for _ in range(self.columns)] for _ in range(self.rows)]

    def __getitem__(self, key):
        row, col = key
        if row < 0 or row >= self.rows:
            return False
        if col < 0 or col >= self.columns:
            return False
        return self.bits[row][col]

    def __setitem__(self, key, value):
        row, col = key
        self.bits[row][col] = value

    def __len__(self):
        count = 0
        for row in range(self.rows):
            for col in range(self.columns):
                if self.bits[row][col]:
                    count += 1
        return count

    def random_location(self):
        while True:
            row = random.randint(0, self.rows - 1)
            col = random.randint(0, self.columns - 1)
            if self[row, col]:
                return row, col


class MaskedGrid(Grid):
    def __init__(self, mask):
        self.mask = mask
        super().__init__(mask.rows, mask.columns)

    def _prepare_grid(self):
        def create_cell(row, col):
            if self.mask[row, col]:
                return Cell(row, col)
            else:
                return None

        def create_row(row):
            return [create_cell(row, col) for col in range(self.columns)]

        return [create_row(row) for row in range(self.rows)]

    def random_cell(self):
        row, col = self.mask.random_location()
        return self[row, col]

    def size(self):
        return len(self.mask)


def longest_path(d):
    start, _ = d.max()
    d = start.distances()
    goal, _ = d.max()
    return d.path_to(goal)


def binary_tree(grid):
    for cell in grid.each_cell():
        neighbors = []
        if cell.north:
            neighbors.append(cell.north)
        if cell.east:
            neighbors.append(cell.east)
        if neighbors:
            neighbor = sample(neighbors)
            cell.link(neighbor)
    return grid


def sidewinder(grid):
    for row in grid.each_row():
        run = []
        for cell in row:
            run.append(cell)
            at_east_bound = cell.east is None
            at_north_bound = cell.north is None
            should_close = at_east_bound or (
                not at_north_bound and random.randint(0, 1) == 0
            )
            if should_close:
                member = sample(run)
                if member.north:
                    member.link(member.north)
                run = []
            else:
                cell.link(cell.east)
    return grid


def aldous_broder(grid):
    cell = grid.random_cell()
    unvisited = grid.size() - 1
    while unvisited > 0:
        neighbor = sample(cell.neighbors())
        if not neighbor.links:
            cell.link(neighbor)
            unvisited -= 1
        cell = neighbor
    return grid


def wilsons(grid):
    unvisited = list(grid.each_cell())
    first = sample(unvisited)
    unvisited.remove(first)
    while unvisited:
        cell = sample(unvisited)
        path = [cell]
        while cell in unvisited:
            cell = sample(cell.neighbors())
            if cell in path:
                pos = path.index(cell)
                path = path[0 : pos + 1]
            else:
                path.append(cell)
        for i in range(0, len(path) - 1):
            path[i].link(path[i + 1])
            unvisited.remove(path[i])
    return grid


def hunt_and_kill(grid):
    current = grid.random_cell()
    while current:
        unvisited_neighbors = list(filter(lambda x: not x.links, current.neighbors()))
        if unvisited_neighbors:
            neighbor = sample(unvisited_neighbors)
            current.link(neighbor)
            current = neighbor
        else:
            current = None
            for cell in grid.each_cell():
                visited_neighbors = list(filter(lambda x: x.links, cell.neighbors()))
                if not cell.links and visited_neighbors:
                    current = cell
                    neighbor = sample(visited_neighbors)
                    current.link(neighbor)
                    break
    return grid


def recursive_backtracker(grid, start_at=None):
    start_at = start_at or grid.random_cell()
    stack = []
    stack.append(start_at)
    while stack:
        current = stack[-1]
        neighbors = list(filter(lambda x: not x.links, current.neighbors()))
        if not neighbors:
            stack.pop()
        else:
            neighbor = sample(neighbors)
            current.link(neighbor)
            stack.append(neighbor)
    return grid


def test_algorithms():
    algorithms = [
        binary_tree,
        sidewinder,
        aldous_broder,
        wilsons,
        hunt_and_kill,
        recursive_backtracker,
    ]
    tries = 100
    size = 20
    averages = dict()
    for algorithm in algorithms:
        print(f"running {algorithm.__name__}")
        deadends_counts = []
        for i in range(tries):
            grid = Grid(size, size)
            algorithm(grid)
            deadends_counts.append(len(list(grid.deadends())))
        total_deadends = sum(deadends_counts)
        averages[algorithm] = total_deadends / len(deadends_counts)
    total_cells = size * size
    print()
    print(f"average dead-dends per {size}x{size} maze ({total_cells} cells):")
    print()
    sorted_algorithms = sorted(averages.items(), key=lambda x: -x[1])
    for algorithm, average in sorted_algorithms:
        percentage = int(average * 100.0 / (size * size))
        name = algorithm.__name__
        print(f"{name: <21} : {int(average): >3}/{total_cells} ({percentage}%)")
    print()


if __name__ == "__main__":
    mask = Mask(5,5)
    mask[0,0] = False
    mask[2,2] = False
    mask[4,4] = False
    grid = MaskedGrid(mask)
    recursive_backtracker(grid)
    print(grid)
