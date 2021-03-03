import math

class Grid2D:
	'''
	2D Grid used to partition a subset of the plane. Used to store values.
	'''
	def __init__(self, xmin, ymin, xmax, ymax, cell_width, cell_height):
		assert(xmin < xmax and ymin < ymax)

		# get grid size
		self.nrows = math.ceil((ymax - ymin) / cell_height)
		self.ncols = math.ceil((xmax - xmin) / cell_width)

		# set cell dimensions
		self.cell_width = cell_width
		self.cell_height = cell_height

		# get grid boundaries
		self.xmin = xmin
		self.xmax = xmin + self.ncols * cell_width
		self.ymin = ymax - self.nrows * cell_height
		self.ymax = ymax

		# initialize hash table
		self.grid = {}

	def __gridkey(self, x, y):
		'''
		Returns integer index of grid cell containing (x, y). Grid is
		indexed with cell in top left as (0, 0).
		'''
		# Grid cells include their left and upper boundaries.
		row_idx = math.floor((self.ymax - y) / self.cell_height)
		col_idx = math.floor((x - self.xmin) / self.cell_width)

		# Edge cases: (x, y) is on the lower or right boundary of grid
		row_idx = min(row_idx, self.nrows - 1)
		col_idx = min(col_idx, self.ncols - 1)
		return (row_idx, col_idx)

	def shape(self):
		return (self.nrows, self.ncols)

	def items(self):
		return self.grid.items()

	def contains(self, x, y):
		'''
		Return: 
			bool
				True if point in grid, False if out of bounds
		'''
		return x >= self.xmin and x <= self.xmax and y >= self.ymin and y <= self.ymax

	def insert(self, x, y, value):
		'''
		Attempts to insert value into grid. Returns true if successful, 
		False if point is out of grid bounds.

		Parameters:
			x : float
				x coordinate of point
			y : float
				y coordinate of point

		Return:
			bool
				True if point added to grid, False if point out of bounds.
		'''
		# Check if point is out of bounds.
		if not self.contains(x, y):
			return False

		key = self.__gridkey(x, y)

		if key in self.grid.keys():
			self.grid[key].append(value)
		else:
			self.grid[key] = [value]

		return True

	def get(self, key):
		'''
		Parameters:
			key : tuple (row_idx, col_idx)
				Key to grid cell
		Return:
			value list:
				List of values in grid cell, may be empty.
		'''
		return self.grid.get(key, [])


	def plot_gridlines(self, ax, colors='k', linewidth=1.0):
		'''
		Plot gridlines on a specified axis. See Axes.vlines() and Axes.hlines()
		for options details.
		'''
		xs_dashed = [self.xmin + i * self.cell_width for i in range(0, self.ncols + 1, 2)]
		ax.vlines(xs_dashed, self.ymin, self.ymax, colors=colors, linestyles='solid', linewidth=linewidth)

		xs_dotted = [self.xmin + i * self.cell_width for i in range(1, self.ncols + 1, 2)]
		ax.vlines(xs_dotted, self.ymin, self.ymax, colors=colors, linestyles='dotted', linewidth=linewidth)

		ys_dashed = [self.ymax - i * self.cell_height for i in range(0, self.nrows + 1, 2)]
		ax.hlines(ys_dashed, self.xmin, self.xmax, colors=colors, linestyles='solid', linewidth=linewidth)

		ys_dotted = [self.ymax - i * self.cell_height for i in range(1, self.nrows + 1, 2)]
		ax.hlines(ys_dotted, self.xmin, self.xmax, colors=colors, linestyles='dotted', linewidth=linewidth)

	def plot_cell_gridlines(self, ax, row_idx, col_idx, colors='k', linestyles='solid'):
		ymin = self.ymax - (row_idx + 1) * self.cell_height
		ymax = self.ymax - row_idx * self.cell_height

		xmin = self.xmin + col_idx * self.cell_width
		xmax = self.xmin + (col_idx + 1) * self.cell_width

		ax.vlines([xmin, xmax], ymin, ymax, colors=colors, linestyles=linestyles)
		ax.hlines([ymin, ymax], xmin, xmax, colors=colors, linestyles=linestyles)
