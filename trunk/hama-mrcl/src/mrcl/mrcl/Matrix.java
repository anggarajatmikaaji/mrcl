package mrcl;


import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

public class Matrix {
	private String _name;
	private int _cols;
	private int _rows;
	private int _blockRows;
	private int _blockCols;

	public Matrix(String matrixName, int rows, int cols) {
		_name = matrixName;
		_rows = rows;
		_cols = cols;

		_blockRows = rows / Block.BLOCK_SIZE;
		_blockCols = cols / Block.BLOCK_SIZE;
	}

	public static Matrix create(String matrixName, int rows, int cols) {
		return createFill(matrixName, rows, cols, 0);
	}

	public static Matrix createFill(String matrixName, int rows, int cols,
			float fill) {
		Matrix matrix = new Matrix(matrixName, rows, cols);
		for (int blockRow = 0; blockRow <= matrix._blockRows; blockRow++) {
			for (int blockCol = 0; blockCol <= matrix._blockCols; blockCol++) {
				Block.createFill(matrix, blockRow, blockCol, fill);
			}
		}
		return matrix;
	}

	public static Matrix createRandom(String matrixName, int rows, int cols,
			long seed) {
		Matrix matrix = new Matrix(matrixName, rows, cols);
		// ./matrix/MATRIX_NAME/descriptor // descriptor, which contains size
		// ./matrix/MATRIX_NAME/blocks/0/0 // block data
		// ./matrix/MATRIX_NAME/blocks/... // ...

		// int innerRows = MatrixBlockDescriptor.BLOCK_SIZE, innerCols =
		// MatrixBlockDescriptor.BLOCK_SIZE;
		for (int blockRow = 0; blockRow <= matrix._blockRows; blockRow++) {
			for (int blockCol = 0; blockCol <= matrix._blockCols; blockCol++) {
				Block.createRandom(matrix, blockRow, blockCol, seed);
			}
		}

		return matrix;
	}

	public static Matrix multiply(String resultName, Matrix a, Matrix b) {
		return multiply(resultName, a, b, 0, a.getBlockCols());
	}

	public static Matrix multiply(String resultName, Matrix a, Matrix b,
			int fromRound, int toRound) {
		int rows = a.getRows();
		int cols = b.getCols();
		int bRows = a.getBlockRows();
		int bCols = b.getBlockCols();

		Matrix result = Matrix.createFill(resultName, rows, cols, 0);

		// map

		// List<Matrix> interList = new ArrayList<Matrix>();

		for (int round = fromRound; round < toRound; round++) {
			// make intermediate results
			Matrix inter = Matrix.createFill("__inter__" + resultName, rows,
					cols, 0);

			for (int bRow = 0; bRow < bRows; bRow++) {
				for (int bCol = 0; bCol < bCols; bCol++) {
					Block interBlock = Block.multiplyJava(inter, a.getBlock(
							round, bCol), b.getBlock(bRow, round));
					Block resultBlock = Block.add(result, result.getBlock(bRow,
							bCol), interBlock);
					resultBlock.write();
				}
			}
		}
		return result;
	}

	public static Matrix add(String resultName, Matrix a, Matrix b) {
		int bRows = a.getBlockRows();
		int bCols = a.getBlockCols();
		Matrix result = new Matrix(resultName, a.getRows(), a.getCols());
		for (int bRow = 0; bRow < bRows; bRow++) {
			for (int bCol = 0; bCol < bCols; bCol++) {
				Block resultBlock = Block.add(result, a.getBlock(bRow, bCol), b
						.getBlock(bRow, bCol));
				resultBlock.write();
			}
		}

		return result;
	}

	public int getBlockCols() {
		if (_cols % Block.BLOCK_SIZE > 0)
			return (_cols / Block.BLOCK_SIZE) + 1;
		return _cols / Block.BLOCK_SIZE;
	}

	public int getBlockRows() {
		if (_rows % Block.BLOCK_SIZE > 0)
			return (_rows / Block.BLOCK_SIZE) + 1;
		return _rows / Block.BLOCK_SIZE;
	}

	public String matrixPath() {
		return "mrcl/matrix/" + _name;
	}

	public String matrixDescPath() {
		return matrixPath() + "/description";
	}

	public int getRows() {
		return _rows;
	}

	public int getCols() {
		return _cols;
	}

	public Block getBlock(int blockRow, int blockCol) {
		return Block.read(this, blockRow, blockCol);
	}

	public FloatBuffer getFloatBuffer() {
		FloatBuffer result = FloatBuffer.allocate(_cols * _rows);
		for (int row = 0; row < _rows; row++) {
			for (int bCol = 0; bCol <= _blockCols; bCol++) {
				int from = Block.BLOCK_SIZE * bCol;
				int to = Math.min(Block.BLOCK_SIZE * (bCol + 1), _cols);

				Block block = Block.read(this, row / Block.BLOCK_SIZE, bCol);
				float[] array = block.getRow(row % Block.BLOCK_SIZE);
				for (int col = from, i = 0; col < to; col++, i++) {
					result.put(array[i]);
				}
			}
		}

		return result;
	}

	public String getContentString() {
		StringBuilder b = new StringBuilder();
		for (int row = 0; row < _rows; row++) {
			for (int bCol = 0; bCol <= _blockCols; bCol++) {
				int from = Block.BLOCK_SIZE * bCol;
				int to = Math.min(Block.BLOCK_SIZE * (bCol + 1), _cols);

				Block block = Block.read(this, row / Block.BLOCK_SIZE, bCol);
				float[] array = block.getRow(row % Block.BLOCK_SIZE);
				for (int col = from, i = 0; col < to; col++, i++) {
					b.append(String.format("%10.3f\t", array[i]));
				}
			}
			b.append('\n');
		}

		return b.toString();
	}

	public String getName() {
		return _name;
	}

}
