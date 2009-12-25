package mrcl.lib;

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.FloatBuffer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;

public class Matrix implements Writable {
	private String _name;
	private int _cols;
	private int _rows;
	private int _blockRows;
	private int _blockCols;

	private Matrix() {
	}

	public Matrix(String matrixName, int rows, int cols) {
		_name = matrixName;
		_rows = rows;
		_cols = cols;

		_blockRows = rows / Block.BLOCK_SIZE;
		_blockCols = cols / Block.BLOCK_SIZE;
	}

	public static Matrix createFillLocal(String matrixName, int rows, int cols,
			float fill) {
		Matrix matrix = new Matrix(matrixName, rows, cols);
		for (int blockRow = 0; blockRow <= matrix._blockRows; blockRow++) {
			for (int blockCol = 0; blockCol <= matrix._blockCols; blockCol++) {
				Content content = Content.make(new Block(matrix, blockRow,
						blockCol));
				content.fill(fill);
				content.writeLocal();
			}
		}
		return matrix;
	}

	public static Matrix createFillRemote(String matrixName, int rows,
			int cols, float fill, Configuration conf) {
		Matrix matrix = new Matrix(matrixName, rows, cols);
		for (int blockRow = 0; blockRow <= matrix._blockRows; blockRow++) {
			for (int blockCol = 0; blockCol <= matrix._blockCols; blockCol++) {
				Content content = Content.make(new Block(matrix, blockRow,
						blockCol));
				content.fill(fill);
				content.writeRemote(conf);
			}
		}
		matrix.writeRemote(conf);
		return matrix;
	}

	public static Matrix createRandomLocal(String matrixName, int rows,
			int cols, long seed) {
		Matrix matrix = new Matrix(matrixName, rows, cols);
		// ./matrix/MATRIX_NAME/descriptor // descriptor, which contains size
		// ./matrix/MATRIX_NAME/blocks/0/0 // block data
		// ./matrix/MATRIX_NAME/blocks/... // ...

		// int innerRows = MatrixBlockDescriptor.BLOCK_SIZE, innerCols =
		// MatrixBlockDescriptor.BLOCK_SIZE;
		for (int blockRow = 0; blockRow <= matrix._blockRows; blockRow++) {
			for (int blockCol = 0; blockCol <= matrix._blockCols; blockCol++) {
				Content content = Content.make(new Block(matrix, blockRow,
						blockCol));
				content.randomize(seed);
				content.writeLocal();
			}
		}
		return matrix;
	}

	public static Matrix createRandomRemote(String matrixName, int rows,
			int cols, int seed, Configuration conf) {
		Matrix matrix = new Matrix(matrixName, rows, cols);

		for (int blockRow = 0; blockRow <= matrix._blockRows; blockRow++) {
			for (int blockCol = 0; blockCol <= matrix._blockCols; blockCol++) {
				Content content = Content.make(new Block(matrix, blockRow,
						blockCol));
				content.randomize(seed);
				content.writeRemote(conf);
			}
		}
		matrix.writeRemote(conf);
		return matrix;
	}

	public static Matrix multiplyLocal(String resultName, Matrix a, Matrix b) {
		return multiplyLocal(resultName, a, b, 0, a.getBlockCols());
	}

	public static Matrix multiplyLocal(String resultName, Matrix a, Matrix b,
			int fromRound, int toRound) {
		int rows = a.getRows();
		int cols = b.getCols();
		int bRows = a.getBlockRows();
		int bCols = b.getBlockCols();

		Matrix result = Matrix.createFillLocal(resultName, rows, cols, 0);

		for (int round = fromRound; round < toRound; round++) {
			// make intermediate results
			Matrix inter = Matrix.createFillLocal("__inter__" + resultName, rows,
					cols, 0);

			for (int bRow = 0; bRow < bRows; bRow++) {
				for (int bCol = 0; bCol < bCols; bCol++) {
					Block interBlock = new Block(inter, bRow, bCol);
					Content interContent = Content.multiplyJava(interBlock,
							Content.readLocal(new Block(a, round, bCol)),
							Content.readLocal(new Block(b, bRow, round)));

					Block resultBlock = new Block(result, bRow, bCol);
					Content resultContent = Content.add(resultBlock, Content
							.readLocal(new Block(result, bRow, bCol)),
							interContent);
					resultContent.writeLocal();
				}
			}
		}
		return result;
	}

	public static Matrix multiplyRemote(String resultName, Matrix a, Matrix b,
			int round, Configuration conf) {
		int rows = a.getRows();
		int cols = b.getCols();
		int bRows = a.getBlockRows();
		int bCols = b.getBlockCols();

		// Matrix result = Matrix
		// .createFillRemote(resultName, rows, cols, 0, conf);

		// make intermediate results
		Matrix inter = Matrix.createFillRemote(String.format("__tmp/%s_%d",
				resultName, round), rows, cols, 0, conf);
		
		for (int bRow = 0; bRow < bRows; bRow++) {
			Content bContent = Content.readRemote(new Block(b, bRow, round),
					conf);
			for (int bCol = 0; bCol < bCols; bCol++) {
				Block interBlock = new Block(inter, bRow, bCol);
				Content interContent = Content.multiplyCublas(interBlock,
						Content.readRemote(new Block(a, round, bCol), conf),
						bContent);
				interContent.writeRemote(conf);
				//
				// Content resultContent = Content.add(result, Content
				// .read(new Block(result, bRow, bCol)), interContent);
				// resultContent.writeRemote(conf);
			}
		}
		return inter;
	}

	public static Matrix add(String resultName, Matrix a, Matrix b) {
		int bRows = a.getBlockRows();
		int bCols = a.getBlockCols();
		Matrix result = new Matrix(resultName, a.getRows(), a.getCols());
		for (int bRow = 0; bRow < bRows; bRow++) {
			for (int bCol = 0; bCol < bCols; bCol++) {
				Block resultBlock = new Block(result, bRow, bCol);
				Content resultContent = Content.add(resultBlock, Content
						.readLocal(new Block(a, bRow, bCol)), Content
						.readLocal(new Block(b, bRow, bCol)));
				resultContent.writeLocal();
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

	public String getMatrixPath() {
		return getPath(_name);
	}

	public String getMatrixDescPath() {
		return getMatrixPath() + "/description";
	}

	public int getRows() {
		return _rows;
	}

	public int getCols() {
		return _cols;
	}

	public FloatBuffer getFloatBufferLocal() {
		FloatBuffer result = FloatBuffer.allocate(_cols * _rows);
		for (int row = 0; row < _rows; row++) {
			for (int bCol = 0; bCol <= _blockCols; bCol++) {
				int from = Block.BLOCK_SIZE * bCol;
				int to = Math.min(Block.BLOCK_SIZE * (bCol + 1), _cols);

				Content content = Content.readLocal(new Block(this, row
						/ Block.BLOCK_SIZE, bCol));
				float[] array = content.getRow(row % Block.BLOCK_SIZE);
				for (int col = from, i = 0; col < to; col++, i++) {
					result.put(array[i]);
				}
			}
		}

		return result;
	}

	public FloatBuffer getFloatBufferRemote(Configuration conf) {
		FloatBuffer result = FloatBuffer.allocate(_cols * _rows);
		for (int row = 0; row < _rows; row++) {
			for (int bCol = 0; bCol < _blockCols; bCol++) {
				int from = Block.BLOCK_SIZE * bCol;
				int to = Math.min(Block.BLOCK_SIZE * (bCol + 1), _cols);

				Content content = Content.readRemote(new Block(this, row
						/ Block.BLOCK_SIZE, bCol), conf);
				float[] array = content.getRow(row % Block.BLOCK_SIZE);
				for (int col = from, i = 0; col < to; col++, i++) {
					result.put(array[i]);
				}
			}
		}
		return result;
	}

	public String getContentStringLocal() {
		StringBuilder b = new StringBuilder();
		for (int row = 0; row < _rows; row++) {
			for (int bCol = 0; bCol <= _blockCols; bCol++) {
				int from = Block.BLOCK_SIZE * bCol;
				int to = Math.min(Block.BLOCK_SIZE * (bCol + 1), _cols);

				Content content = Content.readLocal(new Block(this, row
						/ Block.BLOCK_SIZE, bCol));
				float[] array = content.getRow(row % Block.BLOCK_SIZE);
				for (int col = from, i = 0; col < to; col++, i++) {
					b.append(String.format("%10.3f\t", array[i]));
				}
			}
			b.append('\n');
		}

		return b.toString();
	}

	public String getContentStringRemote(Configuration conf) {
		StringBuilder b = new StringBuilder();
		for (int row = 0; row < _rows; row++) {
			for (int bCol = 0; bCol <= _blockCols; bCol++) {
				int from = Block.BLOCK_SIZE * bCol;
				int to = Math.min(Block.BLOCK_SIZE * (bCol + 1), _cols);

				Content content = Content.readRemote(new Block(this, row
						/ Block.BLOCK_SIZE, bCol), conf);
				float[] array = content.getRow(row % Block.BLOCK_SIZE);
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

	@Override
	public void readFields(DataInput input) throws IOException {
		_name = input.readUTF();
		_rows = input.readInt();
		_cols = input.readInt();
		_blockRows = _rows / Block.BLOCK_SIZE;
		_blockCols = _cols / Block.BLOCK_SIZE;
	}

	@Override
	public void write(DataOutput output) throws IOException {
		output.writeUTF(_name);
		output.writeInt(_rows);
		output.writeInt(_cols);
	}

	public void writeRemote(Configuration conf) {
		try {
			FileSystem fs = FileSystem.get(conf);
			DataOutputStream dos = fs.create(new Path(getDescPath(_name)));
			write(dos);
			dos.close();
			fs.close();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public static Matrix read(DataInput input) {
		try {
			Matrix matrix = new Matrix();
			matrix.readFields(input);
			return matrix;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public static Matrix addRemote(String resultName, Matrix a, Matrix b,
			Configuration conf) {
		int bRows = a.getBlockRows();
		int bCols = a.getBlockCols();
		Matrix result = new Matrix(resultName, a.getRows(), a.getCols());
		for (int bRow = 0; bRow < bRows; bRow++) {
			for (int bCol = 0; bCol < bCols; bCol++) {
				Block resultBlock = new Block(result, bRow, bCol);
				Content resultContent = Content.add(resultBlock, Content
						.readRemote(new Block(a, bRow, bCol), conf), Content
						.readRemote(new Block(b, bRow, bCol), conf));
				resultContent.writeRemote(conf);
			}
		}

		return result;
	}

	public static Matrix readRemote(String name, Configuration conf) {
		try {
			FileSystem fs = FileSystem.get(conf);
			DataInputStream dis = fs.open(new Path(Matrix.getDescPath(name)));
			Matrix matrix = Matrix.read(dis);
			dis.close();
			fs.close();
			return matrix;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public static String getPath(String name) {
		return "mrcl/matrix/" + name;
	}

	public static String getDescPath(String name) {
		return getPath(name) + "/desc";
	}

}
