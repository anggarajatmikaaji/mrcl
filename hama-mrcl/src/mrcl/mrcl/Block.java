package mrcl;


import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.util.Random;
import jcuda.*;
import jcuda.jcublas.JCublas;

public class Block {
	private Matrix _matrix;
	public static final int BLOCK_SIZE = 10;
	public static final int BLOCK_SIZE_2 = BLOCK_SIZE * BLOCK_SIZE;
	private int _blockRow;
	private int _blockCol;
	private int _innerRows;
	private int _innerCols;

	private ByteBuffer _byteBuffer;
	private FloatBuffer _floatBuffer;

	public Block(Matrix matrix, int blockRow, int blockCol) {
		_matrix = matrix;
		_blockRow = blockRow;
		_blockCol = blockCol;

		int rows = matrix.getRows();
		int cols = matrix.getCols();

		int fromRow = blockRow * BLOCK_SIZE;
		int toRow = Math.min((blockRow + 1) * BLOCK_SIZE, rows);
		int fromCol = blockCol * BLOCK_SIZE;
		int toCol = Math.min((blockCol + 1) * BLOCK_SIZE, cols);

		_innerRows = toRow - fromRow;
		_innerCols = toCol - fromCol;

		_byteBuffer = ByteBuffer.allocate(BLOCK_SIZE_2 * 4);
		_floatBuffer = _byteBuffer.asFloatBuffer();

	}

	public static Block createFill(Matrix matrix, int blockRow, int blockCol,
			float fillValue) {
		Block block = new Block(matrix, blockRow, blockCol);
		for (int row = 0; row < block._innerRows; row++) {
			for (int col = 0; col < block._innerCols; col++) {
				block._floatBuffer.position(row * BLOCK_SIZE + col);
				block._floatBuffer.put(fillValue);
			}
		}
		block.write();
		return block;
	}

	public static Block createRandom(Matrix matrix, int blockRow, int blockCol,
			long seed) {
		Block block = new Block(matrix, blockRow, blockCol);
		Random r = new Random(seed + BLOCK_SIZE * blockRow + blockCol);
		for (int row = 0; row < block._innerRows; row++) {
			for (int col = 0; col < block._innerCols; col++) {
				block._floatBuffer.position(row * BLOCK_SIZE + col);
				block._floatBuffer.put(r.nextFloat());
			}
		}
		block.write();
		return block;
	}

	public void write() {
		try {
			File f = new File(blockPath());
			File p = f.getParentFile();
			if (!p.exists())
				p.mkdirs();
			if (!f.exists())
				f.createNewFile();
			FileOutputStream fos = new FileOutputStream(f, false);
			FileChannel fc = fos.getChannel();
			fc.write(_byteBuffer);
			fc.close();
			fos.close();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	static Block read(Matrix matrix, int blockRow, int blockCol) {
		Block block = new Block(matrix, blockRow, blockCol);

		try {
			FileInputStream fis = new FileInputStream(block.blockPath());
			FileChannel fc = fis.getChannel();
			block._byteBuffer = ByteBuffer.allocate(BLOCK_SIZE_2 * 4);
			fc.read(block._byteBuffer);
			block._byteBuffer.rewind();
			block._floatBuffer = block._byteBuffer.asFloatBuffer();
			fc.close();
			fis.close();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		return block;
	}

	public int getBlockRow() {
		return _blockRow;
	}

	public int getBlockCol() {
		return _blockCol;
	}

	public String blockPath() {
		return _matrix.matrixPath() + "/blocks/" + _blockRow + "/" + _blockCol;
	}

	public float[] getRow(int row) {
		float[] array = new float[BLOCK_SIZE];
		_floatBuffer.position(BLOCK_SIZE * row);
		_floatBuffer.get(array, 0, BLOCK_SIZE);
		return array;
	}

	public static Block multiplyCublas(Matrix matrix, Block a, Block b) {
		Block result = new Block(matrix, a._blockRow, b._blockCol);
		sgemmJCublas(BLOCK_SIZE, 1, a._byteBuffer.array(), b._byteBuffer
				.array(), 0, result._byteBuffer.array());
		return result;
	}

	public static Block multiplyJava(Matrix matrix, Block a, Block b) {
		Block result = new Block(matrix, a._blockRow, b._blockCol);
		sgemmJava(BLOCK_SIZE, 1, a._floatBuffer, b._floatBuffer, 0,
				result._floatBuffer);
		return result;
	}

	public static Block add(Matrix matrix, Block a, Block b) {
		Block result = new Block(matrix, a._blockRow, a._blockCol);
		for (int i = 0; i < BLOCK_SIZE_2; i++) {
			result._floatBuffer.put(i, a._floatBuffer.get(i)
					+ b._floatBuffer.get(i));
		}
		return result;
	}

	public static Block reduce(Matrix matrix, Block a, Block b) {
		Block result = new Block(matrix, a._blockRow, b._blockCol);
		for (int i = 0; i < BLOCK_SIZE_2; i++) {
			result._floatBuffer.put(i, a._floatBuffer.get(i)
					- b._floatBuffer.get(i));
		}
		return result;
	}

	private static void sgemmJCublas(int n, float alpha, byte A[], byte B[],
			float beta, byte C[]) {
		int nn = n * n;

		// Initialize JCublas
		JCublas.cublasInit();

		// Allocate memory on the device
		Pointer d_A = new Pointer();
		Pointer d_B = new Pointer();
		Pointer d_C = new Pointer();
		JCublas.cublasAlloc(nn, Sizeof.FLOAT, d_A);
		JCublas.cublasAlloc(nn, Sizeof.FLOAT, d_B);
		JCublas.cublasAlloc(nn, Sizeof.FLOAT, d_C);

		// Copy the memory from the host to the device
		JCublas.cublasSetVector(nn, Sizeof.FLOAT, Pointer.to(A), 1, d_A, 1);
		JCublas.cublasSetVector(nn, Sizeof.FLOAT, Pointer.to(B), 1, d_B, 1);
		JCublas.cublasSetVector(nn, Sizeof.FLOAT, Pointer.to(C), 1, d_C, 1);

		// Execute sgemm
		JCublas.cublasSgemm('n', 'n', n, n, n, alpha, d_A, n, d_B, n, beta,
				d_C, n);

		// Copy the result from the device to the host
		JCublas.cublasGetVector(nn, Sizeof.FLOAT, d_C, 1, Pointer.to(C), 1);

		// Clean up
		JCublas.cublasFree(d_A);
		JCublas.cublasFree(d_B);
		JCublas.cublasFree(d_C);

		JCublas.cublasShutdown();
	}
	
	public static void sgemmJava(int n, float alpha, FloatBuffer A,
			FloatBuffer B, float beta, FloatBuffer C) {
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				float prod = 0;
				for (int k = 0; k < n; ++k) {
					prod += A.get(k * n + i) * B.get(j * n + k);
				}
				C.put(j * n + i, alpha * prod + beta * C.get(j * n + i));
			}
		}
	}

}
