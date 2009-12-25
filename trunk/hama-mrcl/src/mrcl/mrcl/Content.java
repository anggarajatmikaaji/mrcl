/**
 * 
 */
package mrcl;

import java.io.DataInput;

import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;
import jcuda.runtime.JCuda;

public class Content implements Writable {
//	private ByteBuffer _byteBuffer;
	private FloatBuffer _floatBuffer;
	private Block _block;

	private Content(Block block) {
		_block = block;
//		_byteBuffer = ByteBuffer.allocate(Block.BLOCK_SIZE_2 * 4);
//		_byteBuffer.rewind();
//		_floatBuffer = _byteBuffer.asFloatBuffer();
		_floatBuffer = FloatBuffer.allocate(Block.BLOCK_SIZE_2);
		_floatBuffer.rewind();
	}

	public static Content make(Block block) {
		return new Content(block);
	}

	public void fill(float fillValue) {
		int rows = _block.getInnerRows();
		int cols = _block.getInnerCols();
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				_floatBuffer.position(row * Block.BLOCK_SIZE + col);
				_floatBuffer.put(fillValue);
			}
		}
		_floatBuffer.rewind();
	}

	public void randomize(long seed) {
		Random r = new Random(seed + Block.BLOCK_SIZE * _block.getBlockRow()
				+ _block.getBlockCol());
		int rows = _block.getInnerRows();
		int cols = _block.getInnerCols();
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				_floatBuffer.position(row * Block.BLOCK_SIZE + col);
				_floatBuffer.put(r.nextFloat());
			}
		}
		_floatBuffer.rewind();
	}

	public void writeLocal() {
		try {
			File f = new File(_block.getBlockPath());
			File p = f.getParentFile();
			if (!p.exists())
				p.mkdirs();
			System.out.println(f.getAbsolutePath());
			if (!f.exists())
				f.createNewFile();
			FileOutputStream fos = new FileOutputStream(f);
			FileChannel fc = fos.getChannel();
			// fc.position(0);
			fc.write(floatBufferToByteBuffer(_floatBuffer));
			fc.close();
			fos.close();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	private static ByteBuffer floatBufferToByteBuffer(FloatBuffer floatBuffer) {
		ByteBuffer byteBuffer = ByteBuffer.allocate(floatBuffer.limit() * 4);
		int size = floatBuffer.limit();
		for (int i = 0; i < size; i++){
			byteBuffer.putFloat(floatBuffer.get(i));
		}
		return byteBuffer;
	}

	public float[] getRow(int row) {
		float[] ret = new float[_block.getInnerCols()];
		_floatBuffer.position(Block.BLOCK_SIZE * row);
		_floatBuffer.get(ret);
		_floatBuffer.rewind();
		return ret;
	}

	public static Content multiplyCublas(Block block, Content a, Content b) {
		Content content = new Content(block);
		float [] aData = new float[Block.BLOCK_SIZE_2];
		float [] bData = new float[Block.BLOCK_SIZE_2];
		float [] cData = new float[Block.BLOCK_SIZE_2];
		a._floatBuffer.get(aData);
		b._floatBuffer.get(bData);
		sgemmJCublas(Block.BLOCK_SIZE, 1, aData, bData, 0, cData);
		content._floatBuffer.put(cData);
		return content;
	}

	public static Content multiplyJava(Block block, Content a, Content b) {
//		Block block = new Block(matrix, a._block.getBlockRow(), b._block
//				.getBlockCol());
		Content content = new Content(block);
		sgemmJava(Block.BLOCK_SIZE, 1, a._floatBuffer, b._floatBuffer, 0,
				content._floatBuffer);
		return content;
	}

	public static Content add(Block block, Content a, Content b) {
		Content content = new Content(block);
		for (int i = 0; i < Block.BLOCK_SIZE_2; i++) {
			content._floatBuffer.put(i, a._floatBuffer.get(i)
					+ b._floatBuffer.get(i));
		}
		return content;
	}

	public static Content reduce(Matrix matrix, Content a, Content b) {
		Block block = new Block(matrix, a._block.getBlockRow(), a._block
				.getBlockCol());
		Content content = new Content(block);
		for (int i = 0; i < Block.BLOCK_SIZE_2; i++) {
			content._floatBuffer.put(i, a._floatBuffer.get(i)
					- b._floatBuffer.get(i));
		}
		return content;
	}

	private static void sgemmJCublas(int n, float alpha, float A[], float B[],
			float beta, float C[]) {
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

	@Override
	public void readFields(DataInput input) throws IOException {
		byte[] bytes = new byte[Block.BLOCK_SIZE_2 * 4];
		input.readFully(bytes);
		_floatBuffer = byteBufferToFloatBuffer(ByteBuffer.wrap(bytes));
	}

	@Override
	public void write(DataOutput output) throws IOException {
		output.write(floatBufferToByteBuffer(_floatBuffer).array());
	}

	public void writeRemote(Configuration conf) {
		try {
			FileSystem fs = FileSystem.get(conf);
			Path p = new Path(_block.getBlockPath());
			if (!fs.exists(p.getParent()))
				fs.mkdirs(p.getParent());
			DataOutputStream dos = fs.create(p);
			
			write(dos);
			dos.close();
			fs.close();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public static Content readLocal(Block block) {
		try {
			Content content = new Content(block);
			FileInputStream fis = new FileInputStream(content._block
					.getBlockPath());
			FileChannel fc = fis.getChannel();
			ByteBuffer byteBuffer = ByteBuffer.allocate(Block.BLOCK_SIZE_2 * 4);
			fc.read(byteBuffer);
			byteBuffer.rewind();
			content._floatBuffer = byteBufferToFloatBuffer(byteBuffer);
			content._floatBuffer.rewind();
			fc.close();
			fis.close();
			return content;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	private static FloatBuffer byteBufferToFloatBuffer(ByteBuffer byteBuffer) {
		FloatBuffer floatBuffer = FloatBuffer.allocate(byteBuffer.limit() / 4);
		int size = floatBuffer.limit();
		for (int j = 0; j < size; j++){
			byteBuffer.position(j * 4);
			floatBuffer.put(byteBuffer.getFloat());
		}
		return floatBuffer;
	}
	
	public static Content readRemote(Block block, Configuration conf) {
		try {
			Content content = new Content(block);
			FileSystem fs;
			fs = FileSystem.get(conf);
			DataInputStream dis = fs.open(new Path(block.getBlockPath()));
			content.readFields(dis);
			dis.close();
			fs.close();
			return content;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}
}