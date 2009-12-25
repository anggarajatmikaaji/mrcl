package mrcl.lib;

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
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;

/**
 * Contains the actual content of a matrix as byte/float buffer representation.
 */
public class Content implements Writable {
	private ByteBuffer _byteBuffer;
	private FloatBuffer _floatBuffer;
	private Block _block;

	public Content(Block block) {
		_block = block;
		_byteBuffer = ByteBuffer.allocate(Block.BLOCK_SIZE * Block.BLOCK_SIZE * 4);
		_byteBuffer.rewind();
		_floatBuffer = _byteBuffer.asFloatBuffer();
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
			fc.write(_byteBuffer);
			fc.close();
			fos.close();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public float[] getRow(int row) {
		float[] ret = new float[_block.getInnerCols()];
		_floatBuffer.position(Block.BLOCK_SIZE * row);
		_floatBuffer.get(ret);
		_floatBuffer.rewind();
		return ret;
	}
	
	public FloatBuffer getFloatBuffer() {
		return _floatBuffer;
	}
	
	public ByteBuffer getByteBuffer() {
		return _byteBuffer;
	}

	public static Content add(Block block, Content a, Content b) {
		Content content = new Content(block);
		int blockSizeSquared = Block.BLOCK_SIZE * Block.BLOCK_SIZE;
		for (int i = 0; i < blockSizeSquared; i++) {
			content._floatBuffer.put(i, a._floatBuffer.get(i)
					+ b._floatBuffer.get(i));
		}
		return content;
	}

	public static Content reduce(Matrix matrix, Content a, Content b) {
		Block block = new Block(matrix, a._block.getBlockRow(), a._block
				.getBlockCol());
		Content content = new Content(block);
		int blockSizeSquared = Block.BLOCK_SIZE * Block.BLOCK_SIZE;
		for (int i = 0; i < blockSizeSquared; i++) {
			content._floatBuffer.put(i, a._floatBuffer.get(i)
					- b._floatBuffer.get(i));
		}
		return content;
	}

	@Override
	public void readFields(DataInput input) throws IOException {
		input.readFully(_byteBuffer.array());
	}

	@Override
	public void write(DataOutput output) throws IOException {
		output.write(_byteBuffer.array());
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
			content._byteBuffer = ByteBuffer.allocate(Block.BLOCK_SIZE * Block.BLOCK_SIZE * 4);
			fc.read(content._byteBuffer);
			content._byteBuffer.rewind();
			content._floatBuffer = content._byteBuffer.asFloatBuffer();
			content._floatBuffer.rewind();
			fc.close();
			fis.close();
			return content;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
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