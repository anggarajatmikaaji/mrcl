package mrcl;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;

public class Block {
	private Matrix _matrix;
	public static final int BLOCK_SIZE = 1024;
	public static final int BLOCK_SIZE_2 = BLOCK_SIZE * BLOCK_SIZE;
	private int _blockRow;
	private int _blockCol;
	private int _innerRows;
	private int _innerCols;

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
	}

	public int getBlockRow() {
		return _blockRow;
	}

	public int getBlockCol() {
		return _blockCol;
	}
	
	public int getInnerRows() {
		return _innerRows;
	}
	
	public int getInnerCols() {
		return _innerCols;
	}

	public String getBlockPath() {
		return _matrix.getMatrixPath() + "/blocks/r" + _blockRow + "/c" + _blockCol;
	}

}
