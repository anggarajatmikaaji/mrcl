/**
 * Copyright 2007 The Apache Software Foundation
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.hama.matrix;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;
import jcuda.jcublas.cublasStatus;

import org.apache.hadoop.hbase.util.Bytes;
import org.apache.log4j.Logger;

/**
 * A sub matrix is a matrix formed by selecting certain rows and columns from a bigger matrix. This is a in-memory
 * operation only.
 */
public class SubMatrix
{
	static final Logger LOG = Logger.getLogger(SubMatrix.class);
	private float[] matrix;
	private int numColumns, numRows;

	/**
	 * Constructor
	 * 
	 * @param numRows
	 *            the size of rows
	 * @param numColumns
	 *            the size of columns
	 */
	public SubMatrix(int numRows, int numColumns)
	{
		this.matrix = new float[numColumns * numRows];
		this.numColumns = numColumns;
		this.numRows = numRows;
	}

	/**
	 * Constructor
	 * 
	 * @param c
	 *            a two dimensional float array
	 */
	@Deprecated
	public SubMatrix(float[][] c)
	{
		float[][] matrix = c;
		//this.matrix = matrix;
	}

	public SubMatrix(byte[] matrix) throws IOException
	{
		ByteArrayInputStream bos = new ByteArrayInputStream(matrix);
		DataInputStream dis = new DataInputStream(bos);

		int rows = dis.readInt();
		int columns = dis.readInt();
		this.matrix = new float[rows * columns];

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				this.matrix[j * columns + i] = dis.readFloat();
			}
		}
		this.numColumns = rows;
		this.numRows = columns;

		dis.close();
		bos.close();
	}

	/**
	 * Sets the value
	 * 
	 * @param row
	 * @param column
	 * @param value
	 */
	public void set(int row, int column, float value)
	{
		matrix[column * numColumns + row] = value;
	}

	/**
	 * Sets the value
	 * 
	 * @param row
	 * @param column
	 * @param value
	 */
	public void set(int row, int column, byte[] value)
	{
		matrix[column * numColumns + row] = Bytes.toFloat(value);
	}

	/**
	 * Gets the value
	 * 
	 * @param row
	 * @param column
	 * @return the value of submatrix(i, j)
	 */
	public float get(int row, int column)
	{
		return matrix[column * numColumns + row];
	}

	public void add(int row, int column, float value)
	{
		matrix[column * numColumns + row] = matrix[column * numColumns + row] + value;
	}

	/**
	 * c = a+b
	 * 
	 * @param b
	 * @return c
	 */
	public SubMatrix add(SubMatrix b)
	{
		SubMatrix c = new SubMatrix(this.getRows(), this.getColumns());

		for (int i = 0; i < this.getRows(); i++) {
			for (int j = 0; j < this.getColumns(); j++) {
				c.set(i, j, (this.get(i, j) + b.get(i, j)));
			}
		}

		return c;
	}

	/**
	 * c = a*b
	 * 
	 * @param b
	 * @return c
	 */
	public SubMatrix mult(SubMatrix b)
	{
		float alpha = 1.0f;
		float beta = 0.0f;
		
		SubMatrix c = new SubMatrix(this.getRows(), b.getColumns());
		SubMatrix c2 = new SubMatrix(this.getRows(), b.getColumns());
		
		SubMatrix temp = new SubMatrix(this.getRows(),this.getColumns());
		SubMatrix temp2 = new SubMatrix(b.getRows(), b.getColumns());
		
		int ret = JCublas.cublasInit();
		if (ret != cublasStatus.CUBLAS_STATUS_SUCCESS) {
			System.out.println("cublasInit ERROR : " + ret);
		}
		Pointer matrixAdata = new Pointer();
		Pointer matrixBdata = new Pointer();
		Pointer matrixCdata = new Pointer();
		ret = JCublas.cublasAlloc(this.getRows() * this.getColumns(), Sizeof.FLOAT, matrixAdata);
		if (ret != cublasStatus.CUBLAS_STATUS_SUCCESS) {
			System.out.println("cublasAlloc ERROR : " + ret);
		}
		JCublas.cublasAlloc(b.getRows() * b.getColumns(), Sizeof.FLOAT, matrixBdata);
		JCublas.cublasAlloc(c.getRows() * c.getColumns(), Sizeof.FLOAT, matrixCdata);

		ret = JCublas.cublasSetVector(this.getRows() * this.getColumns(), Sizeof.FLOAT, Pointer.to(this.matrix), 1, matrixAdata, 1);
		if (ret != cublasStatus.CUBLAS_STATUS_SUCCESS) {
			System.out.println("cublasSetVector ERROR : " + ret);
		}
		
		JCublas.cublasGetVector(this.getRows() * this.getColumns(), Sizeof.FLOAT, matrixAdata, 1, Pointer.to(temp.matrix), 1);		
		for (int i = 0; i < this.getRows(); i++) {
			for (int j = 0; j < this.getColumns(); j++)
				if (temp.get(i, j) != this.get(i, j)) {
					System.out.println(String.format("DEBUG: matrixA setVector error [%d, %d] = %.09f / %.09f", i, j, temp.get(i, j), this.get(i, j)));
					break;
				}
		}
		
		JCublas.cublasSetVector(b.getRows() * b.getColumns(), Sizeof.FLOAT, Pointer.to(b.matrix), 1, matrixBdata, 1 );
	
		JCublas.cublasGetVector(b.getRows() * b.getColumns(), Sizeof.FLOAT, matrixBdata, 1, Pointer.to(temp2.matrix), 1);		
		for (int i = 0; i < b.getRows(); i++) {
			for (int j = 0; j < b.getColumns(); j++)
				if (temp2.get(i, j) != b.get(i, j)) {
					System.out.println(String.format("DEBUG: matrixB setVector error [%d, %d] = %.09f / %.09f", i, j, temp2.get(i, j), b.get(i, j)));
					break;
				}
		}
		
		JCublas.cublasSetVector(c.getRows() * c.getColumns(), Sizeof.FLOAT, Pointer.to(c.matrix), 1, matrixCdata, 1 );
		// NOTE: We don't use C again here (beta = 0.0f), so we do not initialize C.

		int m, n, k;
		m = this.getRows();
		n = b.getColumns();
		k = this.getColumns();
		JCublas.cublasSgemm('n', 'n', m, n, k, alpha, matrixAdata, m, matrixBdata, k, beta, matrixCdata, m);

		JCublas.cublasGetVector(c.getRows() * c.getColumns(), Sizeof.FLOAT, matrixCdata, 1, Pointer.to(c.matrix), 1);

		JCublas.cublasFree(matrixAdata);
		JCublas.cublasFree(matrixBdata);
		JCublas.cublasFree(matrixCdata);
		JCublas.cublasShutdown();

		for (int i = 0; i < this.getRows(); i++) {
			for (int j = 0; j < b.getColumns(); j++) {
				for (k = 0; k < this.getColumns(); k++) {
					c2.add(i, j, this.get(i, k) * b.get(k, j));
				}
			}
		}

		for (int i = 0; i < this.getRows(); i++) {
			for (int j = 0; j < b.getColumns(); j++)
				if (c2.get(i, j) != c.get(i, j)) {
					System.out.println(String.format("DEBUG: matrix multiplication error [%d, %d] = %.09f / %.09f", i, j, c.get(i, j), c2.get(i, j)));
					break;
				}
		}
		return c;
	}

	/**
	 * Gets the number of rows
	 * 
	 * @return the number of rows
	 */
	public int getRows()
	{
		//return this.matrix[0].length;
		return numRows;
	}

	/**
	 * Gets the number of columns
	 * 
	 * @return the number of columns
	 */
	public int getColumns()
	{
		//return this.matrix.length;
		return numColumns;
	}

	/**
	 * Close
	 */
	public void close()
	{
		matrix = null;
	}

	/**
	 * @return the 2d float array
	 */
	@Deprecated
	public float[][] getFloatArray()
	{
		//float[][] result = matrix;
		return null;
	}

	/**
	 * Gets the bytes of the sub matrix
	 * 
	 * @return the bytes of the sub matrix
	 * @throws IOException
	 */
	public byte[] getBytes() throws IOException
	{
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		DataOutputStream dos = new DataOutputStream(bos);

		dos.writeInt(this.getRows());
		dos.writeInt(this.getColumns());

		for (int i = 0; i < this.getRows(); i++) {
			for (int j = 0; j < this.getColumns(); j++) {
				dos.writeFloat(this.get(i, j));
			}
		}

		byte[] data = bos.toByteArray();
		dos.close();
		bos.close();
		return data;
	}

	public String toString()
	{
		StringBuilder result = new StringBuilder();
		for (int i = 0; i < this.getRows(); i++) {
			for (int j = 0; j < this.getColumns(); j++) {
				result.append(this.get(i, j));
				result.append('\t');
			}
			result.append('\n');
		}
		return result.toString();
	}
}
