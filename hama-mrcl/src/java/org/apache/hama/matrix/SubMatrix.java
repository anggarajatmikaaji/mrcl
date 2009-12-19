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

import org.apache.hadoop.hbase.util.Bytes;
import org.apache.log4j.Logger;

/**
 * A sub matrix is a matrix formed by selecting certain rows and columns from a bigger matrix. This is a in-memory
 * operation only.
 */
public class SubMatrix
{
	static final Logger LOG = Logger.getLogger(SubMatrix.class);
	private double[][] matrix;

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
		this.matrix = new double[numColumns][numRows];
	}

	/**
	 * Constructor
	 * 
	 * @param c
	 *            a two dimensional double array
	 */
	@Deprecated
	public SubMatrix(double[][] c)
	{
		double[][] matrix = c;
		this.matrix = matrix;
	}

	public SubMatrix(byte[] matrix) throws IOException
	{
		ByteArrayInputStream bos = new ByteArrayInputStream(matrix);
		DataInputStream dis = new DataInputStream(bos);

		int rows = dis.readInt();
		int columns = dis.readInt();
		this.matrix = new double[rows][columns];

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				this.matrix[j][i] = dis.readDouble();
			}
		}

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
	public void set(int row, int column, double value)
	{
		matrix[column][row] = value;
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
		matrix[column][row] = Bytes.toDouble(value);
	}

	/**
	 * Gets the value
	 * 
	 * @param row
	 * @param column
	 * @return the value of submatrix(i, j)
	 */
	public double get(int row, int column)
	{
		return matrix[column][row];
	}

	public void add(int row, int column, double value)
	{
		matrix[column][row] = matrix[column][row] + value;
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
		SubMatrix c = new SubMatrix(this.getRows(), b.getColumns());
		SubMatrix c2 = new SubMatrix(this.getRows(), b.getColumns());

		System.out.println("DEBUG: java.library.path = " + System.getenv("LD_LIBRARY_PATH") + " / "
		        + System.getProperty("java.library.path"));

		JCublas.cublasInit();
		Pointer matrixAdata = new Pointer();
		Pointer matrixBdata = new Pointer();
		Pointer matrixCdata = new Pointer();
		
		Pointer tempVector = new Pointer();
		
		JCublas.cublasAlloc(this.getRows() * this.getColumns(), Sizeof.DOUBLE, matrixAdata);
		JCublas.cublasAlloc(b.getRows() * b.getColumns(), Sizeof.DOUBLE, matrixBdata);
		JCublas.cublasAlloc(c.getRows() * c.getColumns(), Sizeof.DOUBLE, matrixCdata);

		for (int j = 0; j < this.getColumns(); j++)
//			JCublas.cublasSetMatrix(this.getRows(), this.getColumns(), Sizeof.DOUBLE, Pointer.to(this.matrix[j]), this
//			        .getRows(), matrixAdata, this.getRows());
			JCublas.cublasSetVector( this.getRows(), Sizeof.DOUBLE, Pointer.to(this.matrix[j]), 1, matrixAdata, 1 );
			
		for (int j = 0; j < b.getColumns(); j++)
			JCublas.cublasSetMatrix(b.getRows(), b.getColumns(), Sizeof.DOUBLE, Pointer.to(b.matrix[j]), b.getRows(),
			        matrixBdata, b.getRows());
		// NOTE: We don't use C again here (beta = 0.0f), so we do not initialize C.

		int m, n, k;
		m = this.getRows();
		n = b.getColumns();
		k = this.getColumns();
		JCublas.cublasDgemm('n', 'n', m, n, k, 1.0, matrixAdata, m, matrixBdata, k, 0.0, matrixCdata, m);

		for (int j = 0; j < c.getColumns(); j++)
			JCublas.cublasGetVector(c.getRows(), Sizeof.DOUBLE, matrixCdata, c.getRows(), Pointer.to(c.matrix[j]), c
			        .getRows());

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
		return this.matrix[0].length;
	}

	/**
	 * Gets the number of columns
	 * 
	 * @return the number of columns
	 */
	public int getColumns()
	{
		return this.matrix.length;
	}

	/**
	 * Close
	 */
	public void close()
	{
		matrix = null;
	}

	/**
	 * @return the 2d double array
	 */
	@Deprecated
	public double[][] getDoubleArray()
	{
		double[][] result = matrix;
		return result;
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
				dos.writeDouble(this.get(i, j));
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
