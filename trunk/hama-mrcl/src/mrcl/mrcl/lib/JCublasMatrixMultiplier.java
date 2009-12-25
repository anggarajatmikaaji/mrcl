package mrcl.lib;

import java.nio.FloatBuffer;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;

/**
 * Performs matrix multiplication with JCublas extension.
 */
public class JCublasMatrixMultiplier implements MatrixMultiplier
{
	public JCublasMatrixMultiplier() {
	}

	@Override
	public Content doMultiplication(Block block, Content a, Content b)
	{
		Content content = new Content(block);
		float [] aData = new float[Block.BLOCK_SIZE * Block.BLOCK_SIZE];
		float [] bData = new float[Block.BLOCK_SIZE * Block.BLOCK_SIZE];
		float [] cData = new float[Block.BLOCK_SIZE * Block.BLOCK_SIZE];
		FloatBuffer floatBufferA = a.getFloatBuffer();
		FloatBuffer floatBufferB = b.getFloatBuffer();
		floatBufferA.rewind();
		floatBufferA.get(aData);
		floatBufferB.rewind();
		floatBufferB.get(bData);
		sgemmJCublas(Block.BLOCK_SIZE, 1, aData, bData, 0, cData);
		content.getFloatBuffer().put(cData);
		return content;
	}

	public static void sgemmJCublas(int n, float alpha, float A[], float B[],
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
}
