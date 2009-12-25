package mrcl.lib;

import java.nio.FloatBuffer;

/**
 * Performs matrix multiplication with pure Java code.
 */
public class JavaMatrixMultiplier implements MatrixMultiplier
{
	public JavaMatrixMultiplier() {
	}

	@Override
	public Content doMultiplication(Block block, Content a, Content b)
	{
		Content content = new Content(block);
		sgemmJava(Block.BLOCK_SIZE, 1, a.getFloatBuffer(), b.getFloatBuffer(), 0,
				content.getFloatBuffer());
		return content;
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
