package mrcl;

import java.nio.FloatBuffer;

import mrcl.lib.JavaMatrixMultiplier;
import mrcl.lib.Matrix;

public class MatrixTest {
	public void testSome() {
		int n = 100;
		Matrix a = Matrix.createRandomLocal("a", n, n, 2);
		Matrix b = Matrix.createRandomLocal("b", n, n, 3);
		
		FloatBuffer result = FloatBuffer.allocate(a.getRows() * b.getCols());
		JavaMatrixMultiplier.sgemmJava(n, 1, a.getFloatBufferLocal(), b.getFloatBufferLocal(), 0, result);
		
		Matrix c = Matrix.multiplyLocal("c", a, b);
		FloatBuffer cData = c.getFloatBufferLocal();
		int size = cData.limit();
		
		for (int i = 0; i < size; i++){
//			Assert.assertEquals(cData.get(i), result.get(i), 0.0001); 
		}
		
	}

}
