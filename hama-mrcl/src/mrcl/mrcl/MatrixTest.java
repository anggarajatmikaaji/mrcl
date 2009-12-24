package mrcl;


import java.nio.FloatBuffer;
import java.util.Arrays;

import org.junit.Assert;
import org.junit.Test;

public class MatrixTest {
	@Test
	public void testSome() {
		int n = 100;
		Matrix a = Matrix.createRandom("a", n, n, 2);
		Matrix b = Matrix.createRandom("b", n, n, 3);
		
		FloatBuffer result = FloatBuffer.allocate(a.getRows() * b.getCols());
		Content.sgemmJava(n, 1, a.getFloatBuffer(), b.getFloatBuffer(), 0, result);
		
		Matrix c = Matrix.multiplyLocal("c", a, b);
		FloatBuffer cData = c.getFloatBuffer();
		int size = cData.limit();
		
		for (int i = 0; i < size; i++){
			Assert.assertEquals(cData.get(i), result.get(i), 0.0001); 
		}
		
	}

}
