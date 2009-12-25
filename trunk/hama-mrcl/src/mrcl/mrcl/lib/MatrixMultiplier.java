package mrcl.lib;

/**
 * Defines a matrix multiplier backend interface. 
 * This contains only one method that performs matrix multiplication.
 */
public interface MatrixMultiplier
{
	public static final String DEFAULT_MULTIPLIER = "Java";
	public Content doMultiplication(Block block, Content a, Content b);
}
