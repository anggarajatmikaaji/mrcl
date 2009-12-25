package mrcl.lib;

/**
 * Defines a matrix multiplier backend interface. 
 * This contains only one method that performs matrix multiplication.
 * 
 * All implementation class should be instantiatable with empty arguments.
 */
public interface MatrixMultiplier
{
	public static final String DEFAULT_MULTIPLIER = "Java";
	public Content doMultiplication(Block block, Content a, Content b);
}
