package edu.illinois.cs.cogcomp.indsup.inference;

import java.io.Serializable;

import edu.illinois.cs.cogcomp.indsup.learning.WeightVector;

/**
 * The inference procedure that is used for finding the best structure. This is
 * the place to put the code for calculating argmax. Note that you should only
 * use (extend) this class when you do not have any structured-label example. If
 * you do, you should extend the {@link AbstractLossSensitiveStructureFinder}
 * class. <p>
 * 
 * In {@link AbstractLossSensitiveStructureFinder}, you need to implement two
 * inference procedures. One for argmax (extended from this class) and another
 * one for loss-sensitive argmax procedure.
 * 
 * @author Ming-Wei Chang
 * 
 */
public abstract class AbstractStructureFinder implements Serializable {

	private static final long serialVersionUID = 1L;

	/**
	 * the inference procedure of solving: <p>
	 * 
	 * \max_{y} w^T \phi(x,y) <p>
	 * 
	 * Note that x is the input example (ins). The return is the best structure
	 * for this example.<p>
	 * 
	 * At test time, this function is usually used to predict the final structure.
	 * 
	 * @param weight
	 *            The weight vector that is used for finding the best structure
	 * @param ins
	 *            The input example
	 * @return The best structure for this input example
	 * @throws Exception
	 */
	public abstract IStructure getBestStructure(WeightVector weight,
			IInstance ins) throws Exception;

}
