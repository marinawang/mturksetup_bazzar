package edu.illinois.cs.cogcomp.indsup.learning;

import edu.illinois.cs.cogcomp.core.datastructures.Pair;
import edu.illinois.cs.cogcomp.indsup.inference.AbstractLossSensitiveStructureFinder;
import edu.illinois.cs.cogcomp.indsup.inference.AbstractStructureFinder;

/**
 * The interface for a JLIS learner
 * @author Ming-Wei Chang
 *
 */
public interface IJLISLearner {

	/**
	 * The function for the users to call for the structured SVM
	 * 
	 * @param struct_finder
	 *            The inference solver (dynamic programming, ILP,...). Given an
	 *            input (IInstance) and a Weight vector (WeightVector), return
	 *            the best structure (AbstractStructures)
	 * @param sp
	 *            Structured Labeled Dataset
	 * @param para
	 *            parameters for JLIS
	 * @return
	 * @throws Exception
	 */
	public WeightVector trainStructuredSVM(
			final AbstractLossSensitiveStructureFinder struct_finder,
			final StructuredProblem sp, JLISParameters para) throws Exception;

	/**
	 * The function is for running LCLR with square hinge-loss.
	 * 
	 * @param init_wv
	 *            The initial weight vector for the input. Given that this
	 *            learning algorithm is not convex, a good initialization point
	 *            is important
	 * @param struct_finder
	 *            The inference solver (dynamic programming, ILP,...). Given an
	 *            input (IInstance) and a Weight vector (WeightVector), return
	 *            the best structure (AbstractStructures)
	 * @param bp
	 *            Binary labeled dataset
	 * @param para
	 *            Parameters for JLIS
	 * @return
	 * @throws Exception
	 */
	public WeightVector trainLCLR(final WeightVector init_wv,
			final AbstractStructureFinder struct_finder,
			final BinaryProblem bp, final JLISParameters para) throws Exception;

	/**
	 * Train structured SVM jointly with a binary labeled dataset. Use the
	 * weight vector of SSVM to initialize the weight for JLIS
	 * 
	 * @param struct_finder
	 *            The inference solver (dynamic programming, ILP,...). Given an
	 *            input (IInstance) and a Weight vector (WeightVector), return
	 *            the best structure (AbstractStructures)
	 * @param sp
	 *            Structured Labeled data
	 * @param bp
	 *            Binary Labeled data
	 * @param para
	 *            Parameters for JLIS
	 * @return returns a pair of weight vectors (first: SSVM, second: JLIS)
	 * @throws Exception
	 */
	public Pair<WeightVector, WeightVector> trainStructuredSVMAndJLIS(
			final AbstractLossSensitiveStructureFinder struct_finder,
			final StructuredProblem sp, final BinaryProblem bp,
			final JLISParameters para) throws Exception;
}
