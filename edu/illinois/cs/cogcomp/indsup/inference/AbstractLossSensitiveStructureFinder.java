package edu.illinois.cs.cogcomp.indsup.inference;

import edu.illinois.cs.cogcomp.core.datastructures.Pair;
import edu.illinois.cs.cogcomp.indsup.learning.WeightVector;

/**
 * A class that is in charge of performing inference. It performs the argmax
 * inference and the loss-sensitive inference procedure. If you want to use
 * structural SVM or JLIS, you should extend this class.
 * <p>
 * 
 * If you only want to use LCLR, you should extend the
 * {@link AbstractStructureFinder}.
 * <p>
 * 
 * If you want to use Latent Structure Structural SVM, you should extend
 * {@link AbstractLatentLossSensitiveStructureFinder}.
 * 
 * @author Ming-Wei Chang
 * 
 */
public abstract class AbstractLossSensitiveStructureFinder extends
		AbstractStructureFinder {

	private static final long serialVersionUID = 7546407873915486925L;

	/**
	 * the inference procedure of solving:
	 * <p>
	 * 
	 * \max_{y} w^T \phi(x,y) + \delta(y,y*)
	 * <p>
	 * 
	 * where the y* is the gold structure for this example. The function \delta
	 * is the distance function between two structures (y,y*).
	 * <p>
	 * 
	 * This inference procedure is used for finding the structure that violates
	 * the constraint the most.
	 * <p>
	 * 
	 * @param weight
	 *            The weight vector that is used for finding the best structure
	 * @param ins
	 *            The input example
	 * @param goldStructure
	 *            The gold structure for this example
	 * @return A {@link Pair} of {@link IStructure} and {@link Double}: The
	 *         {@link IStructure} contains the structure that violates the
	 *         constraints the most. The {@link Double} represents the distance
	 *         (delta) between the returning structure (y) and the gold
	 *         structure (y*) for this example.
	 * @throws Exception
	 */
	public abstract Pair<IStructure, Double> getLossSensitiveBestStructure(
			WeightVector weight, IInstance ins, IStructure goldStructure)
			throws Exception;
}
