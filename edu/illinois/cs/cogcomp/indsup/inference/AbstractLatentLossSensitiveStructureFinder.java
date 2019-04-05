package edu.illinois.cs.cogcomp.indsup.inference;

import edu.illinois.cs.cogcomp.indsup.learning.WeightVector;

/**
 * The function that is used for solving the Latent Structural SVM --- similar
 * to (Yu, Joachims ICML09). However, we used the square-hinge loss here so that
 * the optimization procedure is in fact quite different.
 * <p>
 * 
 * Denote x as the input, h as the hidden structure, and y as the output
 * structure. If you want to implement the this type of Latent Structural SVM,
 * you need to implement three inference procedures (If you extend this class,
 * the java compiler will ask you to implement this three methods in order to
 * prevent compilation error):
 * <p>
 * 
 * (1)
 * {@link AbstractLatentLossSensitiveStructureFinder#getBestStructure(WeightVector, IInstance)}
 * <p>
 * 
 * Given a example, find the best structure. Here, you want to solve
 * <p>
 * 
 * \max_{h,y} w^T \phi(x,h,y)
 * <p>
 * 
 * Note that different from the standard Structural SVM, your output structure
 * now contains both (y,h). This is the procedure you want to use at test time.
 * <p>
 * 
 * (2)
 * {@link AbstractLatentLossSensitiveStructureFinder#getLossSensitiveBestStructure(WeightVector, IInstance, IStructure)}
 * <p>
 * <p>
 * 
 * Given a example and the gold output structure y, solve the loss sensitive
 * procedure. Here, you want to solve
 * <p>
 * 
 * \max_{h,y} w^T \phi(x,h,y) + \delta(h,y,y*)
 * <p>
 * 
 * The procedure will be used internally by the learning algorithm.
 * <p>
 * 
 * (3)
 * {@link AbstractLatentLossSensitiveStructureFinder#getBestLatentStructure(WeightVector, IInstance, IStructure)}
 * <p>
 * 
 * Given a example and the gold output structure y*, find the best latent
 * structure with respect to this gold structure. More precisely,
 * <p>
 * 
 * \max_{h} w^T \phi(x,h,y*)
 * <p>
 * 
 * Note that you should return a IStructure with (\hat{h}, y^*) where \hat{h} is
 * the solution of the above problem. The procedure will be used internally by
 * the learning algorithm.
 * <p>
 * 
 * <tt> Make sure that you understand the differences and relations between these three procedures before using this class </tt>.
 * <p>
 * 
 * @author Ming-Wei Chang
 * 
 */
public abstract class AbstractLatentLossSensitiveStructureFinder extends
		AbstractLossSensitiveStructureFinder {

	private static final long serialVersionUID = 1L;

	/**
	 * The function that find the best "latent structure" given a gold output
	 * structure of a given example.
	 * <p>
	 * 
	 * \max_{h} w^T \phi(x,h,y*)
	 * <p>
	 * 
	 * @param weight
	 *            The weight vector
	 * @param ins
	 *            Input example
	 * @param goldStructure
	 *            The gold output structure of this example
	 * @return The best latent structure with respect to this gold structure.
	 * @throws Exception
	 */
	public abstract IStructure getBestLatentStructure(WeightVector weight,
			IInstance ins, IStructure goldStructure) throws Exception;

}
