package edu.illinois.cs.cogcomp.indsup.learning;

import java.io.Serializable;

/**
 * The class that controls all of the parameters including the hyper paprameters
 * for learning, optimization control parameters, stopping conditions.
 * 
 * @author Ming-Wei Chang
 * 
 */
public class JLISParameters implements Serializable {
	public static final int VLEVEL_LOW = 0;
	public static final int VLEVEL_MID = 1;
	public static final int VLEVEL_HIGH = 2;
	/**
	 * 
	 */
	private static final long serialVersionUID = 3630883016928318230L;

	/**
	 * The regularization parameter that controls how much we want to overfit
	 * the structured labeled data. If the value is higher, the training error
	 * of the structured data should be lower.
	 * <p>
	 * 
	 * This parameter will be omitted in LCLR, because there is no structured
	 * labeled data.
	 * 
	 */
	public double c_struct = 1.0; // this parameter will be omitted in LCLR

	/**
	 * The regularization parameter that controls how much we want to overfit
	 * the binary labeled data. If the value is higher, the training error of
	 * the binary labeled data should be lower.
	 * <p>
	 * 
	 * This parameter will be omitted in training structural SVM, JLIS and
	 * Latent Structure SSVM, because there is no binary labeled data.
	 * 
	 */
	public double c_binary = 1.0; // this parameter will be omitted in train
									// SSVM

	/**
	 * How many (outter) iterations do you allow LCLR, JLIS or Latent Structure
	 * SSVM to run?
	 */
	public int MAX_OUT_ITER = 200; // how many iterations of SSVM do you

	/**
	 * The stopping condition for LCLR, JLIS or Latent Structure SSVM. Learning
	 * will stop if the number of the outter iterations exceeds
	 * {@link JLISParameters#MAX_OUT_ITER}, or this stopping conditional is
	 * achieved.
	 */
	public double OUTTER_STOP = 1e-5f;

	/**
	 * This option is currently only used to speed up the training for
	 * structured SVM (Not used in JLIS, LCLR and Latent Structure SSVM)
	 * <p>
	 * 
	 * When it is set to true, the algorithm will first find a small set to
	 * train (the size of this set is specified in
	 * {@link JLISParameters#TRAINMINI_SIZE}.
	 * 
	 */
	public boolean TRAINMINI = false;
	public int TRAINMINI_SIZE = 1000;

	/**
	 * The parameter that controls how tight do you want to apply the cutting
	 * plane method on structured data. If it is too small (< 0.01), the
	 * algorithm can take a lot of time to stop.
	 */
	public double DUAL_GAP = 0.1;
	/**
	 * The parameter that controls how tight do you want to apply the cutting
	 * plane method on binary data. If it is too small (< 0.01), the algorithm
	 * can take a lot of time to stop.
	 */
	public double BINARY_DUAL_GAP = 0.01; // experimental results found that we
											// should solve binary data
											// tighter...

	/**
	 * Verbose level: you should assign verbose level by using {@link
	 * this#VLEVEL_HIGH},{@link this#VLEVEL_MID} or {@link this#VLEVEL_LOW}.
	 * 
	 */
	public int verbose_level = VLEVEL_MID;

	// -----------------------------------------------------------------------------------
	// You should not care the options below if you do not know their meanings.
	// -----------------------------------------------------------------------------------

	/**
	 * This is an important variable. If you want to do ILP or dynamic
	 * programming and you want to check if your inference procedure is correct.
	 * JLIS will help you (learned from SVM_struct) check your procedure by
	 * compared the current solutions to the solutions of your working set.
	 * 
	 * You should turn this off only if you want to use approximated inference
	 * procedure (like beamsearch).
	 */
	public boolean check_inference_opt = true;

	/**
	 * Usually you do not need to set this variable.
	 * 
	 * Sometimes it is safer to fix the size of the weight vector (for example,
	 * all of the features are already extracted and cached before running the
	 * learning algorithm), in this case, the user can precompute on how many
	 * features we are going to use and tell the learning algorithm. By using
	 * the {@link DenseVector#setExtendable(boolean)}, we can disallow the
	 * weight vector to grow. This is a parameter that tells the weight vector
	 * how many features there are going to be in the \phi(x,y) space.
	 * <p>
	 * 
	 * This parameter lost its meaning if we extract the features on the fly,
	 * given that it need to allow the weight vector to grows its size in the
	 * learning algorithm.
	 */
	public int total_number_features = -1;

	/**
	 * Before convergence, how many iterations do you want to spend on training
	 * the SVM problem (the most inner loop). The default value is only 250,
	 * because we find that we often do not want to solve SVM too tidely at the
	 * beginning of the algorithm.
	 */
	public int MAX_SVM_ITER = 250;

	/**
	 * The stopping conditon that is close to the one used in liblienar. We use
	 * this option to control the precision of the quadratic programming solver
	 * in the most inner loop.
	 */
	public double WORKINGSETSVM_STOP = 0.1;
	/**
	 * The flag that allows to "remove" added cutting plane for the working set
	 * (if the alphas associated to them are very close to zero)
	 */
	public boolean CLEAN_CACHE = true;

	/**
	 * If the {@link JLISParameters#CLEAN_CACHE} is true, we will remove unused
	 * cutting plane every {@link JLISParameters#CLEAN_CACHE_ITER} iterations.
	 */
	public int CLEAN_CACHE_ITER = 5;

	/**
	 * If this option to true. At every iteration, the primal objective function
	 * will be printed out. Note that sometimes this can be expensive because we
	 * need to run the inference procedure for all examples again.
	 */
	public boolean CALCULATE_REAL_OBJ = false; // use to print the

	/**
	 * optimalziation parameters
	 */
	public int MAX_DCD_INNNER_ITER = 20;
	/**
	 * optimalziation parameters
	 */
	public double DCD_INNNER_STOP = 0.1;

}
