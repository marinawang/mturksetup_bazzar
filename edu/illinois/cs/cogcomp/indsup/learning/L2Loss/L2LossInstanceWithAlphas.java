package edu.illinois.cs.cogcomp.indsup.learning.L2Loss;

import edu.illinois.cs.cogcomp.indsup.inference.AbstractStructureFinder;
import edu.illinois.cs.cogcomp.indsup.inference.IInstance;
import edu.illinois.cs.cogcomp.indsup.learning.JLISParameters;
import edu.illinois.cs.cogcomp.indsup.learning.WeightVector;

public abstract class L2LossInstanceWithAlphas {
	protected static int MAX_DCD_INNNER_ITER = 20;
	protected static double DCD_INNNER_STOP = 0.1;
	protected static double DUAL_GAP = 0.1;
	protected static double BINARY_DUAL_GAP = 0.01; //experimental results found that we should solve binary data tighter...
	protected static int verbose_level = 1;
	protected static boolean check_inf_opt = true;
	
	protected final static double UPDATE_CONDITION = 1e-12;
	public final static int INDIRECT_GLOBAL_BIAS = 0;
	

	
	protected IInstance ins = null;
	protected boolean is_binary = false;
	protected double sC = 0.0;
	protected int y = 0;

	public abstract double updateRepresentationCollection(WeightVector wv,AbstractStructureFinder s_finder)
			throws Exception;

	public void cleanCache(WeightVector wv) {

	}

	public abstract void solveSubProblemAndUpdateW(L2SolverInfo si,
			WeightVector w);

	public abstract int getMaxIdx();

	public abstract void fillWeightVector(WeightVector w);

	public abstract double getAlphaSum();

	public abstract double getLossWeightAlphaSum();

	public boolean isBinary() {
		return is_binary;
	}

	public int getY() {
		assert y == 1 || y == -1;
		return y;
	}

	public double getC() {
		return sC;
	}
	
	/**
	 * Important Hack Function: This allows setting optimization parameters using JLIS parameters
	 * @param para
	 */
	public static void setJLISParameters(JLISParameters para){
		MAX_DCD_INNNER_ITER = para.MAX_DCD_INNNER_ITER;
		DCD_INNNER_STOP = para.DCD_INNNER_STOP;		
		DUAL_GAP=para.DUAL_GAP;
		BINARY_DUAL_GAP = para.BINARY_DUAL_GAP;
		verbose_level = para.verbose_level;
		check_inf_opt = para.check_inference_opt;
		
		System.out.println("Dual GAP: " + DUAL_GAP);
		System.out.println("Binary Dual GAP: " + BINARY_DUAL_GAP);
	}
}

