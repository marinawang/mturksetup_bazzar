package edu.illinois.cs.cogcomp.indsup.learning;

import edu.illinois.cs.cogcomp.indsup.inference.AbstractStructureFinder;
import edu.illinois.cs.cogcomp.indsup.inference.IInstance;
import edu.illinois.cs.cogcomp.indsup.inference.IStructure;
import edu.illinois.cs.cogcomp.indsup.learning.L2Loss.L2LossInstanceWithAlphas;

/**
 * The weight vector 
 * @author Ming-Wei Chang
 *
 */
public class WeightVector extends DenseVector {
	
	/**
	 * 
	 * @param n The size of the weightvector
	 */
	public WeightVector(int n) {		
		super(n);
	}
	
	/**
	 * wv = in * multipler
	 * @param in
	 * @param multiplier
	 */
	public WeightVector(double[] in, double multiplier) {		
		u =  new double[in.length];
		for(int i =0; i < in.length; i ++)
			u[i] = in[i] * multiplier;
	}
	
	public int getWeightVectorLength(){
		return super.getVectorLength();
	}
	
	/**
	 * wv = old
	 * @param old
	 * @param additional_space
	 */
	public WeightVector(WeightVector old, int additional_space) {		
		u =  new double[old.u.length + additional_space];
		System.arraycopy(old.u, 0, u, 0, old.u.length);
		extendable = old.extendable;
	}

	/**
	 * Please use the addFeatureVector function instead !
	 * @param fv
	 * @param alpha
	 */
	@Deprecated
	public synchronized void addToW(FeatureVector fv, double alpha) {		
		super.addSparseFeatureVector(fv, alpha);
	}


	public double predictLCLRBinaryScore(IStructure is, IInstance ins){
		FeatureVector fv = is.getFeatureVector();
		fv.normalize(ins.size());		
		fv.slowAddFeature(L2LossInstanceWithAlphas.INDIRECT_GLOBAL_BIAS, 1.0);
		
		return dotProduct(fv);			
	}
	
	/** 
	 * If you use LCLR or JLIS, ALWAYS use this function to get the prediction score for binary labeled examples
	 * 
	 * @param ins       testing example 
	 * @param s_finder  inference solver
	 * @return          the score of the prediction (if > 0, predict positive, OW, predict negative)
	 * @throws Exception
	 */
	 	
	public double predictLCLRBinaryScore(IInstance ins, AbstractStructureFinder s_finder) throws Exception{
		IStructure is = s_finder.getBestStructure(this, ins);
		return predictLCLRBinaryScore(is, ins);	
	}

	
	public double getGlobalBiasTerm(){
		return u[0];
	}
	
	/**
	 * should avoid using this function, currently only for liblinear
	 * @return
	 */
	public double[] getWeightArray(){
		return super.getInternalArray();
	}
}
