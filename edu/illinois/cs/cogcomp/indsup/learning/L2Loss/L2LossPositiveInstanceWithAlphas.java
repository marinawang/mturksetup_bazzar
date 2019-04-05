package edu.illinois.cs.cogcomp.indsup.learning.L2Loss;

import edu.illinois.cs.cogcomp.indsup.inference.AbstractStructureFinder;
import edu.illinois.cs.cogcomp.indsup.inference.IInstance;
import edu.illinois.cs.cogcomp.indsup.inference.IStructure;
import edu.illinois.cs.cogcomp.indsup.learning.FeatureVector;
import edu.illinois.cs.cogcomp.indsup.learning.WeightVector;

public class L2LossPositiveInstanceWithAlphas extends L2LossInstanceWithAlphas {

	// for positive example
	public Double alpha = 0.0;
	public FeatureVector fv = null;
	public IStructure ahs = null;
	

	public L2LossPositiveInstanceWithAlphas(int y, IInstance ins, double C) {
		assert y == 1;
		this.y = y;
		this.ins = ins;
		alpha = 0.0;
		fv = null;
		ahs = null;
		sC = C;
		
		is_binary = true;
	}

	@Override
	public void cleanCache(WeightVector wv) {
	}

	@Override
	public double updateRepresentationCollection(WeightVector wv,AbstractStructureFinder struct_finder)
			throws Exception {

		IStructure abs = struct_finder.getBestStructure(wv,
				ins);
		FeatureVector bestFeatures = abs.getFeatureVector();

		if (ahs != null && abs.equals(ahs))
			return -1;

//		System.out.println("New Structures is ====================>");
//		System.out.println(abs);
//		
		fv = bestFeatures;
		fv.normalize(ins.size());
//		// adding a bias term for binary, important	
		fv.slowAddFeature(INDIRECT_GLOBAL_BIAS, 1.0);
		
		ahs = abs;

		return Double.POSITIVE_INFINITY;
	}

	@Override
	public int getMaxIdx() {
		return fv.maxIdx();
	}

	@Override
	public void fillWeightVector(WeightVector w) {
		w.addSparseFeatureVector(fv, y * alpha);
	}

	@Override
	public void solveSubProblemAndUpdateW(L2SolverInfo si, WeightVector w) {
		double C = sC;

		double dot_product = w.dotProduct(fv);
		double xij_norm2 = fv.l2NormSqure();

		double NG = 1.0 - y * dot_product - (alpha / (2.0 * C));

		double PG = -NG;
		if (alpha == 0f) {
			PG = Math.min(-NG, 0);
		}
		si.PGmax_new = Math.max(si.PGmax_new, PG);
		si.PGmin_new = Math.min(si.PGmin_new, PG);

		if (Math.abs(PG) > UPDATE_CONDITION) {
			double step = NG / (xij_norm2 + (1.0 / (2.0 * C)));
			double new_alpha = Math.max(alpha + step, 0);// make sure
			// alpha_[i][j]
			// is

			w.addSparseFeatureVector(fv, (new_alpha - alpha) * y);
			alpha = new_alpha;
		}

	}

	@Override
	public double getAlphaSum() {
		return alpha;
	}

	@Override
	public double getLossWeightAlphaSum() {

		return getAlphaSum();
	}

}