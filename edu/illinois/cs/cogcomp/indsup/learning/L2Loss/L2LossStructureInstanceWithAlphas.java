package edu.illinois.cs.cogcomp.indsup.learning.L2Loss;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import edu.illinois.cs.cogcomp.core.datastructures.Pair;
import edu.illinois.cs.cogcomp.indsup.inference.AbstractLossSensitiveStructureFinder;
import edu.illinois.cs.cogcomp.indsup.inference.AbstractStructureFinder;
import edu.illinois.cs.cogcomp.indsup.inference.IInstance;
import edu.illinois.cs.cogcomp.indsup.inference.IStructure;
import edu.illinois.cs.cogcomp.indsup.learning.FeatureVector;
import edu.illinois.cs.cogcomp.indsup.learning.WeightVector;

public class L2LossStructureInstanceWithAlphas extends L2LossInstanceWithAlphas {

	public IStructure gold_struct;
	public FeatureVector gold_fv;
	// each pair contains first double: alpha, second double: loss !!!
	// public Map<IStructure, Pair<double[], FeatureVector>> alphafv_map = null;
	public List<IStructure> structure_list;
	public List<Pair<double[], FeatureVector>> al_fv_list;
	public Map<IStructure, Integer> removed_structure = null; // contains

	public L2LossStructureInstanceWithAlphas(IInstance ins,
			IStructure goldStruct, double C) {
		this.gold_struct = goldStruct;
		this.gold_fv = goldStruct.getFeatureVector();
		this.ins = ins;
		// alphafv_map = new TreeMap<IStructure, Pair<double[],
		// FeatureVector>>();
		structure_list = new ArrayList<IStructure>();
		al_fv_list = new ArrayList<Pair<double[], FeatureVector>>();
		removed_structure = new HashMap<IStructure, Integer>();

		is_binary = false;
		sC = C;
	}

	@Override
	public void fillWeightVector(WeightVector w) {
		// for (Pair<double[], FeatureVector> p : alphafv_map.values()) {
		for (Pair<double[], FeatureVector> p : al_fv_list) {
			double alpha = p.getFirst()[0];
			FeatureVector fv = p.getSecond();			
			w.addSparseFeatureVector(fv, alpha);
		}
	}

	@Override
	public double getAlphaSum() {
		double sum_alpha = 0f;
		for (Pair<double[], FeatureVector> p : al_fv_list) {
			sum_alpha += p.getFirst()[0];
		}
		return sum_alpha;
	}

	@Override
	public int getMaxIdx() {
		int max_idx = -1;
		for (Pair<double[], FeatureVector> p : al_fv_list) {
			FeatureVector fv = p.getSecond();
			int curidx = fv.maxIdx();
			if (curidx > max_idx)
				max_idx = curidx;
		}
		return max_idx;
	}

	@Override
	public void solveSubProblemAndUpdateW(L2SolverInfo si, WeightVector w) {

		double C = sC;
		double sum_alpha = getAlphaSum();
		// solve subproblem for EACH representation

		int i = 0;
		double stop;
		for (i = 0; i < MAX_DCD_INNNER_ITER; i++) {
			double inner_PGmax_new = Double.NEGATIVE_INFINITY;
			double inner_PGmin_new = Double.POSITIVE_INFINITY;
			// for (IStructure h : alphafv_map.keySet()) {
			for (int j = 0; j < al_fv_list.size(); j++) {
				// IStructure h = structure_list.get(j);

				// Pair<double[], FeatureVector> p = alphafv_map.get(h);
				Pair<double[], FeatureVector> p = al_fv_list.get(j);

				double alpha = p.getFirst()[0];
				double loss = p.getFirst()[1];

				FeatureVector fv = p.getSecond();
				double dot_product = w.dotProduct(fv);
				double xij_norm2 = fv.l2NormSqure();

				double NG = (loss - dot_product) - sum_alpha / (2.0 * C);

				double PG = -NG;
				if (alpha == 0)
					PG = Math.min(-NG, 0);

				si.PGmax_new = Math.max(si.PGmax_new, PG);
				si.PGmin_new = Math.min(si.PGmin_new, PG);

				inner_PGmax_new = Math.max(inner_PGmax_new, PG);
				inner_PGmin_new = Math.min(inner_PGmin_new, PG);

				if (Math.abs(PG) > UPDATE_CONDITION) {
					double step = NG / (xij_norm2 + 1.0 / (2.0 * C));
					double new_alpha = Math.max(alpha + step, 0);
					sum_alpha += (new_alpha - alpha);
					w.addSparseFeatureVector(fv, (new_alpha - alpha));
					double[] alpha_loss = p.getFirst();
					alpha_loss[0] = new_alpha;
					p.setFirst(alpha_loss);
				}
			}

			stop = inner_PGmax_new - inner_PGmin_new;

			if (stop < DCD_INNNER_STOP)
				break;

		}
	}

	@Override
	public void cleanCache(WeightVector wv) {

		// Remove cache
		HashSet<IStructure> remove_set = new HashSet<IStructure>();
		for (int i = 0; i < structure_list.size(); i++) {
			IStructure saved_h = structure_list.get(i);
			// if (alphafv_map.get(saved_h).getFirst()[0] <= 1e-10) // not in
			if (al_fv_list.get(i).getFirst()[0] <= 1e-10) // not in
			// the
			// cut
			{
				// if (visited_structure.get(saved_h) <= 2)
				remove_set.add(saved_h);

				if (!removed_structure.containsKey(saved_h)) { // if I have not
																// removed this
																// structure,
																// remove it!
					removed_structure.put(saved_h, 1);
					remove_set.add(saved_h);
				} else
					// if I have removed this structure but somehow it get back,
					// then keep it
					removed_structure.put(saved_h,
							removed_structure.get(saved_h) + 1);
			}
		}

		for (IStructure remove_h : remove_set) {
			for (int i = 0; i < structure_list.size(); i++) {
				if (remove_h == structure_list.get(i)) {
					structure_list.remove(i);
					al_fv_list.remove(i);
					break;
				}
			}
			// alphafv_map.remove(remove_h);
		}
	}

	@Override
	public double updateRepresentationCollection(WeightVector wv,
			AbstractStructureFinder structure_finder) throws Exception {

		double C = sC;

		Pair<IStructure, Double> inf_res = ((AbstractLossSensitiveStructureFinder) structure_finder)
				.getLossSensitiveBestStructure(wv, ins, gold_struct);

		double loss = inf_res.getSecond();
		IStructure h = inf_res.getFirst();

		FeatureVector best_features = h.getFeatureVector();

		// implement a looser condition
		// System.out.println("cache size:" + alphafv_map.size());
		// if (!alphafv_map.containsKey(h)) {
		FeatureVector diff = FeatureVector.minus(gold_fv, best_features);

		double xi = getAlphaSum() / (2.0 * C);
		double dotProduct = wv.dotProduct(diff);
		double score = (loss - dotProduct) - xi;

		// double check if the code is right
		double max_score_in_cache = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < structure_list.size(); i++) {
			Pair<double[], FeatureVector> tmp_alpha_loss = al_fv_list.get(i);
			// IStructure f= structure_list.get(i);

			double s = tmp_alpha_loss.getFirst()[1]
					- wv.dotProduct(tmp_alpha_loss.getSecond()) - xi;
			if (max_score_in_cache < s)
				max_score_in_cache = s;
		}

		if (check_inf_opt && score < max_score_in_cache - 1e-20) {
			System.out.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
			System.out.println("The inference procedure is not correct!");
			System.out.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
			
			System.out.println("Pred: " + h);
			System.out.println("docproduct on pred: "
					+ wv.dotProduct(h.getFeatureVector()));
			System.out.println("dotproduct on diff: " + dotProduct);
			System.out.println("loss: " + loss);
			System.out.println("xi: " + xi);
			System.out.println("Warning: the inference (argmax) code is not right......");
			System.out.println("score: " + score);
			System.out.println("max score in cache: " + max_score_in_cache);

			for (int i = 0; i < structure_list.size(); i++) {
				IStructure f = structure_list.get(i);
				Pair<double[], FeatureVector> pair = al_fv_list.get(i);
				System.out.println(">>>" + f + " alpha: " + pair.getFirst()[0]
						+ " loss: " + pair.getFirst()[1]);
				System.out.println(">>> (dot) "
						+ wv.dotProduct(pair.getSecond()));
			}
			System.out.println("[GOLD]" + gold_struct);
			System.out.println("gold dot product: "
					+ wv.dotProduct(gold_struct.getFeatureVector()));
			throw new Exception(
			"The inference procedure is not correct! The max solution is worse than some of the solution in the cacse! If you want to use approximated inference. Check JLISParameter.check_inference_opt ");
	}

		if (score < DUAL_GAP) // not enough contribution
			return score;

		double[] alpha_loss = new double[2];
		alpha_loss[0] = 0.0;
		alpha_loss[1] = loss;
		structure_list.add(h);
		al_fv_list.add(new Pair<double[], FeatureVector>(alpha_loss, diff));


		return score;
		

	}

	@Override
	public double getLossWeightAlphaSum() {

		double sum_alpha = 0f;
		for (Pair<double[], FeatureVector> p : al_fv_list) {
			double[] alpha_loss = p.getFirst();
			sum_alpha += alpha_loss[0] * alpha_loss[1];
		}
		return sum_alpha;

	}

}