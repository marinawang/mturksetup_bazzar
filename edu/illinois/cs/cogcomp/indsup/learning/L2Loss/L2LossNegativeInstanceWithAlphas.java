package edu.illinois.cs.cogcomp.indsup.learning.L2Loss;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import edu.illinois.cs.cogcomp.core.datastructures.Pair;
import edu.illinois.cs.cogcomp.indsup.inference.AbstractStructureFinder;
import edu.illinois.cs.cogcomp.indsup.inference.IInstance;
import edu.illinois.cs.cogcomp.indsup.inference.IStructure;
import edu.illinois.cs.cogcomp.indsup.learning.FeatureVector;
import edu.illinois.cs.cogcomp.indsup.learning.WeightVector;

public class L2LossNegativeInstanceWithAlphas extends L2LossInstanceWithAlphas {

	
	public List<IStructure> structure_list;
	public List<Pair<Double, FeatureVector>> alphafv_list;
	public HashMap<IStructure, Integer> visited_structure = null; // contains

	public L2LossNegativeInstanceWithAlphas(int y, IInstance ins, double C) {
		assert y == -1;
		this.y = y;
		this.ins = ins;
		
		structure_list = new ArrayList<IStructure>();
		alphafv_list = new ArrayList<Pair<Double, FeatureVector>>();

		visited_structure = new HashMap<IStructure, Integer>();
		sC = C;
		// System.out.println("C = " + C);
		is_binary = true;
	}

	@Override
	public void cleanCache(WeightVector wv) {
		// Remove cache
		HashSet<IStructure> remove_set = new HashSet<IStructure>();
		for (int i = 0; i < structure_list.size(); i++) {
			IStructure saved_h = structure_list.get(i);
			
			if (alphafv_list.get(i).getFirst() <= 1e-10) // not in
			
			{
				if (visited_structure.get(saved_h) <= 2)
					remove_set.add(saved_h);

			}
		}

		for (IStructure remove_h : remove_set) {
			for (int i = 0; i < structure_list.size(); i++) {
				if (remove_h == structure_list.get(i)) {
					structure_list.remove(i);
					alphafv_list.remove(i);
					break;
				}
			}
			// alphafv_map.remove(remove_h);
		}
	}

	@Override
	public double updateRepresentationCollection(WeightVector wv,
			AbstractStructureFinder structure_finder) throws Exception {

		IStructure abs = structure_finder.getBestStructure(wv, ins);

		if (abs == null) {
			System.out
					.println("!!!!!!!!!!!!!!!!!!!! WARNING: inference solver return null structure!! ");
			return 0;
		}

		FeatureVector bestFeatures = abs.getFeatureVector();

		// if (!alphafv_map.containsKey(abs)) {

		double xi = getAlphaSum() / (2.0 * getC());

		bestFeatures.normalize(ins.size());
		bestFeatures.slowAddFeature(INDIRECT_GLOBAL_BIAS, 1.0);

		double score = 1.0 - y * wv.dotProduct(bestFeatures) - xi;

		// double check if the code is right
		double max_score_in_cache = Double.NEGATIVE_INFINITY;
		int max_structure_i = -1;

		// System.out.println("========================================");
		// System.out.println(score);
		// System.out.println("=========================================");
		for (int i = 0; i < structure_list.size(); i++) {
			Pair<Double, FeatureVector> tmp_alpha_loss = alphafv_list.get(i);
			double s = 1.0 - y * wv.dotProduct(tmp_alpha_loss.getSecond()) - xi;

			if (max_score_in_cache < s) {
				max_score_in_cache = s;
				max_structure_i = i;
			}
		}

		if (check_inf_opt && score < max_score_in_cache - 1e-6) {
			System.out.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
			System.out.println("The inference procedure is not correct!");
			System.out.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
			System.out.println("Pred: " + abs);

			System.out.println("pred fea(normalized): " + bestFeatures);

			System.out.println("max cache: "
					+ structure_list.get(max_structure_i));

			System.out.println("max cache fea(normalized): "
					+ alphafv_list.get(max_structure_i).getSecond());

			System.out.println("xi: " + xi);
			System.out.println("bias term:" + wv.getGlobalBiasTerm());

			System.out.println("Warning: the maxing code is not right......");
			System.out.println("score: " + score);
			System.out.println("max score in cache: " + max_score_in_cache);

			String fname = "BUG_inference.bin";
			ObjectOutputStream oos = new ObjectOutputStream(
					new BufferedOutputStream(new FileOutputStream(fname)));
			oos.writeObject(wv);
			// oos.writeObject(structure_finder);
			oos.writeObject(structure_list);
			oos.writeObject(alphafv_list);
			oos.writeObject(ins);
			oos.writeObject(abs);
			oos.close();
			System.out.println("save debuging information in .... " + fname);

			throw new Exception(
					"The inference procedure is not correct! The max solution is worse than some of the solution in the cacse! If you want to use approximated inference. Check JLISParameter.check_inference_opt ");

		}

		if (score < BINARY_DUAL_GAP) // not enough contribution
			return score;

		// System.out.println(bestFeatures.toString());

		structure_list.add(abs);
		alphafv_list.add(new Pair<Double, FeatureVector>(0.0, bestFeatures));
		// alphafv_map.put(abs, new Pair<Double, FeatureVector>(0.0,
		// bestFeatures));

		if (!visited_structure.containsKey(abs))
			visited_structure.put(abs, 1);
		else
			visited_structure.put(abs, visited_structure.get(abs) + 1);

		return score;
		// } else
		// return 0;
	}

	@Override
	public int getMaxIdx() {
		int max_idx = -1;
		for (Pair<Double, FeatureVector> p : alphafv_list) {
			FeatureVector fv = p.getSecond();
			int curidx = fv.maxIdx();
			if (curidx > max_idx)
				max_idx = curidx;
		}
		return max_idx;
	}

	@Override
	public void fillWeightVector(WeightVector w) {
		for (Pair<Double, FeatureVector> p : alphafv_list) {
			double alpha = p.getFirst();

			FeatureVector fv = p.getSecond();
			w.addSparseFeatureVector(fv, y * alpha);
		}

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

			for (Pair<Double, FeatureVector> p : alphafv_list) {
				FeatureVector fv = p.getSecond();
				double dot_product = w.dotProduct(fv);
				double xij_norm2 = fv.l2NormSqure();

				double alpha = p.getFirst();

				if (alpha < 0) {
					System.out.println("alpha = " + alpha);
					System.out.flush();
					System.exit(1);
				}

				double NG = (1.0 - y * dot_product) * 2.0 * C - sum_alpha;
				NG /= (2.0 * C);

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
					w.addSparseFeatureVector(fv, y * (new_alpha - alpha));
					p.setFirst(new_alpha);
				}
			}

			stop = inner_PGmax_new - inner_PGmin_new;

			if (stop < DCD_INNNER_STOP)
				break;
		}

		// System.out.println("Spend " + i + " stop : " + (stop));
	}

	@Override
	public double getAlphaSum() {
		double sum_alpha = 0;
		for (Pair<Double, FeatureVector> p : alphafv_list) {
			sum_alpha += p.getFirst();
		}

		return sum_alpha;
	}

	@Override
	public double getLossWeightAlphaSum() {
		return getAlphaSum();
	}

}