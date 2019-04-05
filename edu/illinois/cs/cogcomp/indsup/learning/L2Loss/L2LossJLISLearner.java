package edu.illinois.cs.cogcomp.indsup.learning.L2Loss;

import java.util.ArrayList;
import java.util.Random;

import edu.illinois.cs.cogcomp.core.datastructures.Pair;
import edu.illinois.cs.cogcomp.indsup.inference.AbstractLossSensitiveStructureFinder;
import edu.illinois.cs.cogcomp.indsup.inference.AbstractStructureFinder;
import edu.illinois.cs.cogcomp.indsup.inference.IInstance;
import edu.illinois.cs.cogcomp.indsup.inference.IStructure;
import edu.illinois.cs.cogcomp.indsup.learning.BinaryProblem;
import edu.illinois.cs.cogcomp.indsup.learning.FeatureVector;
import edu.illinois.cs.cogcomp.indsup.learning.IJLISLearner;
import edu.illinois.cs.cogcomp.indsup.learning.JLISParameters;
import edu.illinois.cs.cogcomp.indsup.learning.StructuredProblem;
import edu.illinois.cs.cogcomp.indsup.learning.WeightVector;

public class L2LossJLISLearner implements IJLISLearner {
	public static int MAX_RESOLVED_SVM_ITER = 5000;
	// Optimization Parameters: these parameters usually do not affect on the
	// learning performance. For example, the stopping criteria we
	// used here is rather loose but we can still get pretty reasonable models.
	// However, they do affect the speed of the optimization algorithm.

	public boolean CALCULATE_REAL_OBJ = false; // use to print the
												// "real objective function"
												// it is expensive!

	protected static final StructuredProblem empty_s;
	protected static final BinaryProblem empty_b;

	static {
		empty_s = new StructuredProblem();
		empty_s.input_list = new ArrayList<IInstance>();
		empty_s.output_list = new ArrayList<IStructure>();
		empty_b = new BinaryProblem();
		empty_b.input_list = new ArrayList<IInstance>();
		empty_b.output_list = new ArrayList<Integer>();
	}

	protected int verbose_step = 0;

	public static WorkingSetSVMResult getWeightVectorWithWorkingSetCDSVM(
			L2LossInstanceWithAlphas[] alpha_ins_list, boolean is_extendable,
			double WORKINGSETSVM_STOP, int MAX_SVM_ITER, int verbose_level) {
		// initialize w: w = sum alpha_i x_i
		int n_ex = alpha_ins_list.length;

		WeightVector cur_wv = getWeightVectorBySumAlpahFv(alpha_ins_list,
				is_extendable, n_ex, verbose_level);

		if (verbose_level >= JLISParameters.VLEVEL_HIGH)
			System.out.println("STOPPING criteria:" + WORKINGSETSVM_STOP);
		// coordinate descent

		int[] index = new int[n_ex];

		for (int i = 0; i < n_ex; i++)
			index[i] = i;

		int t = 0;
		Random random = new Random(0);
		boolean finished = false;

		for (t = 0; t < MAX_SVM_ITER; t++) {

			L2SolverInfo si = new L2SolverInfo();

			// shuffle the index
			for (int i = 0; i < n_ex; i++) {
				int j = i + random.nextInt(n_ex - i);
				int tmp = index[i];
				index[i] = index[j];
				index[j] = tmp;
			}

			if (t % 10 == 0 && (verbose_level >= JLISParameters.VLEVEL_MID)) {
				System.out.print(".");
				System.out.flush();
			}

			// coordinate descent
			for (int i = 0; i < n_ex; i++) {
				int idx = index[i];
				alpha_ins_list[idx].solveSubProblemAndUpdateW(si, cur_wv);
			}

			if (si.PGmax_new - si.PGmin_new <= WORKINGSETSVM_STOP) {
				finished = true;
				break;
			}
		}

		double obj = getDualObjectiveWithCurrentCuts(alpha_ins_list, cur_wv);

		obj = -obj;
		if (verbose_level >= JLISParameters.VLEVEL_MID) {
			System.out.println("In Iter " + t + " negative dual obj = " + obj);
		}
		// System.out.println("###########END############");
		// wv.printWeightVector(LexManager.getLM());
		return new WorkingSetSVMResult(cur_wv, obj, finished);
	}

	public static WeightVector getWeightVectorBySumAlpahFv(
			L2LossInstanceWithAlphas[] alpha_ins_list, boolean is_extendable,
			int n_ex, int verbose_level) {
		int max_n = -1;

		for (int i = 0; i < n_ex; i++) {
			int cur_idx = alpha_ins_list[i].getMaxIdx();
			if (cur_idx > max_n)
				max_n = cur_idx;
		}

		if (verbose_level >= JLISParameters.VLEVEL_MID)
			System.out.println("number of features: " + max_n);

		WeightVector cur_wv = new WeightVector(max_n + 1);
		cur_wv.setExtendable(is_extendable);
		// double[] cur_w = new double[max_n + 1];
		for (int i = 0; i < n_ex; i++) {
			alpha_ins_list[i].fillWeightVector(cur_wv);
		}
		return cur_wv;
	}

	protected static void printTotalNumberofAlphas(
			L2LossInstanceWithAlphas[] alpha_ins_list) {
		int n_total_alphas = 0;
		int n_ex = alpha_ins_list.length;
		for (int i = 0; i < n_ex; i++) {
			if (alpha_ins_list[i].is_binary) {
				if (alpha_ins_list[i].y == 1) {
					n_total_alphas += 1;
				} else {
					L2LossNegativeInstanceWithAlphas n_alpha = (L2LossNegativeInstanceWithAlphas) alpha_ins_list[i];
					n_total_alphas += n_alpha.alphafv_list.size();
				}
			} else {
				L2LossStructureInstanceWithAlphas s_alpha = (L2LossStructureInstanceWithAlphas) alpha_ins_list[i];
				n_total_alphas += s_alpha.al_fv_list.size();
			}
		}

		System.out.println("Number of ex: " + alpha_ins_list.length);
		System.out.println("Number of alphas: " + n_total_alphas);
	}

	protected WeightVector getJointWeightVectorFast(WeightVector old_wv,
			final AbstractStructureFinder struct_finder, StructuredProblem sp,
			BinaryProblem bp, JLISParameters para) throws Exception {
		L2LossInstanceWithAlphas.setJLISParameters(para);
		int struct_size = sp.size();
		int binary_size = bp.size();

		int total_size = struct_size + binary_size;

		System.out.println("Number of traing data: #struct: " + struct_size
				+ " #binary: " + binary_size);

		WeightVector new_wv = new WeightVector(old_wv, 0); // allocate bias term
															// for indirect
															// supervision

		L2LossInstanceWithAlphas[] alpha_ins_list = initArrayOfInstances(sp,
				bp, para.c_struct, para.c_binary, struct_size, total_size);

		if (CALCULATE_REAL_OBJ) {
			System.out
					.println("The real objective value : "
							+ getPrimalObjective(
									alpha_ins_list,
									new_wv,
									(AbstractLossSensitiveStructureFinder) struct_finder));
		}
		// start training
		double out_pre_obj = Double.POSITIVE_INFINITY; // outer loop: minimize
		double out_obj;
		boolean inner_finished = false;
		boolean resolved = false;

		verbose_step = (int) Math.max(1.0 * total_size / 5, 100);

		for (int k = 0; k < para.MAX_OUT_ITER; k++) {
			System.out.println("!!! OUTER Iteration: " + k);

			// update for binary examples
			updateStructuresForBinaryPositiveExamples(new_wv, alpha_ins_list,
					struct_finder);

			// train the inner loop
			double obj = Double.NEGATIVE_INFINITY;
			int inner_loop = 0;

			resolved = false;
			inner_finished = false;

			while (true) {

				// if we update the postive example, go to solve SVM for the
				// first time
				if (k == 0 || inner_loop > 0) {

					// update for negative and structured labeled data
					Pair<Integer, Integer> count = updateStructuresforNegativeAndStructuredExamples(
							alpha_ins_list, new_wv, struct_size, struct_finder);
					int n_s_new = count.getFirst();
					int n_b_new = count.getSecond();

					if (para.verbose_level >= JLISParameters.VLEVEL_MID) {
						System.out.println("In the inner loop " + inner_loop
								+ ": Add " + n_s_new
								+ "structures for  structured examples..");

						System.out.println("In the inner loop " + inner_loop
								+ ": Add " + n_b_new
								+ "structures for  negative examples..");
					}

					// no more update is necessary, exit the internal loop
					if (n_s_new == 0 && n_b_new == 0) {
						if (inner_finished == false)
							resolved = true;
						else {
							if (para.verbose_level >= JLISParameters.VLEVEL_MID) {
								System.out
										.println("Met the stopping condition; Exit Inner loop");
							}
							break;
						}
					}
				}

				// solve svm in daul with current "cuts"
				if (resolved) {
					WorkingSetSVMResult svm_res = getWeightVectorWithWorkingSetCDSVM(
							alpha_ins_list, new_wv.isExtendable(),
							para.WORKINGSETSVM_STOP, MAX_RESOLVED_SVM_ITER,
							para.verbose_level);
					new_wv = svm_res.wv;
					obj = svm_res.objective_vaule;
					inner_finished = true;
					if (para.verbose_level >= JLISParameters.VLEVEL_MID) {
						System.out
								.println("(Resolved) Met the stopping condition; Exit Inner loop");
					}
					break;

				} else {
					WorkingSetSVMResult svm_res = getWeightVectorWithWorkingSetCDSVM(
							alpha_ins_list, new_wv.isExtendable(),
							para.WORKINGSETSVM_STOP, para.MAX_SVM_ITER,
							para.verbose_level);
					new_wv = svm_res.wv;
					obj = svm_res.objective_vaule;
					inner_finished = svm_res.finished;
				}
				printTotalNumberofAlphas(alpha_ins_list);
				// remove unused alphas
				if (para.CLEAN_CACHE && inner_loop % para.CLEAN_CACHE_ITER == 0) {
					for (int i = 0; i < total_size; i++) {
						alpha_ins_list[i].cleanCache(new_wv);
					}
					System.out.println("Cleaning cache....");
					printTotalNumberofAlphas(alpha_ins_list);
				}

				inner_loop++;
			}

			out_obj = obj;

			double outer_stop = (out_pre_obj - out_obj) / Math.abs(out_obj);
			System.out.println("outter_stop: " + outer_stop);

			if (CALCULATE_REAL_OBJ) {
				System.out
						.println("The real objective value : "
								+ getPrimalObjective(
										alpha_ins_list,
										new_wv,
										(AbstractLossSensitiveStructureFinder) struct_finder));
			}

			if (struct_size == total_size || (outer_stop < para.OUTTER_STOP)) {
				System.out
						.println("Met the stopping condition; Exit Outer loop");
				break;
			}

			out_pre_obj = out_obj;
		}

		return new_wv;

	}

	private Pair<Integer, Integer> updateStructuresforNegativeAndStructuredExamples(
			L2LossInstanceWithAlphas[] alpha_ins_list, WeightVector new_wv,
			int struct_size, AbstractStructureFinder struct_finder)
			throws Exception {
		int n_s_new = 0;
		int n_b_new = 0;
		int total_size = alpha_ins_list.length;

		for (int i = 0; i < total_size; i++) {
			// positive h has already been fixed
			if (alpha_ins_list[i].isBinary() && alpha_ins_list[i].getY() == 1)
				continue;
			double score = alpha_ins_list[i].updateRepresentationCollection(
					new_wv, struct_finder);
			if (i < struct_size) {
				if (score > L2LossInstanceWithAlphas.DUAL_GAP) {
					n_s_new += 1;
				}
			} else {
				if (score > L2LossInstanceWithAlphas.BINARY_DUAL_GAP) {
					n_b_new += 1;
				}
			}
		}

		return new Pair<Integer, Integer>(n_s_new, n_b_new);
	}

	public static double getPrimalObjective(
			L2LossInstanceWithAlphas[] alpha_ins_list, WeightVector cur_wv,
			AbstractLossSensitiveStructureFinder s_finder) throws Exception {
		double obj = 0;

		obj += cur_wv.getTwoNormSquare() * 0.5;

		for (int i = 0; i < alpha_ins_list.length; i++) {
			L2LossInstanceWithAlphas ins = alpha_ins_list[i];
			if (ins.is_binary) {
				IStructure res = s_finder.getBestStructure(cur_wv, ins.ins);
				FeatureVector fv = res.getFeatureVector();
				fv.normalize(ins.ins.size());
				// adding a bias term for binary, important

				fv.slowAddFeature(
						L2LossInstanceWithAlphas.INDIRECT_GLOBAL_BIAS, 1.0);
				double dot_product = cur_wv.dotProduct(fv);

				double loss = 1.0 - ins.getY() * dot_product;
				if (loss > 0) {
					obj += ins.sC * loss * loss;
				}
			} else {
				L2LossStructureInstanceWithAlphas sins = (L2LossStructureInstanceWithAlphas) ins;
				Pair<IStructure, Double> res = s_finder
						.getLossSensitiveBestStructure(cur_wv, sins.ins,
								sins.gold_struct);
				double loss = res.getSecond()
						+ cur_wv.dotProduct(res.getFirst().getFeatureVector())
						- cur_wv.dotProduct(sins.gold_struct.getFeatureVector());
				// System.out.println("loss = " + loss);
				if (loss > 0) {
					obj += sins.sC * loss * loss;
				}
			}

		}
		return obj;
	}

	public static double getDualObjectiveWithCurrentCuts(
			L2LossInstanceWithAlphas[] alpha_ins_list, WeightVector cur_wv) {
		double obj = 0;

		obj += cur_wv.getTwoNormSquare() * 0.5;

		for (int i = 0; i < alpha_ins_list.length; i++) {
			L2LossInstanceWithAlphas instanceWithAlphas = alpha_ins_list[i];
			double w_sum = instanceWithAlphas.getLossWeightAlphaSum();
			double sum = instanceWithAlphas.getAlphaSum();
			double C = instanceWithAlphas.getC();
			obj -= w_sum;
			obj += (1.0 / (4.0 * C)) * sum * sum;
		}
		return obj;
	}

	protected void updateStructuresForBinaryPositiveExamples(
			WeightVector new_wv, L2LossInstanceWithAlphas[] alpha_ins_list,
			AbstractStructureFinder struct_finder) throws Exception {
		// update positive examples
		int n_p = 0;
		int n_p_changed = 0;
		int total_size = alpha_ins_list.length;

		for (int i = 0; i < total_size; i++) {
			L2LossInstanceWithAlphas p_ins = alpha_ins_list[i];
			if (i % verbose_step == 0)
				System.out.println("positive_xi inference stage: " + i + "/"
						+ total_size);

			if (p_ins.isBinary() && p_ins.getY() == 1) {
				n_p += 1;
				double updated = p_ins.updateRepresentationCollection(new_wv,
						struct_finder);
				if (updated > 0)
					n_p_changed += 1;
			}
		}

		System.out.println("Among " + n_p + " examples, " + n_p_changed
				+ "updated ");
	}

	protected L2LossInstanceWithAlphas[] initArrayOfInstances(
			StructuredProblem sp, BinaryProblem bp, final double C_structure,
			final double C_binary, int struct_size, int total_size) {
		// create the dual variables for each example
		L2LossInstanceWithAlphas[] alpha_ins_list = new L2LossInstanceWithAlphas[total_size];

		// initialization: structure
		if (sp.weight_list == null) {
			for (int i = 0; i < sp.size(); i++) {
				alpha_ins_list[i] = new L2LossStructureInstanceWithAlphas(
						sp.input_list.get(i), sp.output_list.get(i),
						C_structure);
			}
		} else {
			for (int i = 0; i < sp.size(); i++) {
				alpha_ins_list[i] = new L2LossStructureInstanceWithAlphas(
						sp.input_list.get(i), sp.output_list.get(i),
						C_structure * sp.weight_list.get(i));
			}
		}

		// initialization: binary
		if (bp.weight_list == null) {
			for (int i = 0; i < bp.size(); i++)
				if (bp.output_list.get(i) == 1)
					alpha_ins_list[i + struct_size] = new L2LossPositiveInstanceWithAlphas(
							bp.output_list.get(i), bp.input_list.get(i),
							C_binary);
				else
					alpha_ins_list[i + struct_size] = new L2LossNegativeInstanceWithAlphas(
							bp.output_list.get(i), bp.input_list.get(i),
							C_binary);
		} else {
			for (int i = 0; i < bp.size(); i++) {
				if (bp.output_list.get(i) == 1)
					alpha_ins_list[i + struct_size] = new L2LossPositiveInstanceWithAlphas(
							bp.output_list.get(i), bp.input_list.get(i),
							C_binary * bp.weight_list.get(i));
				else
					alpha_ins_list[i + struct_size] = new L2LossNegativeInstanceWithAlphas(
							bp.output_list.get(i), bp.input_list.get(i),
							C_binary * bp.weight_list.get(i));
			}
		}
		return alpha_ins_list;
	}

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
	@Override
	public WeightVector trainStructuredSVM(
			final AbstractLossSensitiveStructureFinder struct_finder,
			final StructuredProblem sp, JLISParameters para) throws Exception {

		WeightVector wv = new WeightVector(para.total_number_features + 1); // +1
																			// because
																			// we
																			// skip
																			// wv.u[0]
		//wv.setExtendable(false);
		return getJointWeightVectorFast(wv, struct_finder, sp, empty_b, para);
	}

	/**
	 * The function is for running LCLR with square hinge-loss.
	 * <p>
	 * 
	 * Remember ALWAYS use
	 * {@link WeightVector#predictLCLRBinaryScore(IInstance, AbstractStructureFinder)}
	 * to get the prediction score for binary examples
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
	@Override
	public WeightVector trainLCLR(final WeightVector init_wv,
			final AbstractStructureFinder struct_finder,
			final BinaryProblem bp, final JLISParameters para) throws Exception {

		return getJointWeightVectorFast(init_wv, struct_finder, empty_s, bp,
				para);
	}

	/**
	 * 
	 * Train structured SVM jointly with a binary labeled dataset. Use the
	 * weight vector of SSVM to initalize the weight for JLIS
	 * <p>
	 * 
	 * Remember ALWAYS use
	 * {@link WeightVector#predictLCLRBinaryScore(IInstance, AbstractStructureFinder)}
	 * to get the prediction score for binary examples
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
	@Override
	public Pair<WeightVector, WeightVector> trainStructuredSVMAndJLIS(
			final AbstractLossSensitiveStructureFinder struct_finder,
			final StructuredProblem sp, final BinaryProblem bp,
			final JLISParameters para) throws Exception {

		// train an SSVM first
		WeightVector init_wv = trainStructuredSVM(struct_finder, sp, para);

		System.out.println("w size: " + init_wv.getWeightVectorLength());
		System.out.println("w extendable ?: " + init_wv.isExtendable());

		// train with binary labeled data
		WeightVector res_wv = getJointWeightVectorFast(init_wv, struct_finder,
				sp, bp, para);

		return new Pair<WeightVector, WeightVector>(init_wv, res_wv);
	}
}
