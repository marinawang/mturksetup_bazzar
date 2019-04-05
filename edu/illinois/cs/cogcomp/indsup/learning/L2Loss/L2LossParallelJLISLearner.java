package edu.illinois.cs.cogcomp.indsup.learning.L2Loss;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import edu.illinois.cs.cogcomp.core.datastructures.Pair;
import edu.illinois.cs.cogcomp.indsup.inference.AbstractLatentLossSensitiveStructureFinder;
import edu.illinois.cs.cogcomp.indsup.inference.AbstractLossSensitiveStructureFinder;
import edu.illinois.cs.cogcomp.indsup.inference.AbstractStructureFinder;
import edu.illinois.cs.cogcomp.indsup.inference.IInstance;
import edu.illinois.cs.cogcomp.indsup.inference.IStructure;
import edu.illinois.cs.cogcomp.indsup.learning.BinaryProblem;
import edu.illinois.cs.cogcomp.indsup.learning.FeatureVector;
import edu.illinois.cs.cogcomp.indsup.learning.JLISParameters;
import edu.illinois.cs.cogcomp.indsup.learning.StructuredProblem;
import edu.illinois.cs.cogcomp.indsup.learning.WeightVector;

/**
 * The optimization procedure for the learning algorithms that support
 * multithread
 * 
 * @author Ming-Wei Chang
 * 
 */

public class L2LossParallelJLISLearner extends L2LossJLISLearner {
	Random rnd = new Random(0);

	class LatentStructInferenceHandler extends Thread {
		public int n_s_new = 0;
		private AbstractLatentLossSensitiveStructureFinder s_finder;
		private List<L2LossInstanceWithAlphas> alpha_ins_list;
		private WeightVector wv;
		int verbose_level;

		public LatentStructInferenceHandler(
				AbstractLatentLossSensitiveStructureFinder struct_finder,
				List<L2LossInstanceWithAlphas> subset, WeightVector wv,
				int verbose_level) {
			this.s_finder = struct_finder;
			this.alpha_ins_list = subset;
			this.wv = wv;
			this.verbose_level = verbose_level;
			if (verbose_level >= JLISParameters.VLEVEL_HIGH) {
				System.out.println("Thread: (Latent) this one will handle "
						+ subset.size() + " instances!");
			}
		}

		@Override
		public void run() {

			for (L2LossInstanceWithAlphas ins : alpha_ins_list) {
				assert ins.isBinary() == false;
				L2LossStructureInstanceWithAlphas sins = (L2LossStructureInstanceWithAlphas) ins;
				IStructure pre_gold = sins.gold_struct;
				try {
					IStructure newLatentStructureWithSameOutputStructure = s_finder
							.getBestLatentStructure(wv, sins.ins, pre_gold);
					if (!pre_gold
							.equals(newLatentStructureWithSameOutputStructure))
						n_s_new += 1;
					sins.gold_struct = newLatentStructureWithSameOutputStructure;
					FeatureVector gold_fv = sins.gold_struct.getFeatureVector();
					sins.gold_fv = gold_fv;

					// update the feature vector;
					int n_structure = sins.structure_list.size();
					for (int i = 0; i < n_structure; i++) {
						Pair<double[], FeatureVector> p = sins.al_fv_list
								.get(i);
						// assume the loss does not change!!! ==> gold structure
						// does not change,
						// hence, only need to modify the feature vector
						FeatureVector fv = FeatureVector.minus(gold_fv,
								sins.structure_list.get(i).getFeatureVector());
						p.setSecond(fv);
					}
					sins.removed_structure = new HashMap<IStructure, Integer>();

				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}

			if (verbose_level >= JLISParameters.VLEVEL_HIGH) {
				System.out.println("Thread: (b,s) udpate =  " + "  " + n_s_new);
			}
		}
	}

	class NegAndStructInferenceHandler extends Thread {
		public int n_s_new = 0;
		public int n_b_new = 0;
		private AbstractStructureFinder s_finder;
		private List<L2LossInstanceWithAlphas> alpha_ins_list;
		private WeightVector wv;
		private int verbose_level;

		public NegAndStructInferenceHandler(
				AbstractStructureFinder struct_finder,
				List<L2LossInstanceWithAlphas> subset, WeightVector wv,
				int verbose_level) {
			this.s_finder = struct_finder;
			this.alpha_ins_list = subset;
			this.wv = wv;
			this.verbose_level = verbose_level;
			if (verbose_level >= JLISParameters.VLEVEL_HIGH) {
				System.out.println("Thread: this one will handle "
						+ subset.size() + " instances!");
			}
		}

		@Override
		public void run() {
			int index = 0;

			for (L2LossInstanceWithAlphas ins : alpha_ins_list) {
				// positive h has already been fixed
				if (ins.isBinary() && ins.getY() == 1)
					continue;
				double score = 0;
				try {
					score = ins.updateRepresentationCollection(wv, s_finder);
				} catch (Exception e) {
					e.printStackTrace();
					System.exit(1);
				}

				if (ins.isBinary()) {
					if (score > L2LossInstanceWithAlphas.BINARY_DUAL_GAP) {
						n_b_new += 1;
					}
				} else {
					if (score > L2LossInstanceWithAlphas.DUAL_GAP) {
						n_s_new += 1;
					}
				}
				// System.out.println("now: " + index);
				index++;
			}
			if (verbose_level >= JLISParameters.VLEVEL_HIGH) {
				System.out.println("Thread: (b,s) udpate =  " + n_b_new + "  "
						+ n_s_new);
			}
		}
	}

	class PositiveInferenceHandler extends Thread {
		public int n_b_new = 0;
		private AbstractStructureFinder s_finder;
		private List<L2LossInstanceWithAlphas> alpha_ins_list;
		private WeightVector wv;
		private int verbose_level;

		public PositiveInferenceHandler(AbstractStructureFinder struct_finder,
				List<L2LossInstanceWithAlphas> subset, WeightVector wv,
				int verbose_level) {
			this.s_finder = struct_finder;
			this.alpha_ins_list = subset;
			this.wv = wv;
			this.verbose_level = verbose_level;
			if (verbose_level >= JLISParameters.VLEVEL_HIGH) {

				System.out.println("Thread: this one will handle "
						+ subset.size() + " instances!");
			}
		}

		@Override
		public void run() {
			int index = 0;

			for (L2LossInstanceWithAlphas ins : alpha_ins_list) {

				// focus on positive know
				if (!ins.isBinary() || ins.getY() == -1)
					continue;

				double score = 0;
				try {
					score = ins.updateRepresentationCollection(wv, s_finder);
				} catch (Exception e) {
					e.printStackTrace();
					System.exit(1);
				}

				if (score > L2LossInstanceWithAlphas.BINARY_DUAL_GAP) {
					n_b_new += 1;
				}
				// System.out.println("now: " + index);
				index++;
			}
			if (verbose_level >= JLISParameters.VLEVEL_HIGH) {
				System.out.println("Thread: (b) udpate =  " + n_b_new);
			}
		}
	}

	protected WeightVector multiThreadGetJointWeightVector(WeightVector old_wv,
			final AbstractStructureFinder[] struct_finder_list,
			StructuredProblem sp, BinaryProblem bp, JLISParameters para)
			throws Exception {

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

		return multitreadTrainJLIS(struct_finder_list, para.MAX_OUT_ITER,
				struct_size, total_size, new_wv, alpha_ins_list, para)
				.getFirst();

	}

	private Pair<WeightVector, Double> multitreadTrainJLIS(
			final AbstractStructureFinder[] struct_finder_list,
			final int OUT_ITER, int struct_size, int total_size,
			WeightVector new_wv, L2LossInstanceWithAlphas[] alpha_ins_list,
			JLISParameters para) throws Exception {
		L2LossInstanceWithAlphas.setJLISParameters(para);
		// counting n_positive
		int n_postive = 0;
		for (int i = 0; i < alpha_ins_list.length; i++)
			if (alpha_ins_list[i].isBinary() && alpha_ins_list[i].getY() == 1)
				n_postive++;

		// start training
		double out_pre_obj = Double.POSITIVE_INFINITY; // outer loop: minimize
		double out_obj = 0;
		boolean inner_finished = false;
		boolean resolved = false;

		verbose_step = (int) Math.max(1.0 * total_size / 5, 100);

		for (int k = 0; k < OUT_ITER; k++) {
			System.out.println("!!! OUTER Iteration: " + k);

			// update for binary examples
			int n_p_update = multiThreadUpdateStructuresforPositiveExamples(
					alpha_ins_list, new_wv, struct_finder_list,
					para.verbose_level);

			if (para.verbose_level >= JLISParameters.VLEVEL_MID) {
				System.out.println("Among " + n_postive + " examples, "
						+ n_p_update + "updated ");
			}

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
					Pair<Integer, Integer> count = multiThreadUpdateStructuresforNegativeAndStructuredExamples(
							alpha_ins_list, new_wv, struct_finder_list,
							para.verbose_level);
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
					if (inner_loop != 0 && n_s_new == 0 && n_b_new == 0) { // cannot
																			// start
																			// resolve
																			// at
																			// the
																			// first
																			// iteration:
																			// design
																			// for
																			// the
																			// latent
																			// Structure
																			// SVM
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
							para.WORKINGSETSVM_STOP,
							L2LossJLISLearner.MAX_RESOLVED_SVM_ITER,
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

				// remove unused alphas
				if (para.verbose_level >= JLISParameters.VLEVEL_HIGH)
					printTotalNumberofAlphas(alpha_ins_list);
				// remove unused alphas
				if (para.CLEAN_CACHE && inner_loop % para.CLEAN_CACHE_ITER == 0) {
					for (int i = 0; i < total_size; i++) {
						alpha_ins_list[i].cleanCache(new_wv);
					}
					if (para.verbose_level >= JLISParameters.VLEVEL_HIGH) {
						System.out.println("Cleaning cache....");
						printTotalNumberofAlphas(alpha_ins_list);
					}
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
										(AbstractLossSensitiveStructureFinder) struct_finder_list[0]));
			}
			if (struct_size == total_size || outer_stop < para.OUTTER_STOP) {
				System.out
						.println("Met the stopping condition; Exit Outer loop");
				break;
			}

			out_pre_obj = out_obj;

		}

		return new Pair<WeightVector, Double>(new_wv, out_obj);
	}

	private int multiThreadUpdateStructuresforPositiveExamples(
			L2LossInstanceWithAlphas[] alpha_ins_list, WeightVector new_wv,
			AbstractStructureFinder[] struct_finder_list, int verbose_level)
			throws InterruptedException {

		// initialize thread
		int n_thread = struct_finder_list.length;
		PositiveInferenceHandler[] inf_runner_list = new PositiveInferenceHandler[n_thread];
		for (int i = 0; i < n_thread; i++) {
			List<L2LossInstanceWithAlphas> subset = new ArrayList<L2LossInstanceWithAlphas>();
			for (int j = 0; j < alpha_ins_list.length; j++) {
				if (j % n_thread == i)
					subset.add(alpha_ins_list[j]);
			}
			inf_runner_list[i] = new PositiveInferenceHandler(
					struct_finder_list[i], subset, new_wv, verbose_level);
		}

		// run the thread
		for (int i = 0; i < n_thread; i++) {
			inf_runner_list[i].start();
		}

		// wait until all of them are finished
		for (int i = 0; i < n_thread; i++) {
			inf_runner_list[i].join();
		}

		// collect results
		int n_b_new = 0;

		for (int i = 0; i < n_thread; i++) {
			n_b_new += inf_runner_list[i].n_b_new;
		}

		return n_b_new;
	}

	private Pair<Integer, Integer> multiThreadUpdateStructuresforNegativeAndStructuredExamples(
			L2LossInstanceWithAlphas[] alpha_ins_list, WeightVector new_wv,
			AbstractStructureFinder[] struct_finder_list, int verbose_level)
			throws InterruptedException {

		// initialize thread
		int n_thread = struct_finder_list.length;
		NegAndStructInferenceHandler[] inf_runner_list = new NegAndStructInferenceHandler[n_thread];
		for (int i = 0; i < n_thread; i++) {
			List<L2LossInstanceWithAlphas> subset = new ArrayList<L2LossInstanceWithAlphas>();
			for (int j = 0; j < alpha_ins_list.length; j++) {
				if (j % n_thread == i)
					subset.add(alpha_ins_list[j]);
			}
			inf_runner_list[i] = new NegAndStructInferenceHandler(
					struct_finder_list[i], subset, new_wv, verbose_level);
		}

		// run the thread
		for (int i = 0; i < n_thread; i++) {
			inf_runner_list[i].start();
		}

		// wait until all of them are finished
		for (int i = 0; i < n_thread; i++) {
			inf_runner_list[i].join();
		}

		// collect results
		int n_b_new = 0;
		int n_s_new = 0;
		for (int i = 0; i < n_thread; i++) {
			n_b_new += inf_runner_list[i].n_b_new;
			n_s_new += inf_runner_list[i].n_s_new;
		}

		return new Pair<Integer, Integer>(n_s_new, n_b_new);
	}

	/**
	 * The parallel version of
	 * {@link L2LossJLISLearner#trainStructuredSVM(AbstractLossSensitiveStructureFinder, StructuredProblem, JLISParameters)}
	 * <p>
	 * 
	 * The only difference now is that we need to create more than one inference
	 * solver. The number of the inference solvers will determine how many
	 * threads we will use.
	 * 
	 * @param struct_finder_list
	 *            The list of the inference solvers. It determines how many
	 *            threads the learner will use.
	 * @param sp
	 * @param para
	 * @return
	 * @throws Exception
	 */
	public WeightVector parallelTrainStructuredSVM(
			final AbstractLossSensitiveStructureFinder[] struct_finder_list,
			final StructuredProblem sp, JLISParameters para) throws Exception {

		WeightVector wv = new WeightVector(10000);

		if (para.TRAINMINI && 5 * para.TRAINMINI_SIZE < sp.size()) {
			int t_size = para.TRAINMINI_SIZE;
			System.out.println("Train a mini sp to speed up! size = " + t_size);
			StructuredProblem minisp = new StructuredProblem();
			minisp.input_list = new ArrayList<IInstance>();
			minisp.output_list = new ArrayList<IStructure>();
			ArrayList<Integer> index_list = new ArrayList<Integer>();
			for (int i = 0; i < sp.size(); i++)
				index_list.add(i);
			Collections.shuffle(index_list, new Random(0));

			for (int i = 0; i < t_size; i++) {
				int idx = index_list.get(i);
				minisp.input_list.add(sp.input_list.get(idx));
				minisp.output_list.add(sp.output_list.get(idx));
			}

			wv = multiThreadGetJointWeightVector(wv, struct_finder_list,
					minisp, empty_b, para);
		}

		// wv.setExtendable(false);

		return multiThreadGetJointWeightVector(wv, struct_finder_list, sp,
				empty_b, para);
	}

	/**
	 * The parallel version of
	 * {@link L2LossParallelJLISLearner#trainLCLR(WeightVector, AbstractStructureFinder, BinaryProblem, JLISParameters)}
	 * .
	 * <p>
	 * 
	 * The only difference now is that we need to create more than one inference
	 * solver. The number of the inference solvers will determine how many
	 * threads we will use.
	 * <p>
	 * 
	 * Remember ALWAYS use
	 * {@link WeightVector#predictLCLRBinaryScore(IInstance, AbstractStructureFinder)}
	 * to get the prediction score for binary examples
	 * <p>
	 * 
	 * @param init_wv
	 * @param struct_finder_list
	 *            The list of the inference solvers. It determines how many
	 *            threads the learner will use.
	 * @param bp
	 * @param para
	 * @return
	 * @throws Exception
	 */
	public WeightVector parallelTrainLCLR(final WeightVector init_wv,
			final AbstractStructureFinder[] struct_finder_list,
			final BinaryProblem bp, final JLISParameters para) throws Exception {

		return multiThreadGetJointWeightVector(init_wv, struct_finder_list,
				empty_s, bp, para);
	}

	/**
	 * The parallel version of
	 * {@link L2LossParallelJLISLearner#trainStructuredSVMAndJLIS(AbstractLossSensitiveStructureFinder, StructuredProblem, BinaryProblem, JLISParameters)}
	 * <p>
	 * 
	 * The only difference now is that we need to create more than one inference
	 * solver. The number of the inference solvers will determine how many
	 * threads we will use.
	 * <p>
	 * 
	 * Remember ALWAYS use
	 * {@link WeightVector#predictLCLRBinaryScore(IInstance, AbstractStructureFinder)}
	 * to get the prediction score for binary examples
	 * 
	 * @param struct_finder_list
	 *            The list of the inference solvers. It determines how many
	 *            threads the learner will use.
	 * @param sp
	 * @param bp
	 * @param para
	 * @return
	 * @throws Exception
	 */
	public Pair<WeightVector, WeightVector> parallelTrainStructuredSVMAndJLIS(
			final AbstractLossSensitiveStructureFinder[] struct_finder_list,
			final StructuredProblem sp, final BinaryProblem bp,
			final JLISParameters para) throws Exception {

		// train an SSVM first
		WeightVector init_wv = parallelTrainStructuredSVM(struct_finder_list,
				sp, para);

		System.out.println("w size: " + init_wv.getWeightVectorLength());
		System.out.println("w extendable ?: " + init_wv.isExtendable());

		// train with binary labeled data
		WeightVector res_wv = multiThreadGetJointWeightVector(init_wv,
				struct_finder_list, sp, bp, para);

		return new Pair<WeightVector, WeightVector>(init_wv, res_wv);
	}

	public WeightVector parallelTrainLatentStructuredSVMWithInitStructures_old(
			final AbstractLatentLossSensitiveStructureFinder[] struct_finder_list,
			final StructuredProblem sp, final JLISParameters para)
			throws Exception {

		WeightVector wv = new WeightVector(para.total_number_features + 1); // +1

		for (int i = 0; i < para.MAX_OUT_ITER; i++) {
			wv = multiThreadGetJointWeightVector(wv, struct_finder_list, sp,
					empty_b, para);
			for (int j = 0; j < sp.size(); j++) {
				IStructure newLatentStructureWithSameOutputStructure = struct_finder_list[0]
						.getBestLatentStructure(wv, sp.input_list.get(j),
								sp.output_list.get(j));
				sp.output_list
						.set(j, newLatentStructureWithSameOutputStructure);
			}
		}

		return wv;
	}

	/**
	 * Learning Latent Structural SVM --- similar to (Yu, Joachims ICML09).
	 * However, we used the square-hinge loss here so that the optimization
	 * procedure is in fact quite different.
	 * <p>
	 * 
	 * !!!!!!!!!!!!!!
	 * <p>
	 * Be sure you have read the javadoc of
	 * {@link AbstractLatentLossSensitiveStructureFinder} before using this
	 * function
	 * <p>
	 * !!!!!!!!!!!!!!! 1
	 * 
	 * @param struct_finder_list
	 *            The list of the inference solvers. It determines how many
	 *            threads the learner will use.
	 * @param sp
	 *            The structured problem here is different from the one we
	 *            defined for training a regular SSVM. In fact, the structured
	 *            SVM also contains the fake "labels" for hidden variables. The
	 *            users can assign whatever they want for the "labels" of hidden
	 *            variables. (The label for the output structures still needs to
	 *            match the labeled data) We use thse "fake" labels to initialize our model.
	 * 
	 * @param para
	 * @return
	 * @throws Exception
	 */
	public WeightVector parallelTrainLatentStructuredSVMWithInitStructures(
			final AbstractLatentLossSensitiveStructureFinder[] struct_finder_list,
			final StructuredProblem sp, final JLISParameters para)
			throws Exception {
		L2LossInstanceWithAlphas.setJLISParameters(para);

		WeightVector wv = new WeightVector(para.total_number_features + 1); // +1

		int struct_size = sp.size();
		int binary_size = empty_b.size();
		int total_size = struct_size + binary_size;

		L2LossInstanceWithAlphas[] alpha_ins_list = initArrayOfInstances(sp,
				empty_b, para.c_struct, para.c_binary, struct_size, total_size);

		double out_pre_obj = Double.POSITIVE_INFINITY; // outer loop: minimize
		double out_obj = 0;

		for (int i = 0; i < para.MAX_OUT_ITER; i++) {
			System.out.println("Number of traing data: #struct: " + struct_size
					+ " #binary: " + binary_size);

			Pair<WeightVector, Double> res = multitreadTrainJLIS(
					struct_finder_list, para.MAX_OUT_ITER, struct_size,
					total_size, wv, alpha_ins_list, para);

			wv = res.getFirst();
			out_obj = res.getSecond();

			double latent_outer_stop = (out_pre_obj - out_obj)
					/ Math.abs(out_obj);
			System.out.println("latent_outter_stop: " + latent_outer_stop);

			if (latent_outer_stop < para.OUTTER_STOP) {
				if (para.verbose_level >= JLISParameters.VLEVEL_MID) {
					System.out
							.println("Met the stopping condition; Exit Latent Outer loop");
				}
				break;
			}

			out_pre_obj = out_obj;

			if (para.CALCULATE_REAL_OBJ) {
				System.out
						.println("Before doing latent variable inference (the one with gold structures fixed....)");
				System.out
						.println("primal: "
								+ getPrimalObjective(
										alpha_ins_list,
										wv,
										(AbstractLossSensitiveStructureFinder) struct_finder_list[0]));
			}

			// initialize thread
			int n_thread = struct_finder_list.length;
			LatentStructInferenceHandler[] inf_runner_list = new LatentStructInferenceHandler[n_thread];
			for (int j = 0; j < n_thread; j++) {
				List<L2LossInstanceWithAlphas> subset = new ArrayList<L2LossInstanceWithAlphas>();
				for (int k = 0; k < alpha_ins_list.length; k++) {
					if (k % n_thread == j)
						subset.add(alpha_ins_list[k]);
				}
				inf_runner_list[j] = new LatentStructInferenceHandler(
						struct_finder_list[j], subset, wv, para.verbose_level);
			}

			// run the thread
			for (int j = 0; j < n_thread; j++) {
				inf_runner_list[j].start();
			}

			// wait until all of them are finished
			for (int j = 0; j < n_thread; j++) {
				inf_runner_list[j].join();
			}

			// for(int j=0; j < sp.size(); j ++){
			// IStructure old_gold = sp.output_list.get(j);
			// IStructure newLatentStructureWithSameOutputStructure =
			// struct_finder_list[0]
			// .getBestLatentStructure(wv, sp.input_list.get(j), old_gold);
			// sp.output_list.set(j,newLatentStructureWithSameOutputStructure);
			// }
			//
			// alpha_ins_list = initArrayOfInstances(sp,
			// empty_b, para.c_struct, para.c_binary, struct_size, total_size);

			if (para.CALCULATE_REAL_OBJ) {
				System.out
						.println("After doing latent variable inference (the one with gold structures fixed....)");
				System.out
						.println("primal: "
								+ getPrimalObjective(
										alpha_ins_list,
										wv,
										(AbstractLossSensitiveStructureFinder) struct_finder_list[0]));
			}

			WeightVector new_wv = new WeightVector(wv, 0);
			new_wv.setExtendable(wv.isExtendable());
			// double[] cur_w = new double[max_n + 1];
			for (int j = 0; j < alpha_ins_list.length; j++) {
				alpha_ins_list[j].fillWeightVector(new_wv);
			}

			wv = new_wv;

		}

		return wv;
	}
}
