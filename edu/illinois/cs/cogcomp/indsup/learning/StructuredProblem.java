package edu.illinois.cs.cogcomp.indsup.learning;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import edu.illinois.cs.cogcomp.core.datastructures.Pair;
import edu.illinois.cs.cogcomp.indsup.inference.IInstance;
import edu.illinois.cs.cogcomp.indsup.inference.IStructure;

/**
 * This class represents a structured problem. A structured problem is a
 * collection of the (input-structure) pairs.
 * 
 * @author Ming-Wei Chang
 * 
 */
public class StructuredProblem {
	/**
	 * The output list that contains the corresponding gold output structures
	 * (y) for the input examples in the input_list (x)
	 */
	public List<IStructure> output_list;
	/**
	 * The input list contains the input examples (x)
	 */
	public List<IInstance> input_list;
	/**
	 * The weight list. Our JLIS implementation allows using different values of
	 * C for different examples! If this list is null. It means that every
	 * example should be treated equally. If it is not null, it should have the
	 * same number of elements as that of input/output_list.
	 * <p>
	 * 
	 * More precisely, this weight list changes the formation to the following
	 * one:
	 * <p>
	 * 
	 * \min 1/2 w*w + \sum_i C * weight_i * Loss(w,x,y)
	 * <p>
	 * 
	 * Therefore, if you put more weight on an example, it means that this
	 * example is more important and the learning algorithm will try harder to
	 * fit this example.
	 */
	public List<Double> weight_list = null;

	public StructuredProblem() {
		output_list = new ArrayList<IStructure>();
		input_list = new ArrayList<IInstance>();
	}

	/**
	 * @return the number of instances of this structured problem.
	 */
	public int size() {
		assert output_list.size() == input_list.size();
		assert weight_list == null
				|| (output_list.size() == weight_list.size());

		return output_list.size();
	}

	/**
	 * A helper function that shuffles the order of the examples in this
	 * problem.
	 * 
	 * @param rnd
	 *            A random number generator ---- if you use the same random
	 *            generator (with the same seed), you will get the same
	 *            ordering.
	 */
	public void shuffle(Random rnd) {
		int n_ex = size();
		for (int i = 0; i < n_ex; i++) {
			int j = i + rnd.nextInt(n_ex - i);

			IInstance tmp_ins = input_list.get(i);
			input_list.set(i, input_list.get(j));
			input_list.set(j, tmp_ins);

			IStructure tmp_st = output_list.get(i);
			output_list.set(i, output_list.get(j));
			output_list.set(j, tmp_st);

			if (weight_list != null) {
				Double tmp_weight = weight_list.get(i);
				weight_list.set(i, weight_list.get(j));
				weight_list.set(j, tmp_weight);
			}
		}
	}

	/**
	 * A helper function that helps you to split the training data
	 * 
	 * @param n_train
	 *            The number of the training examples.
	 * @return A {@link Pair} of the Binary Problem. The first one represents
	 *         the training set (which has n_train examples). The second one
	 *         represents the testing set
	 */
	public Pair<StructuredProblem, StructuredProblem> splitTrainTest(int n_train) {
		StructuredProblem train = new StructuredProblem();
		StructuredProblem test = new StructuredProblem();
		if (weight_list != null) {
			train.weight_list = new ArrayList<Double>();
			test.weight_list = new ArrayList<Double>();
		}

		for (int i = 0; i < size(); i++) {
			if (i < n_train) {
				train.input_list.add(input_list.get(i));
				train.output_list.add(output_list.get(i));
				if (weight_list != null) {
					train.weight_list.add(weight_list.get(i));
				}
			} else {
				test.input_list.add(input_list.get(i));
				test.output_list.add(output_list.get(i));
				if (weight_list != null) {
					test.weight_list.add(weight_list.get(i));
				}
			}
		}
		return new Pair<StructuredProblem, StructuredProblem>(train, test);
	}

	/**
	 * A helper function that helps you to perform cross validation. It splits
	 * the data in to n_fold {@link Pair}s, and each pair contains the (Training
	 * and Testing) split.
	 * 
	 * @param n_fold
	 *            The number of fold you wish to performance cross validations.
	 *            It equals to the length of the returned list.
	 * @param rnd
	 *            A random number generator. If you use the same seed, you will
	 *            generate the same split. It makes the comparisons between
	 *            different algorithms easier.
	 * @return
	 */
	public List<Pair<StructuredProblem, StructuredProblem>> splitData(
			int n_fold, Random rnd) {

		List<Integer> index_list = new ArrayList<Integer>();
		int bp_size = size();
		for (int i = 0; i < bp_size; i++)
			index_list.add(i);

		Collections.shuffle(index_list, rnd);
		List<Pair<StructuredProblem, StructuredProblem>> res = new ArrayList<Pair<StructuredProblem, StructuredProblem>>();

		for (int f = 0; f < n_fold; f++) {
			StructuredProblem cv_train = new StructuredProblem();
			StructuredProblem cv_test = new StructuredProblem();

			if (weight_list != null) {
				cv_train.weight_list = new ArrayList<Double>();
				cv_test.weight_list = new ArrayList<Double>();
			}

			for (int i = 0; i < bp_size; i++) {
				int real_idx = index_list.get(i);
				if ((i) % n_fold == f) {
					// test
					cv_test.input_list.add(input_list.get(real_idx));
					cv_test.output_list.add(output_list.get(real_idx));
					if (weight_list != null) {
						cv_test.weight_list.add(weight_list.get(real_idx));
					}
				} else {
					// train
					cv_train.input_list.add(input_list.get(real_idx));
					cv_train.output_list.add(output_list.get(real_idx));
					if (weight_list != null) {
						cv_train.weight_list.add(weight_list.get(real_idx));
					}
				}
			}

			res.add(new Pair<StructuredProblem, StructuredProblem>(cv_train,
					cv_test));
		}
		return res;
	}

}
