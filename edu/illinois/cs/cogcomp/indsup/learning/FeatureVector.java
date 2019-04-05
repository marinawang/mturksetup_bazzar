package edu.illinois.cs.cogcomp.indsup.learning;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import edu.illinois.cs.cogcomp.core.datastructures.Pair;

/**
 * Sparse Feature Vector Representation
 * 
 * @author Ming-Wei Chang
 * 
 */
public class FeatureVector implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1738932616256244560L;

	/**
	 * the indexes of the active features in this feature vector.
	 * 
	 * <b> Note that the feature should always start from 1 </b> 0 is preserved
	 * for some special operations.
	 */
	int[] idx;

	/**
	 * The values of the corresponding active features in this feature vector.
	 */
	double[] value;

	public int[] getIdx() {
		return idx;
	}

	public double[] getValue() {
		return value;
	}

	/**
	 * This function is used to handle adding a special bias term feature in
	 * LCLR/JLIS. The function will throw an exception if the feature which is
	 * about to be added already exists in the feature vector.
	 * 
	 * @param new_idx
	 *            The index of this new feature
	 * @param new_value
	 *            The value of this new feature
	 */
	public void slowAddFeature(int new_idx, double new_value) {
		boolean found = false;

		for (int i = 0; i < idx.length; i++) {
			if (idx[i] == new_idx) {
				found = true;
				break;
			}
		}

		assert !found;

		{
			int[] next_idx = new int[idx.length + 1];
			double[] next_value = new double[idx.length + 1];
			System.arraycopy(idx, 0, next_idx, 0, idx.length);
			System.arraycopy(value, 0, next_value, 0, idx.length);
			next_idx[idx.length] = new_idx;
			next_value[idx.length] = new_value;
			idx = next_idx;
			value = next_value;

		}
	}

	/**
	 * Divide each value of the features by c
	 * 
	 * @param c
	 */
	public void normalize(double c) {
		for (int i = 0; i < idx.length; i++) {
			value[i] /= c;
		}
	}

	/**
	 * return v*v
	 * 
	 * @return Two norm of this feature vector
	 */
	public double l2NormSqure() {
		double res = 0.0;
		for (int i = 0; i < idx.length; i++) {
			res += value[i] * value[i];
		}
		return res;
	}

	public FeatureVector(int[] f_idx_list, double[] f_value_list) {
		assert f_idx_list.length == f_value_list.length;

		// the users should never use feature index 0
		// the only way to add zero index feature is through slowAddFeature,
		// which should only be used by LCLR or JLIS
		for (int i = 0; i < f_idx_list.length; i++) {
			assert f_idx_list[i] > 0;
		}

		idx = new int[f_idx_list.length];
		System.arraycopy(f_idx_list, 0, idx, 0, idx.length);
		value = new double[f_value_list.length];
		System.arraycopy(f_value_list, 0, value, 0, idx.length);

	}

	/**
	 * 
	 * @return The biggest feature index of a feature vector
	 */
	public int maxIdx() {
		int max_id = -1;
		for (int i = 0; i < idx.length; i++) {
			if (max_id < idx[i])
				max_id = idx[i];
		}
		return max_id;
	}

	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer();

		for (int i = 0; i < idx.length; i++) {
			int cur_idx = idx[i];
			double cur_value = 0;
			cur_value = value[i];

			sb.append(cur_idx + ":" + cur_value + " ");
		}
		return sb.toString();
	}

	/**
	 * return a new feature vector which is exactly the same as the original
	 * one, except for shifting each index by "gap".
	 * 
	 * @param gap
	 * @return
	 */
	public FeatureVector copyWithShift(int gap) {
		FeatureVector res = new FeatureVector(this.idx, this.value);
		for (int i = 0; i < res.idx.length; i++) {
			res.idx[i] += gap;
			assert res.idx[i] >= 0;
		}
		return res;
	}

	/**
	 * return a new vector for (a-b)
	 * 
	 * @param a
	 * @param b
	 * @return
	 */

	public static FeatureVector minus(FeatureVector a, FeatureVector b) {
		FeatureVector fv = null;
		List<FeatureItem> af = convert2SortedFeatureNodeArray(a);
		List<FeatureItem> bf = convert2SortedFeatureNodeArray(b);
		Pair<int[], double[]> p = minus(af, bf);
		fv = new FeatureVector(p.getFirst(), p.getSecond());

		return fv;
	}

	/**
	 * return a new feature vector for a + b
	 * 
	 * @param a
	 * @param b
	 * @return
	 */
	public static FeatureVector plus(FeatureVector a, FeatureVector b) {
		FeatureVector fv = null;
		List<FeatureItem> af = convert2SortedFeatureNodeArray(a);
		List<FeatureItem> bf = convert2SortedFeatureNodeArray(b);
		Pair<int[], double[]> p = plus(af, bf);
		fv = new FeatureVector(p.getFirst(), p.getSecond());

		return fv;
	}

	private static Pair<int[], double[]> minus(List<FeatureItem> a,
			List<FeatureItem> b) {

		List<FeatureItem> tmp = new ArrayList<FeatureItem>();

		int a_idx = 0;
		int b_idx = 0;

		List<FeatureItem> alist = a;
		List<FeatureItem> blist = b;
		while (a_idx < alist.size() && b_idx < blist.size()) {
			FeatureItem af = alist.get(a_idx);
			FeatureItem bf = blist.get(b_idx);

			if (af.index < bf.index) {
				tmp.add(af);
				a_idx++;
			} else if (af.index > bf.index) {
				tmp.add(new FeatureItem(bf.index, -bf.value));
				b_idx++;
			} else {
				tmp.add(new FeatureItem(bf.index, af.value - bf.value));
				a_idx++;
				b_idx++;
			}
		}

		while (a_idx < alist.size()) {
			FeatureItem af = alist.get(a_idx);
			tmp.add(af);
			a_idx++;
		}

		while (b_idx < blist.size()) {
			FeatureItem bf = blist.get(b_idx);
			tmp.add(new FeatureItem(bf.index, -bf.value));
			b_idx++;
		}

		int[] res_idx = new int[tmp.size()];
		double[] res_value = new double[tmp.size()];

		for (int i = 0; i < tmp.size(); i++) {
			res_idx[i] = tmp.get(i).index;
			res_value[i] = tmp.get(i).value;
		}

		return new Pair<int[], double[]>(res_idx, res_value);
	}

	private static Pair<int[], double[]> plus(List<FeatureItem> a,
			List<FeatureItem> b) {

		List<FeatureItem> tmp = new ArrayList<FeatureItem>();

		int a_idx = 0;
		int b_idx = 0;

		List<FeatureItem> alist = a;
		List<FeatureItem> blist = b;
		while (a_idx < alist.size() && b_idx < blist.size()) {
			FeatureItem af = alist.get(a_idx);
			FeatureItem bf = blist.get(b_idx);

			if (af.index < bf.index) {
				tmp.add(af);
				a_idx++;
			} else if (af.index > bf.index) {
				tmp.add(bf);
				b_idx++;
			} else {
				tmp.add(new FeatureItem(bf.index, af.value + bf.value));
				a_idx++;
				b_idx++;
			}
		}

		while (a_idx < alist.size()) {
			FeatureItem af = alist.get(a_idx);
			tmp.add(af);
			a_idx++;
		}

		while (b_idx < blist.size()) {
			FeatureItem bf = blist.get(b_idx);
			tmp.add(bf);
			b_idx++;
		}

		int[] res_idx = new int[tmp.size()];
		double[] res_value = new double[tmp.size()];

		for (int i = 0; i < tmp.size(); i++) {
			res_idx[i] = tmp.get(i).index;
			res_value[i] = tmp.get(i).value;
		}

		return new Pair<int[], double[]>(res_idx, res_value);
	}

	private static List<FeatureItem> convert2SortedFeatureNodeArray(
			FeatureVector fv) {

		List<FeatureItem> res = new ArrayList<FeatureItem>();

		for (int i = 0; i < fv.idx.length; i++) {
			res.add(new FeatureItem(fv.idx[i], fv.value[i]));
		}

		Collections.sort(res, new Comparator<FeatureItem>() {
			// @Override
			public int compare(FeatureItem o1, FeatureItem o2) {
				if (o1.index < o2.index)
					return -1;
				else if (o1.index > o2.index)
					return 1;
				else
					return 0;
			}
		});

		return res;
	}

	public void sort(){
		List<FeatureItem> items = convert2SortedFeatureNodeArray(this);
		assert items.size() == this.idx.length;
		this.idx = new int[items.size()];
		this.value = new double[items.size()];
		
		for(int i =0; i < items.size(); i ++){
			this.idx[i] = items.get(i).index;
			this.value[i] = items.get(i).value;
		}
	}
}
