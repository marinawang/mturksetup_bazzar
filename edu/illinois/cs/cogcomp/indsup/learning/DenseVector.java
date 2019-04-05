package edu.illinois.cs.cogcomp.indsup.learning;

import java.io.Serializable;

/**
 * The class that represents a dense vector. It also provides some operations
 * like dot-product, add, and norm.
 * 
 * @author Ming-Wei Chang
 * 
 */
public class DenseVector implements Serializable{
	private static final long serialVersionUID = 4496565917496408855L;
	protected double[] u = null;
	protected boolean extendable = true; // indicate if the weight vector is a
											// fixed size weight vector (true,
											// can allotcate memory);

	public DenseVector() {
	}

	/**
	 * @return If this vector is allowed to grow
	 */
	public boolean isExtendable() {
		return extendable;
	}

	/**
	 * Set the flag to indicate if this vector can grow or not
	 * 
	 * @param flag
	 */
	public void setExtendable(boolean flag) {
		extendable = flag;
	}

	/**
	 * @param n
	 *            Initialize the size of this dense vector
	 */
	public DenseVector(int n) {
		u = new double[n];
	}

	/**
	 * return The size of this dense vector
	 */
	public int getVectorLength() {
		return u.length;
	}

	/**
	 * return the dot product of a sparse feature vector and the dense vector
	 * itself. Note that if the sparse vector contains some elements (feature
	 * indexes) that do not exist in the dense vector, the dot product function
	 * will ignore them (instead of throwing an exception)
	 * 
	 * @param fv
	 *            Sparse feature vector
	 * @return
	 */
	public double dotProduct(FeatureVector fv) {
		double ret = dotProduct(u, fv);
		return ret;
	}

	/**
	 * Return the dot product of a dense feature vector and the dense vector
	 * itself
	 * 
	 * @param df
	 * @return
	 */
	public double dotProduct(DenseVector df) {
		double res = 0.0;
		int l = df.getVectorLength();

		if (u.length < l) {
			l = u.length;
		}

		for (int i = 0; i < l; i++) {
			res += u[i] * df.u[i];
		}
		return res;
	}

	/**
	 * Decide if the feature vector contains features which are not covered by
	 * the dense vector
	 * 
	 * @param fv
	 * @return
	 */
	public boolean needAllocateSpace(FeatureVector fv) {
		boolean ret = (fv.maxIdx() >= u.length);
		return ret;
	}

	/**
	 * Decide if the dense vector contains features which are not covered by the
	 * dense vector
	 * 
	 * @param fv
	 * @return
	 */

	public boolean needAllocateSpace(DenseVector dv) {
		boolean ret = (dv.getVectorLength() >= u.length);
		return ret;
	}

	/**
	 * increase the size of the dense vector by 2 or contain the new idx (pick
	 * the bigger one
	 * 
	 * @param new_idx
	 */
	public synchronized void allocateSpace(int new_idx) {
		assert extendable == true;
		assert new_idx >= u.length;
		double[] new_u = new double[Math.max(u.length * 2, new_idx + 1)];
		System.arraycopy(u, 0, new_u, 0, u.length);
		u = new_u;
	}

	/**
	 * Initialize an element of the weight vector, if the weight vector does not
	 * contain this item, allocate space internally
	 * 
	 * @param index
	 *            The index of the initialized item
	 * @param v
	 *            The value of the initialized item
	 */
	public synchronized void setElement(int index, double v) {
		assert extendable == true;
		if (index >= u.length) {
			System.out.println(index + " " + u.length);
			allocateSpace(index);
		}
		u[index] = v;
	}

	/**
	 * Add a dense vector back into the dense vector itself
	 * <p>
	 * 
	 * w = w + alpha * dv
	 * 
	 * @param dv
	 *            the dense vector
	 * @param alpha
	 *            the scalar
	 */
	public synchronized void addDenseVector(DenseVector dv, double alpha) {
		if (this.isExtendable() && this.needAllocateSpace(dv)) {
			this.allocateSpace(dv.getVectorLength());
		}

		int n = dv.getVectorLength();
		for (int i = 0; i < n; i++) {
			u[i] += alpha * dv.u[i];
		}
	}

	/**
	 * Add a sparse vector back into the dense vector itself
	 * <p>
	 * 
	 * w = w + alpha * fv
	 * 
	 * @param fv
	 *            A sparse feature vector
	 * @param alpha
	 *            The scalar
	 */
	public synchronized void addSparseFeatureVector(FeatureVector fv,
			double alpha) {
		if (this.isExtendable() && this.needAllocateSpace(fv)) {
			this.allocateSpace(fv.maxIdx());
		}

		int[] idx = fv.getIdx();
		double[] value = fv.getValue();

		for (int i = 0; i < idx.length; i++) {
			u[idx[i]] += alpha * value[i];
		}
	}

	/**
	 * The dot product between a sparse feature vector and a dense vector.
	 * <p>
	 * 
	 * TODO Note that we currently do not assume that the fv is sorted so that
	 * we need to check the out of bounary condition everytime. Vivek's
	 * experiments show that we can speed up by 20~30% by fixing this issue.
	 * 
	 * @param u
	 *            a dense vector
	 * @param fv
	 *            a sparse feature vector
	 * @return dot product
	 */
	private static double dotProduct(double[] u, FeatureVector fv) {

		double res = 0.0;

		for (int i = 0; i < fv.idx.length; i++) {
			if (fv.idx[i] < u.length)
				res += u[fv.idx[i]] * fv.value[i];

		}

		return res;
	}

	/**
	 * return the square of the 2-norm of this dense vector 
	 * @return
	 */
	public double getTwoNormSquare() {
		double res = 0;
		for (int i = 0; i < u.length; i++)
			res += u[i] * u[i];
		return res;
	}

	/** should avoid using this function, currently only for liblinear
	 * 
	 * @return the internal array representation of this vector
	 */
	public double[] getInternalArray() {
		return u;
	}
}
