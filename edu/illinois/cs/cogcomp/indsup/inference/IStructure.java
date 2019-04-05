package edu.illinois.cs.cogcomp.indsup.inference;

import edu.illinois.cs.cogcomp.indsup.learning.FeatureVector;

/**
 * An important class that represents a structure (y). Note that it should also
 * contains the information of the corresponding example (x) because the feature
 * vector is over (x,y). We let the users to decide how to implement this
 * relationship. In the POS tagging example, this class represents the POS tags.
 * 
 * @author Ming-Wei Chang
 * 
 */
public interface IStructure {

	/**
	 * The function that returns the feature vector \Phi(x,y). Note that this
	 * feature function is generated over the input-structure pair. Therefore,
	 * different structures will result in different feature vectors.
	 * 
	 * @return Feature Vector \Phi(x,y), where x is the input and y is the
	 *         output structure
	 */
	public abstract FeatureVector getFeatureVector();

	/**
	 * Override the toString function. It usually prints the structure
	 * information so that it is easier to debug.
	 * 
	 */
	@Override
	public abstract String toString();

	/**
	 * Tell if two structures are equal or not
	 */
	@Override
	public abstract boolean equals(Object aThat);

	/**
	 * Hash code for the structure
	 */
	@Override
	public abstract int hashCode();
}
