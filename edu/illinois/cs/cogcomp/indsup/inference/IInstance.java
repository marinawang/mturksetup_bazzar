package edu.illinois.cs.cogcomp.indsup.inference;

/**
 * This interface represents an input example x. In the part-of-speech example,
 * the IInstance represents an input sentence.
 * 
 * @author Ming-Wei Chang
 * 
 */

public interface IInstance {
	/***
	 * The function that returns the size of this input example. We only use
	 * this function to normalize the feature vector in LCLR and JLIS. The
	 * normalization is sometimes important for the case of binary output
	 * problem with latent variables, given that we try to learn a linear
	 * decision function over examples with various size. In the POS example,
	 * the size() is just the number of tokens in a sentence.
	 * 
	 * @return the size of this instance
	 */
	public double size();
}
