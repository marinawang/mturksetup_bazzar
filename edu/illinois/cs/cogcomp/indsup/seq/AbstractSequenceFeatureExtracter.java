package edu.illinois.cs.cogcomp.indsup.seq;

import java.io.Serializable;
import java.util.Map;

public abstract class AbstractSequenceFeatureExtracter implements Serializable{
	/**
	 * Application dependent code
	 * Extract emission features with feature name(String) and feature value (Double)
	 * Will append the current state  
	 * @param idx the index of the token, start from 0 and end with tokens.length-1 
	 * @return
	 */
	public abstract Map<String, Double> getEmissionFeatures(Sequence s, int index);
	
	/**
	 * Application dependent code
	 * Extract transition features with feature name(String) and feature value (Double)
	 * Will append the current state and the previous sate. When the idx ==0, think about these features as the prior features.  
	 * @param idx the index of the token, start from 0 and end with tokens.length-1
	 * @return
	 */
	public abstract Map<String, Double> getTransitionFeatures(Sequence s, int index);
}
