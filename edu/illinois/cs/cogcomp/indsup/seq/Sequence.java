package edu.illinois.cs.cogcomp.indsup.seq;


import edu.illinois.cs.cogcomp.indsup.inference.IInstance;
import edu.illinois.cs.cogcomp.indsup.learning.FeatureVector;

public class Sequence implements IInstance {

	public final String[] tokens;
	
	public FeatureVector[] em_feature_list; //will only append current state
	public FeatureVector[] tr_feature_list; //will append both current state and previous state (including prior)
	
	public Sequence(String[] tokens, AbstractSequenceLexManager lm, AbstractSequenceFeatureExtracter afe) {
		this.tokens = new String[tokens.length];
		System.arraycopy(tokens, 0, this.tokens, 0, tokens.length);		
		recalcuateFeatureCache(lm,afe);
	}
	
	public Sequence(String[] tokens, AbstractSequenceLexManager lm, AbstractSequenceFeatureExtracter afe, boolean recalculate) {
		this.tokens = new String[tokens.length];
		System.arraycopy(tokens, 0, this.tokens, 0, tokens.length);		
		if (recalculate)
			recalcuateFeatureCache(lm,afe);
	}

	/**
	 * Calculate base feature cache first, when use these features to create the real features once the labels are known
	 * @param lm
	 * @param afe 
	 */
	public void recalcuateFeatureCache(AbstractSequenceLexManager lm, AbstractSequenceFeatureExtracter afe) {
		em_feature_list = new FeatureVector[tokens.length];
		tr_feature_list = new FeatureVector[tokens.length];
		
		for (int i=0; i < tokens.length; i++){											
			em_feature_list[i] = lm.convertEmissionFeatures(afe.getEmissionFeatures(this,i));
			tr_feature_list[i] = lm.convertTransitionFeatures(afe.getTransitionFeatures(this,i));
		}		
	}

	@Override
	public double size() {
		return tokens.length;
	}

	@Override
	public String toString() {

		StringBuffer sb = new StringBuffer();

		for (int i = 0; i < tokens.length; i++)
			sb.append(tokens[i] + "\t");
		sb.append("\n");

		return sb.toString();
	} 
}