package edu.illinois.cs.cogcomp.indsup.seq;

import java.util.HashMap;
import java.util.Map;


public class FirstOrderHMMFeatureExtractor extends
		AbstractSequenceFeatureExtracter {

	@Override
	public Map<String, Double> getEmissionFeatures(Sequence s, int index) {
		Map<String, Double> res = new HashMap<String, Double>();
		String t = s.tokens[index];
		res.put("Emission@" + "-" + t,1.0);		
		return res;
	}

	/**
	 * Will append current tag and previous tag automatically
	 */
	@Override
	public Map<String, Double> getTransitionFeatures(Sequence s, int index) {
		Map<String, Double> res = new HashMap<String, Double>();
		res.put("Transition",1.0);		
		return res;
	}
	
	

}
