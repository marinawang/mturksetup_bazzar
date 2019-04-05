package edu.illinois.cs.cogcomp.indsup.seq;

import java.io.Serializable;
import java.util.Map;

import edu.illinois.cs.cogcomp.indsup.learning.FeatureVector;
import edu.illinois.cs.cogcomp.indsup.learning.LexManager;

public class FirstOrderSequenceLexManager extends AbstractSequenceLexManager{
	
	public int getTotalNumberOfFeatures(){
		int total_features = getNEmissionFeas() * getNLabels()
		+ getNTransitionFeas() * getNLabels()
		+ getNTransitionFeas() * getNLabels()		
		* getNLabels();
		
		System.out.println("Feature Count: (emission)" + getNEmissionFeas()
				+ " (transition) " + getNTransitionFeas() + " (label) "
				+ getNLabels() + " (total) " + total_features);	
					
		return total_features;
	}

	
	
}
