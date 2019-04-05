package edu.illinois.cs.cogcomp.indsup.seq;

import java.io.IOException;


import edu.illinois.cs.cogcomp.indsup.inference.AbstractLossSensitiveStructureFinder;
import edu.illinois.cs.cogcomp.indsup.learning.IJLISModel;
import edu.illinois.cs.cogcomp.indsup.learning.JLISParameters;
import edu.illinois.cs.cogcomp.indsup.learning.WeightVector;

public class SequenceModel implements IJLISModel{	
	private static final long serialVersionUID = 3566686199447745188L;
	public WeightVector wv;
	public JLISParameters para;
	public AbstractSequenceFeatureExtracter ief;
	public AbstractSequenceLexManager lm;	
	public AbstractLossSensitiveStructureFinder s_finder;
			
}
