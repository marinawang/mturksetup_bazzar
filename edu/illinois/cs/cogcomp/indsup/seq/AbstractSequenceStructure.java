package edu.illinois.cs.cogcomp.indsup.seq;

import edu.illinois.cs.cogcomp.indsup.inference.IStructure;
import edu.illinois.cs.cogcomp.indsup.learning.FeatureVector;

public abstract class AbstractSequenceStructure implements IStructure{
	
	protected final AbstractSequenceLexManager lm;
	protected final Sequence ins;	
	public final String[] tags;
	
	public AbstractSequenceStructure(Sequence ins,String[] in_tags, AbstractSequenceLexManager lm){
		assert ins.size() == in_tags.length;
		this.ins = ins;
		// copy contains in order to be safe!
		this.tags = new String[in_tags.length];
		for(int i=0; i<in_tags.length; i ++){
			this.tags[i] = in_tags[i];
		}
		this.lm = lm;
		
	}
}
