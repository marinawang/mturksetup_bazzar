package edu.illinois.cs.cogcomp.indsup.seq;

public interface ISequenceStructureFactory {	
	public AbstractSequenceStructure genSequence(Sequence ins,String[] in_tags, AbstractSequenceLexManager lm);
}
