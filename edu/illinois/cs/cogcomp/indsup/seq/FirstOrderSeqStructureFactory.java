package edu.illinois.cs.cogcomp.indsup.seq;

public class FirstOrderSeqStructureFactory implements ISequenceStructureFactory {

	@Override
	public AbstractSequenceStructure genSequence(Sequence ins,
			String[] in_tags, AbstractSequenceLexManager lm) {
		
		return new FirstOrderSequenceStructure(ins, in_tags, (FirstOrderSequenceLexManager) lm);
	}

}
