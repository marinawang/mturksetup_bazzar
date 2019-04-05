package edu.illinois.cs.cogcomp.indsup.seq;

import java.io.Serializable;

import edu.illinois.cs.cogcomp.indsup.inference.IStructure;
import edu.illinois.cs.cogcomp.indsup.learning.FeatureVector;

public class FirstOrderSequenceStructure extends AbstractSequenceStructure implements Serializable{
		
	public FirstOrderSequenceStructure(Sequence ins,String[] in_tags, FirstOrderSequenceLexManager lm){
		super(ins,in_tags,lm);		
	}
	
	@Override
	public boolean equals(Object aThat) {
		// check for self-comparison
		if (this == aThat)
			return true;

		if (!(aThat instanceof FirstOrderSequenceStructure))
			return false;		

		// cast to native object is now safe
		FirstOrderSequenceStructure that = (FirstOrderSequenceStructure) aThat;
		
		//check if their instance is the same
		boolean sameInstance = true;
		
		if (this.getIns().size() != that.getIns().size())
			sameInstance = false;
		else{
			for(int i=0; i < this.getIns().size(); i ++){
				if (!this.getIns().tokens[i].equals(that.getIns().tokens[i])){
					sameInstance = false;
					break;
				}
			}
		}
		
		assert sameInstance;
		
		//check if their tags are the same
		for(int i=0; i < this.tags.length; i ++){
			if (!this.tags[i].equals(that.tags[i])){
				return false;
			}
		}
				
		return true;
	}


	@Override
	public FeatureVector getFeatureVector() {
		// Very important! In order to save time we generate features over \phi(x,y) pairs by shifting the base features
		// This will only works if we know the gap, so we cannot allow new features here
		assert lm.isAllowNewfeatures() == false;

		/* Weight Vector Structure: arrange features in the following way:
		 *  
		 * emission_features_lab_1
		 * emission_features_lab_2 ... emission_features_lab_m
		 * prior_features_label_1,prior_features_label_2
		 * ......transitionfeatures_label1_label2 .....
		 * 
		 * total_features = 
		 */

		FeatureVector fv = new FeatureVector(new int[0], new double[0]);
		int len = tags.length;
		int[] lab_list = new int[len];
		
		for(int i=0; i < len; i ++){
			lab_list[i] = lm.getLabelID(tags[i]);
		}
		// calculate emission features.....

		
		int emission_step = lm.getNEmissionFeas();
		{
			
			for (int i = 0; i < ins.tokens.length; i++) {
				FeatureVector iFv = ins.em_feature_list[i];
				int lab_id = lab_list[i];				
				fv = FeatureVector.plus(iFv.copyWithShift(emission_step * lab_id), fv);				
			}
		}

		// calculate prior features
		int n_labs = lm.getNLabels();
		int emission_gap = emission_step * n_labs;
		int prior_step = lm.getNTransitionFeas();
		
		
		{		
			int lab_id = lab_list[0];
			FeatureVector iFv = ins.tr_feature_list[0];
			int s = emission_gap + prior_step * lab_id;			
			fv = FeatureVector.plus(iFv.copyWithShift(s),fv);			
		}

		int prior_emission_gap = emission_gap + prior_step * n_labs;
		int transition_step = prior_step; // we assume prior features and transition are from the same base features
		// calculate transition features
		{
			for (int i = 1; i < len; i++) {
				FeatureVector iFv = ins.tr_feature_list[i];
				int cur_lab_id = lab_list[i];
				int pre_lab_id = lab_list[i - 1];
				int tr_s = prior_emission_gap + transition_step
						* (pre_lab_id * n_labs + cur_lab_id);
				
				fv = FeatureVector.plus(iFv.copyWithShift(tr_s),fv);					
			}
		}
		
		//int comma_gap = prior_emission_gap + transition_step * n_labs * n_labs;
		
		return fv;		
	}

	
	public Sequence getIns() {
		return ins;
	}

}
