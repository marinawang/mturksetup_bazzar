package edu.illinois.cs.cogcomp.indsup.seq;

import java.io.Serializable;
import java.util.Map;

import edu.illinois.cs.cogcomp.indsup.learning.FeatureVector;
import edu.illinois.cs.cogcomp.indsup.learning.LexManager;

public abstract class AbstractSequenceLexManager implements Serializable{
	private LexManager label_lex = new LexManager();
	private LexManager emission_lex = new LexManager();
	private LexManager transition_lex = new LexManager();
	private boolean allow_new_features = true;
	
	
	
	public boolean isAllowNewfeatures() {
		return allow_new_features;
	}
	
	public void initializeLabels(String [] labs){
		if (allow_new_features)
			label_lex.initializeLabels(labs);
	}
	
	public boolean hasLabel(String str){
		return label_lex.containsLabel(str);
	}
	
	public int getLabelID(String s ){
		return label_lex.getLabelId(s);
	}
	
	public String getLabelStr(int id){
		return label_lex.getLabelString(id);
	}
	
	public int getNEmissionFeas(){
		return emission_lex.totalNumofFeature();
	}
	
	public int getNTransitionFeas(){
		return transition_lex.totalNumofFeature();
	}
	
	
	public int getNLabels(){
		return label_lex.totalNumofLabels();
	}
	
	public FeatureVector convertEmissionFeatures(Map<String,Double> raw_features){
		return emission_lex.convertRawFeaMap2LRFeatures(raw_features);
	}
	
	
	public FeatureVector convertTransitionFeatures(Map<String,Double> raw_features){
		return transition_lex.convertRawFeaMap2LRFeatures(raw_features);
	}
	
	public void disallowNewFeatures(){
		emission_lex.setAllowNewFeatures(false);
		transition_lex.setAllowNewFeatures(false);		
		allow_new_features = false;
		System.out.println("No More new Features in lexicon");
	}

	
	public abstract int getTotalNumberOfFeatures();

	public LexManager getEmissionLex() {
		return emission_lex;
	}
	
	public LexManager getTransitionLex() {
		return transition_lex;
	}

}
