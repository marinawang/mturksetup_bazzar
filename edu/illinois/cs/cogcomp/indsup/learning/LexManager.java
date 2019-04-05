package edu.illinois.cs.cogcomp.indsup.learning;

import java.io.Serializable;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A class that managers the lexicon (the mappings between feature names and
 * feature indexes)
 * 
 * 
 * @author Ming-Wei Chang
 * 
 */
public class LexManager implements Serializable {

	private static final long serialVersionUID = 1L;

	// be careful, start from zero
	Map<String, Integer> feaStr2Id_map = null;
	Map<Integer, String> feaId2Str_map = null;

	Map<String, Integer> LabelStr2Id_map = null;
	Map<Integer, String> Id2LabelStr_map = null;

	// It is truly important to have this. look at the comment in the
	// constructor
	public static final String bias_str = "*-global-bias-*";

	private boolean allow_new_features = true;

	/**
	 * The constructor: Note that the feature indexes need to be greater than
	 * zero. We preview the 0 feature in the constructor to prevent the user
	 * tries to use feature 0
	 */
	public LexManager() {
		feaStr2Id_map = new ConcurrentHashMap<String, Integer>();
		feaId2Str_map = new ConcurrentHashMap<Integer, String>();

		// VERY IMPORTANT
		// In the current of the weight wector, we use zero as the global bias
		// term
		// therefore, it is very important to preview the first feature
		// so no one can take over the zero index!
		this.previewFeature(bias_str); // zero should always be bias

		// What happened in Structured case?
		// The structure case might have their own lex
		// However, they also need to find a way to keep the zero index for the
		// bias term
		// In JLIS-sequence, we use this Lexmanager inside the structural lex
		// manger
		// Therefore, the zero index will still be not used by regular features

	}

	/**
	 * The function that views all of the label names. Should always call this
	 * function before using {@link LexManager#hasLabel(String)},
	 * {@link LexManager#getLabelId(String)} and
	 * {@link LexManager#getLabelId(String)}
	 * 
	 * @param labs
	 */
	public void initializeLabels(String[] labs) {
		LabelStr2Id_map = new ConcurrentHashMap<String, Integer>();
		Id2LabelStr_map = new ConcurrentHashMap<Integer, String>();

		for (String str : labs) {
			int v = LabelStr2Id_map.size();
			LabelStr2Id_map.put(str, v);
			Id2LabelStr_map.put(v, str);
		}
	}

	// /**
	// * Check if the labels has been set before
	// * @param str The string that represets the label
	// * @return
	// */
	// public boolean hasLabel(String str){
	// return LabelStr2Id_map.containsKey(str);
	// }

	/**
	 * Get the index for this label string
	 * 
	 * @param str
	 *            A label string
	 * @return
	 */
	public int getLabelId(String str) {
		if (!containsLabel((str)))
			System.out.println("!!!!! " + str);

		return LabelStr2Id_map.get(str);
	}

	/**
	 * Get the name of the label
	 * 
	 * @param id
	 * @return
	 */
	public String getLabelString(int id) {
		return Id2LabelStr_map.get(id);
	}

	/**
	 * @return Total number of labels
	 */
	public int totalNumofLabels() {
		return LabelStr2Id_map.size();
	}

	public String getFeatureString(int id) {
		return feaId2Str_map.get(id);
	}

	public int totalNumofFeature() {
		return feaStr2Id_map.size();
	}

	public int getFeatureID(String s) {
		assert feaStr2Id_map.containsKey(s);
		return feaStr2Id_map.get(s);
	}

	public boolean containsLabel(String s) {
		return LabelStr2Id_map.containsKey(s);
	}

	public boolean containFeature(String s) {
		return feaStr2Id_map.containsKey(s);
	}

	/**
	 * Let the LexManager remember this feature
	 * 
	 * @param s
	 */
	public synchronized void previewFeature(String s) {
		assert allow_new_features == true;

		if (!feaStr2Id_map.containsKey(s)) {
			int v = feaStr2Id_map.size();
			feaStr2Id_map.put(s, v);
			feaId2Str_map.put(v, s);
		}
	}

	public Set<String> allCurrentFeatures() {
		return feaStr2Id_map.keySet();
	}

	/**
	 * Note that only {@link LexManager#previewFeature(String)} and this
	 * function can let {@link LexManager} remember the features and their
	 * corresponding indexes.
	 * 
	 * @param ex_fea_map
	 *            The String representations of the feature vector
	 * @return The final FeatureVector.
	 */
	public synchronized FeatureVector convertRawFeaMap2LRFeatures(
			Map<String, Double> ex_fea_map) {
		int real_size = 0;

		for (String s : ex_fea_map.keySet()) {
			if (!feaStr2Id_map.containsKey(s) && !allow_new_features)
				continue;
			real_size++;
		}

		int[] idx = new int[real_size];
		double[] values = new double[real_size];

		int cur_i = 0;
		for (String s : ex_fea_map.keySet()) {
			if (!feaStr2Id_map.containsKey(s)) {

				if (!allow_new_features)
					continue; // do not create any features..

				int new_id = feaStr2Id_map.size();
				feaStr2Id_map.put(s, new_id);
				feaId2Str_map.put(new_id, s);

			}
			int id = feaStr2Id_map.get(s);

			idx[cur_i] = id;
			values[cur_i] = ex_fea_map.get(s);
			cur_i += 1;
		}

		return new FeatureVector(idx, values);
	}

	/**
	 * Control if you want the LexManger to create more features.
	 * 
	 * @param allow_new_features
	 */
	public void setAllowNewFeatures(boolean allow_new_features) {
		this.allow_new_features = allow_new_features;
	}

	/**
	 * @return return if the lex allows new feature
	 */
	public boolean isAllowNewFeatures() {
		return allow_new_features;
	}

}
