package edu.illinois.cs.cogcomp.indsup.seq;

import java.util.Random;

import edu.illinois.cs.cogcomp.core.datastructures.Pair;
import edu.illinois.cs.cogcomp.indsup.inference.AbstractLossSensitiveStructureFinder;
import edu.illinois.cs.cogcomp.indsup.inference.IInstance;
import edu.illinois.cs.cogcomp.indsup.inference.IStructure;
import edu.illinois.cs.cogcomp.indsup.learning.FeatureVector;
import edu.illinois.cs.cogcomp.indsup.learning.WeightVector;

public class HammingLossFirstOrderSeqFinder extends
		AbstractLossSensitiveStructureFinder {

	int MAX_TOKENS = 500;
	Random rnd = null;
	boolean RND_ORDER = false;	
	protected FirstOrderSequenceLexManager lm = null;	
	double score = 0;
	double[][] emission_scores;
	double[] prior_scores;
	double[][][] transition_scores;
	private double[][] dp_table;
	private int[][] path_table;


	public HammingLossFirstOrderSeqFinder(FirstOrderSequenceLexManager lm) {		
		this.lm = lm;
		int n_lab = lm.getNLabels();
		emission_scores = new double[MAX_TOKENS][n_lab];
		prior_scores = new double[n_lab];
		transition_scores = new double[MAX_TOKENS][n_lab][n_lab];
		dp_table = new double[MAX_TOKENS][n_lab];
		path_table = new int[MAX_TOKENS][n_lab];

	}

	@Override
	public Pair<IStructure, Double> getLossSensitiveBestStructure(
			WeightVector wv, IInstance input, IStructure gold)
			throws Exception {
		assert lm.isAllowNewfeatures() == false;
		FirstOrderSequenceStructure gold_labeled_seq = (FirstOrderSequenceStructure) gold;

		Sequence sen = (Sequence) input;
		int n_lab = lm.getNLabels();
		int n_tokens = sen.tokens.length;
		int n_emi_fea = lm.getNEmissionFeas();
		int n_tran_fea = lm.getNTransitionFeas();


		prepareDP(wv, sen, n_lab, n_tokens, n_emi_fea, n_tran_fea);


		int[] gold_lab_id_list = new int[gold_labeled_seq.tags.length];
		for (int i = 0; i < gold_labeled_seq.tags.length; i++) {
			gold_lab_id_list[i] = lm.getLabelID(gold_labeled_seq.tags[i]);
		}

		// modify emission score to consider delta
		for (int i = 0; i < n_tokens; i++) {
			for (int j = 0; j < n_lab; j++) {
				if (j != gold_lab_id_list[i])
					emission_scores[i][j] += 1.0; // hmming loss
			}
		}

		Pair<double[][], int[][]> res = viterbi(n_tokens,n_lab);
		
		String[] labels = getSequence(n_tokens,n_lab);
		double loss = 0;
		for (int i = 0; i < n_tokens; i++)
			if (!labels[i].equals(gold_labeled_seq.tags[i]))
				loss += 1.0;
		
		return new Pair<IStructure, Double>(getNewCLS(sen, labels), loss);

	}

	private Pair<double[][], int[][]> viterbi(int n_tokens,int n_tags) {
		
		// System.out.println("trainsition size : " +
		// transition_score[0][0].length);
		// assert transition_score[0].length == transition_score[0][0].length;
		// assert emission_score.length == transition_score.length + 1;

		
		// initialization
		for (int i = 0; i < n_tokens; i++) {
			for (int j = 0; j < n_tags; j++) {
				dp_table[i][j] = Double.NEGATIVE_INFINITY;
				path_table[i][j] = -1;
			}
		}

		for (int i = 0; i < n_tokens; i++) {

			if (i == 0) {
				for (int j = 0; j < n_tags; j++) {
					double priorScore = prior_scores[j];
					double zeroOrderScore = emission_scores[i][j];
					dp_table[i][j] = priorScore + zeroOrderScore;
				}
			} else {
				for (int j = 0; j < n_tags; j++) {
					double zeroOrderScore = emission_scores[i][j];
					for (int k = 0; k < n_tags; k++) {
						double firstOrderScore = transition_scores[i - 1][k][j];
						double candidate_score = dp_table[i - 1][k]
								+ zeroOrderScore + firstOrderScore;

						if (candidate_score > dp_table[i][j]) {
							dp_table[i][j] = candidate_score;
							path_table[i][j] = k;
						}
					}
				}
			}
		}

		return new Pair<double[][], int[][]>(dp_table, path_table);
	}

	private String[] getSequence(int n_tokens, int n_lab) {
		score = 0;
		int n_tags = n_lab;		

		String[] tags = new String[n_tokens];
		// find the best tags at the end
		double max_score = Double.NEGATIVE_INFINITY;
		int max_tag = -1;

		for (int i = 0; i < n_tags; i++)
			if (dp_table[n_tokens - 1][i] > max_score) {
				max_score = dp_table[n_tokens - 1][i];
				max_tag = i;
			}

		//System.out.println("max_score:" + max_score);
		//IMPORTANT: fix the bias introduced
		score = max_score;
		tags[n_tokens - 1] = lm.getLabelStr(max_tag);

		int cur_tag = max_tag;
		for (int i = n_tokens - 1; i >= 1; i--) {
			cur_tag = path_table[i][cur_tag]; // trace back one step;
			tags[i - 1] = lm.getLabelStr(cur_tag);
		}
		return tags;
	}

	public double getCurrentMaxScore() {
		
		return score;
	}

	@Override
	public IStructure getBestStructure(WeightVector wv,
			IInstance input) throws Exception {
		Sequence sen = (Sequence) input;
		int n_lab = lm.getNLabels();
		int n_tokens = sen.tokens.length;
		int n_emi_fea = lm.getNEmissionFeas();
		int n_tran_fea = lm.getNTransitionFeas();

		prepareDP(wv, sen, n_lab, n_tokens, n_emi_fea, n_tran_fea);

		Pair<double[][], int[][]> res = viterbi(n_tokens,n_lab);

		String[] labels = getSequence(n_tokens,n_lab);
			
		return getNewCLS(sen, labels);
	}

	private FirstOrderSequenceStructure getNewCLS(Sequence sen, String[] labels) {		
		FirstOrderSequenceStructure res = new FirstOrderSequenceStructure(sen, labels, lm);		
		return res;
	}

	private void prepareDP(WeightVector wv, Sequence sen, int n_lab,
			int n_tokens, int n_emi_fea, int n_tran_fea) {
		
		for (int i = 0; i < n_tokens; i++) {
			FeatureVector fv = sen.em_feature_list[i]; // cached features
			for (int j = 0; j < n_lab; j++) {				
				emission_scores[i][j] = wv.dotProduct(fv.copyWithShift(j*n_emi_fea));				
			}
		}

		int emission_gap = n_emi_fea * n_lab;
		{
			FeatureVector fv = sen.tr_feature_list[0]; // cached
																		// features
			for (int j = 0; j < n_lab; j++) {
				int s = emission_gap + n_tran_fea * j;				
				prior_scores[j] = wv.dotProduct(fv.copyWithShift(s));				
			}
		}

		int prior_emission_gap = emission_gap + n_tran_fea * n_lab;

		// calculate transition features
		{
			for (int i = 1; i < n_tokens; i++) {
				FeatureVector temp_fv = sen.tr_feature_list[i];
				for (int k = 0; k < n_lab; k++) {
					for (int j = 0; j < n_lab; j++) {
						int tr_s = prior_emission_gap + n_tran_fea
								* (k * n_lab + j);						
						transition_scores[i - 1][k][j] = wv
								.dotProduct(temp_fv.copyWithShift(tr_s));						

					}
				}
			}
		}
	}

}
