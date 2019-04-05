/**
 * 
 */
package edu.illinois.cs.cogcomp.indsup.seqdemo;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import edu.illinois.cs.cogcomp.core.datastructures.Pair;
import edu.illinois.cs.cogcomp.indsup.inference.AbstractLossSensitiveStructureFinder;
import edu.illinois.cs.cogcomp.indsup.learning.BinaryProblem;
import edu.illinois.cs.cogcomp.indsup.learning.JLISModelIOManager;
import edu.illinois.cs.cogcomp.indsup.learning.JLISParameters;
import edu.illinois.cs.cogcomp.indsup.learning.StructuredProblem;
import edu.illinois.cs.cogcomp.indsup.learning.WeightVector;
import edu.illinois.cs.cogcomp.indsup.learning.L2Loss.L2LossJLISLearner;
import edu.illinois.cs.cogcomp.indsup.learning.L2Loss.L2LossParallelJLISLearner;
import edu.illinois.cs.cogcomp.indsup.seq.FirstOrderHMMFeatureExtractor;
import edu.illinois.cs.cogcomp.indsup.seq.FirstOrderSeqStructureFactory;
import edu.illinois.cs.cogcomp.indsup.seq.SequenceDataLoader;
import edu.illinois.cs.cogcomp.indsup.seq.FirstOrderSequenceLexManager;
import edu.illinois.cs.cogcomp.indsup.seq.SequenceModel;
import edu.illinois.cs.cogcomp.indsup.seq.FirstOrderSequenceStructure;
import edu.illinois.cs.cogcomp.indsup.seq.HammingLossFirstOrderSeqFinder;

public class AllTest {

	public static Random rnd = new Random(0);

	private static String getFileNameWithoutDir(String train_name) {
		File f = new File(train_name);
		String[] tokens = f.getAbsolutePath().split("/");
		String model_name = tokens[tokens.length - 1];
		return model_name;
	}

	public static void trainSequenceSSVM(String train_name, String C_st_str)
			throws Exception {

		SequenceModel model = new SequenceModel();
		model.ief = new FirstOrderHMMFeatureExtractor();
		model.lm = new FirstOrderSequenceLexManager();

		List<String> all_files = new ArrayList<String>();
		all_files.add(train_name);

		//A special design class to let the learner know we are use FirstOrderSequence
		FirstOrderSeqStructureFactory ff = new FirstOrderSeqStructureFactory();
		
		//load all of the training data points 
		List<StructuredProblem> data_list = SequenceDataLoader
				.readStructureProblems(all_files, model.lm, model.ief, ff);
		
		// Disallow the creation of new features ()
		model.lm.disallowNewFeatures();
		
		// initialize the inference solver
		model.s_finder = new HammingLossFirstOrderSeqFinder(
				(FirstOrderSequenceLexManager) model.lm);

		
		JLISParameters para = new JLISParameters();

		// precalcuate all necessary features
		para.total_number_features = model.lm.getTotalNumberOfFeatures();
		para.c_struct = Double.parseDouble(C_st_str);

		L2LossJLISLearner learner = new L2LossJLISLearner();

		// train the model!
		model.wv = learner.trainStructuredSVM(model.s_finder, data_list.get(0),
				para);
		
		// save the model
		JLISModelIOManager iom = new JLISModelIOManager();
		iom.saveModel(model, getFileNameWithoutDir(train_name) + ".ssvm.model");
	}

	public static void trainSequenceSSVMParallel(String train_name,
			String C_st_str, String n_thread_str) throws Exception {

		SequenceModel model = new SequenceModel();
		model.ief = new FirstOrderHMMFeatureExtractor();
		model.lm = new FirstOrderSequenceLexManager();

		List<String> all_files = new ArrayList<String>();
		all_files.add(train_name);

		FirstOrderSeqStructureFactory ff = new FirstOrderSeqStructureFactory();

		List<StructuredProblem> data_list = SequenceDataLoader
				.readStructureProblems(all_files, model.lm, model.ief, ff);
		StructuredProblem train_sp = data_list.get(0);
		// Very important! so that we will not miscalculate the feature index
		model.lm.disallowNewFeatures();
		// initialize the inference solver

		JLISParameters para = new JLISParameters();

		// precalcuate all necessary features
		para.total_number_features = model.lm.getTotalNumberOfFeatures();
		para.c_struct = Double.parseDouble(C_st_str);
		

		L2LossParallelJLISLearner learner = new L2LossParallelJLISLearner();

		// allocate multiple solvers!
		int n_thread = Integer.parseInt(n_thread_str);
		System.out.println("Initializing Solvers...");
		System.out.flush();
		AbstractLossSensitiveStructureFinder[] s_finder_list = new AbstractLossSensitiveStructureFinder[n_thread];
		for (int i = 0; i < s_finder_list.length; i++) {
			s_finder_list[i] = new HammingLossFirstOrderSeqFinder(
					(FirstOrderSequenceLexManager) model.lm);
		}
		System.out.println("Done!");
		System.out.flush();

		// all the solvers are the same; put the first one in the model
		model.s_finder = s_finder_list[0];

		// parallel training the classifiers
		model.wv = learner.parallelTrainStructuredSVM(s_finder_list,train_sp
				, para);
		JLISModelIOManager iom = new JLISModelIOManager();
		iom.saveModel(model, getFileNameWithoutDir(train_name) + ".ssvm.model");
	}

	
	public static void testSequenceSSVM(String model_name, String test_name)
			throws Exception {
		JLISModelIOManager iom = new JLISModelIOManager();
		SequenceModel model = (SequenceModel) iom.loadModel(model_name);
		List<String> all_files = new ArrayList<String>();
		all_files.add(test_name);
		FirstOrderSeqStructureFactory ff = new FirstOrderSeqStructureFactory();
		List<StructuredProblem> data_list = SequenceDataLoader
				.readStructureProblems(all_files, model.lm, model.ief, ff);

		printTestACC(data_list.get(0), model.s_finder, model.wv);
	}

	private static List<StructuredProblem> getStructuredProblems(
			String train_name, String test_name,
			FirstOrderHMMFeatureExtractor ief, FirstOrderSequenceLexManager lm)
			throws IOException {
		List<String> all_files = new ArrayList<String>();
		all_files.add(train_name);
		all_files.add(test_name);

		FirstOrderSeqStructureFactory ff = new FirstOrderSeqStructureFactory();
		List<StructuredProblem> data_list = SequenceDataLoader
				.readStructureProblems(all_files, lm, ief, ff);
		return data_list;
	}

	private static void printTestACC(StructuredProblem sp,
			AbstractLossSensitiveStructureFinder s_finder, WeightVector ssvm_wv)
			throws IOException, Exception {

		double acc = 0.0;
		double total = 0.0;

		for (int i = 0; i < sp.input_list.size(); i++) {

			FirstOrderSequenceStructure gold = (FirstOrderSequenceStructure) sp.output_list
					.get(i);
			FirstOrderSequenceStructure prediction = (FirstOrderSequenceStructure) s_finder
					.getBestStructure(ssvm_wv, sp.input_list.get(i));

			for (int j = 0; j < prediction.tags.length; j++) {
				total += 1.0;
				if (prediction.tags[j].equals(gold.tags[j]))
					acc += 1.0;
			}
		}

		System.out.println("Acc = " + acc / total);
	}

	public static void trainSequenceSSVMWithIndirect(String train_name,
			String b_name, String C_st_str, String C_b_str) throws Exception {

		SequenceModel model = new SequenceModel();
		model.ief = new FirstOrderHMMFeatureExtractor();
		model.lm = new FirstOrderSequenceLexManager();

		// load training data
		List<String> all_files = new ArrayList<String>();
		all_files.add(train_name);

		FirstOrderSeqStructureFactory ff = new FirstOrderSeqStructureFactory();
		List<StructuredProblem> data_list = SequenceDataLoader
				.readStructureProblems(all_files, model.lm, model.ief, ff);

		// load binary labeled data
		BinaryProblem bp = SequenceDataLoader.readBinaryData(b_name, model.lm,
				model.ief);

		// Very important! so that we will not miscalculate the feature index
		model.lm.disallowNewFeatures();

		JLISParameters para = new JLISParameters();

		// precalcuate all necessary features
		para.total_number_features = model.lm.getTotalNumberOfFeatures();
		para.c_struct = Double.parseDouble(C_st_str);
		para.c_binary = Double.parseDouble(C_b_str);

		model.s_finder = new HammingLossFirstOrderSeqFinder(
				(FirstOrderSequenceLexManager) model.lm);
		L2LossJLISLearner learner = new L2LossJLISLearner();

		// train with both structure and binary labeled examples
		Pair<WeightVector, WeightVector> res = learner
				.trainStructuredSVMAndJLIS(model.s_finder, data_list.get(0),
						bp, para);

		model.wv = res.getFirst();
		JLISModelIOManager iom = new JLISModelIOManager();
		iom.saveModel(model, getFileNameWithoutDir(train_name) + ".ssvm.model");

		SequenceModel jlis_model = new SequenceModel();
		jlis_model.ief = model.ief;
		jlis_model.lm = model.lm;
		jlis_model.s_finder = model.s_finder;
		jlis_model.wv = res.getSecond();
		iom.saveModel(jlis_model, getFileNameWithoutDir(train_name) + "-"
				+ getFileNameWithoutDir(b_name) + ".jlis.model");
	}
}
