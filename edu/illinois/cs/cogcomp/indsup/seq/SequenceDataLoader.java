package edu.illinois.cs.cogcomp.indsup.seq;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.illinois.cs.cogcomp.core.io.LineIO;
import edu.illinois.cs.cogcomp.indsup.inference.IInstance;
import edu.illinois.cs.cogcomp.indsup.inference.IStructure;
import edu.illinois.cs.cogcomp.indsup.learning.BinaryProblem;
import edu.illinois.cs.cogcomp.indsup.learning.StructuredProblem;

public class SequenceDataLoader {

	/**
	 * File format: odd lines: words even lines: tags
	 * 
	 * @param fname
	 * @return
	 * @throws IOException
	 */
	private static StructuredProblem readStructuredData(String fname,
			AbstractSequenceLexManager lm,
			AbstractSequenceFeatureExtracter afe, ISequenceStructureFactory f) throws IOException {
		List<String> str_list = LineIO.read(fname);

		assert str_list.size() % 2 == 0; // must be even; contains labels

		int step = Math.max(str_list.size() / 10, 100);

		List<IInstance> x_list = new ArrayList<IInstance>();
		List<IStructure> y_list = new ArrayList<IStructure>();

		for (int i = 0; i < str_list.size(); i += 2) {
			if ((i / 2) % step == 0)
				System.out.println("read data ... " + (i / 2) + "/"
						+ (str_list.size() / 2));

			String[] words = str_list.get(i).split("\\s+");
			Sequence x = new Sequence(words, lm, afe);

			String[] tags = str_list.get(i + 1).split("\\s+");
			assert words.length == tags.length;

			x_list.add(x);
			y_list.add(f.genSequence(x, tags, lm));
		}

		StructuredProblem ans = new StructuredProblem();
		ans.input_list = x_list;
		ans.output_list = y_list;
		return ans;
	}

	/**
	 * File format: odd lines: +1 or -1 even lines: words
	 * 
	 * 
	 * @param fname
	 * @return
	 * @throws IOException
	 */
	public static BinaryProblem readBinaryData(String fname,
			AbstractSequenceLexManager lm,
			AbstractSequenceFeatureExtracter afe) throws IOException {
		List<String> str_list = LineIO.read(fname);

		assert str_list.size() % 2 == 0; // must be even; contains labels

		int step = Math.max(str_list.size() / 10, 100);

		List<IInstance> x_list = new ArrayList<IInstance>();
		List<Integer> y_list = new ArrayList<Integer>();

		for (int i = 0; i < str_list.size(); i += 2) {
			if ((i / 2) % step == 0)
				System.out.println("read data ... " + (i / 2) + "/"
						+ (str_list.size() / 2));

			String[] words = str_list.get(i + 1).split("\\s+");
			Sequence x = new Sequence(words, lm, afe);

			int label = Integer.parseInt(str_list.get(i));
			assert label == 1 || label == -1;

			x_list.add(x);
			y_list.add(label);
		}

		BinaryProblem ans = new BinaryProblem();
		ans.input_list = x_list;
		ans.output_list = y_list;
		return ans;
	}

	public static List<StructuredProblem> readStructureProblems(
			List<String> sp_file_list, AbstractSequenceLexManager lm,
			AbstractSequenceFeatureExtracter afe, ISequenceStructureFactory f) throws IOException {

		// read all data
		List<StructuredProblem> sp_list = new ArrayList<StructuredProblem>();

		for (String fname : sp_file_list) {
			sp_list.add(readStructuredData(fname, lm, afe,f));
		}

		// collect all labels and tell the lexicon manager only in the training phase
		if (lm.isAllowNewfeatures()) {
			Set<String> lab_set = new HashSet<String>();
			for (StructuredProblem sp : sp_list) {
				for (IStructure s : sp.output_list) {
					for (String lab : ((AbstractSequenceStructure) s).tags) {
						if (!lab_set.contains(lab))
							lab_set.add(lab);
					}
				}
			}

			String[] labels = lab_set.toArray(new String[lab_set.size()]);
			lm.initializeLabels(labels);
		}
		
		return sp_list;
	}

}
