package edu.illinois.cs.cogcomp.indsup.util;

import java.io.File;

public class JLISUtils {
	public static String getFileNameWithoutDir(String train_name) {
		File f = new File(train_name);
		String[] tokens = f.getAbsolutePath().split("/");
		String model_name = tokens[tokens.length - 1];
		return model_name;
	}
}
