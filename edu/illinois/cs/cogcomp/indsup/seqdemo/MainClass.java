package edu.illinois.cs.cogcomp.indsup.seqdemo;

import java.io.IOException;

import edu.illinois.cs.cogcomp.core.io.IOUtils;
import edu.illinois.cs.cogcomp.core.utilities.commands.InteractiveShell;
import edu.illinois.cs.cogcomp.indsup.seqdemo.AllTest;


public class MainClass {
	
	public static String getProjectRoot() {
		String projectRoot = null;
		try {
			projectRoot = IOUtils.pwd();
		} catch (IOException e) {
			e.printStackTrace();
		}

		int len = projectRoot.length();
		if (projectRoot.substring(len - 4, len).equals("dist"))
			projectRoot += "/../";
		return projectRoot;
	}


	public static void main(String[] args) throws Exception{
		

		InteractiveShell<AllTest> tester = new InteractiveShell<AllTest>(
				AllTest.class);

		if (args.length == 0)
			tester.showDocumentation();
		else
		{
			long start_time = System.currentTimeMillis();
			tester.runCommand(args);
					
			System.out.println("This experiment took "
					+ (System.currentTimeMillis() - start_time) / 1000.0
					+ " secs");
		}
	}
}
