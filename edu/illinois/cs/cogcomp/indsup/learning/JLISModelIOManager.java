package edu.illinois.cs.cogcomp.indsup.learning;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

public class JLISModelIOManager {

	/**
	 * The function that is used to save the model into disk. This function just
	 * serialize the whole object into disk.
	 * 
	 * You can modify the save/load
	 * behavior by overriding this function.
	 * 
	 * @param fname
	 *            The filename of the saved model.
	 * @throws IOException
	 */
	public void saveModel(IJLISModel model, String fname)
			throws IOException {
		System.out.println("Save Model to " + fname + ".....");
		System.out.flush();
		ObjectOutputStream oos = new ObjectOutputStream(
				new BufferedOutputStream(new FileOutputStream(fname)));
		oos.writeObject(model);
		oos.close();
		System.out.println("Done!");
		System.out.flush();
	}

	/**
	 * The function is used to load the model. You can modify the save/load
	 * behavior by overriding this function.
	 * 
	 * @param fname
	 *            The filename of the saved model.
	 * @return
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	public IJLISModel loadModel(String fname) throws IOException,
			ClassNotFoundException {
		System.out.println("Load trained Models.....");
		System.out.flush();

		IJLISModel res = null;
		ObjectInputStream ios = new ObjectInputStream(new BufferedInputStream(
				new FileInputStream(fname)));

		res = (IJLISModel) ios.readObject();
		ios.close();
		System.out.println("Load Model complete!");
		return res;
	}
}
