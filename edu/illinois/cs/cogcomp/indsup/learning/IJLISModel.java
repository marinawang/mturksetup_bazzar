package edu.illinois.cs.cogcomp.indsup.learning;

import java.io.Serializable;

/**
 * The class that allows the future implementations of a learned model.
 * <p>
 * 
 * To save and load of this function, please refer to {@link JLISModelIOManager}
 * <p>
 * 
 * Note that this an empty interface. In the implementation, you often want to
 * store the weight vector, the JLISParameter, store the lexicon manager, the
 * feature extractor and the inference solver.
 * 
 * @author Ming-Wei Chang
 * 
 */
public interface IJLISModel extends Serializable {

}
