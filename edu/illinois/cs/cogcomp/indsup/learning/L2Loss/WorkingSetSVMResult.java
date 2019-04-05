package edu.illinois.cs.cogcomp.indsup.learning.L2Loss;

import edu.illinois.cs.cogcomp.indsup.learning.WeightVector;

public class WorkingSetSVMResult {
	public WeightVector wv;
	public double objective_vaule;
	public boolean finished;
	
	public WorkingSetSVMResult(WeightVector wv, double obj_value, boolean finished){
		this.wv =wv;
		this.objective_vaule = obj_value;
		this.finished = finished;
	}
}
