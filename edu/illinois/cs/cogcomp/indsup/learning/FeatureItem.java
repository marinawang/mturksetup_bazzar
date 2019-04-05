package edu.illinois.cs.cogcomp.indsup.learning;
/**
 * A class that helps {@link FeatureVector} manages some operations. The users should not use this class.
 * @author Ming-Wei Chang
 *
 */
public class FeatureItem {
	

    public final int    index;
    public final double value;

    public FeatureItem( final int index, final double value ) {
        if (index < 1) throw new IllegalArgumentException("index must be >= 1");
        this.index = index;
        this.value = value;
    }
    
	@Override 	
	public boolean equals( Object aThat ) {
		
	    //check for self-comparison
	    if ( this == aThat ) return true;

	    if ( !(aThat instanceof FeatureItem) ) return false;
	    //Alternative to the above line :
	    //if ( aThat == null || aThat.getClass() != this.getClass() ) return false;

	    //cast to native object is now safe
	    FeatureItem that = (FeatureItem)aThat;

	    //now a proper field-by-field evaluation can be made
	    return this.index == that.index && (Math.abs(this.value -that.value) < 1e-30); 	        
	  }	

	 @Override 
	 public int hashCode() {
	    return (17*37 + index) + (23*37 + (int) Double.doubleToLongBits(value));
	 }	
}

