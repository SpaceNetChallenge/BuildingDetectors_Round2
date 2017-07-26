
public class ClassificationNode {
	ClassificationNode left, right;
	float splitVal;
	int classif, splitFeature, startRow, endRow, numRows;
	double impurity, average = Double.NaN;

	public ClassificationNode(int classif, int numRows, double impurtity, int startRow, int endRow) {
		this.classif = classif;
		this.numRows = numRows;
		this.startRow = startRow;
		this.endRow = endRow;
		this.impurity = impurtity;
	}

	public ClassificationNode(int numRows, int startRow, int endRow, double average, double error) {
		this.numRows = numRows;
		this.startRow = startRow;
		this.endRow = endRow;
		this.average = average;
		this.impurity = error;
	}

	public float getValue() {
		return Double.isNaN(average) ? classif / (float) numRows : (float) average;
	}

	public boolean isLeaf() {
		return left == null && right == null;
	}

	public boolean isPure() {
		return classif == 0 || classif == numRows;
	}
}