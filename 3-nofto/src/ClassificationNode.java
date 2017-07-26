
public class ClassificationNode {
    ClassificationNode left, right;
    float splitVal;
    int classif, level, splitFeature, startRow, endRow, total;
    double impurity, average = Double.NaN;

    public ClassificationNode(int classif, int total, double impurtity, int level, int startRow, int endRow) {
        this.classif = classif;
        this.total = total;
        this.startRow = startRow;
        this.endRow = endRow;
        this.level = level;
        this.impurity = impurtity;
    }

    public ClassificationNode(int level, int startRow, int endRow, double average, double error) {
        this.startRow = startRow;
        this.endRow = endRow;
        this.level = level;
        this.average = average;
        this.impurity = error;
    }

    public float getValue() {
        return Double.isNaN(average) ? classif / (float) total : (float) average;
    }

    public boolean isLeaf() {
        return left == null && right == null;
    }

    public boolean isPure() {
        return classif == 0 || classif == total;
    }

    public int getNumRows() {
        return endRow - startRow + 1;
    }
}