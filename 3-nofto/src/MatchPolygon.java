import java.awt.Polygon;

public class MatchPolygon implements Comparable<MatchPolygon> {
    double val;
    Polygon polygon;

    public MatchPolygon(double val, Polygon poly) {
        this.val = val;
        this.polygon = poly;
    }

    public int compareTo(MatchPolygon o) {
        return Double.compare(o.val, val);
    }
}
