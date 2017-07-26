import java.awt.Polygon;
import java.awt.geom.Area;
import java.util.ArrayList;
import java.util.List;

public class Building {
	final int id;
	final List<Polygon> in = new ArrayList<Polygon>();
	final List<Polygon> out = new ArrayList<Polygon>();
	private Area area;
	private double areaVal = -1;
	// private Polygon rp;

	public Building(int id) {
		this.id = id;
	}

	/*
	 * public Polygon getRectPolygon() { if (rp == null) rp =
	 * Util.toRect(in.get(0)); return rp; }
	 */
	public Area getArea() {
		if (area == null) {
			area = new Area();
			for (Polygon p : in) {
				area.add(new Area(p));
			}
			for (Polygon p : out) {
				area.subtract(new Area(p));
			}
		}
		return area;
	}

	public double getAreaVal() {
		if (areaVal == -1)
			areaVal = Util.areaVal(getArea());
		return areaVal;
	}
}
