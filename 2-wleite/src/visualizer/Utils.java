package visualizer;

import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.LineNumberReader;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.List;
import java.util.Vector;

public class Utils {
		
	private static DecimalFormat df; 
	private static DecimalFormat df6; 
	static {
		df = new DecimalFormat("0.000");
		df6 = new DecimalFormat("0.000000");
		DecimalFormatSymbols dfs = new DecimalFormatSymbols();
		dfs.setDecimalSeparator('.');
		df.setDecimalFormatSymbols(dfs);
		df6.setDecimalFormatSymbols(dfs);		
	}

	/**
	 * Pretty print a double
	 */
	public static String f(double d) {
		return df.format(d);
	}
	public static String f6(double d) {
		return df6.format(d).replace('.', ',');
	}
	
	// Gets the lines of a text file at the given path 
	public static List<String> readTextLines(String path) {
		List<String> ret = new Vector<>();
		try {
			InputStream is = new FileInputStream(path);
	        InputStreamReader isr = new InputStreamReader(is, "UTF-8");
	        LineNumberReader lnr = new LineNumberReader(isr);
	        while (true) {
				String line = lnr.readLine();
				if (line == null) break;
				line = line.trim();
				if (line.isEmpty() || line.startsWith("#")) continue;
				ret.add(line);
			}
			lnr.close();
		} 
		catch (Exception e) {
			e.printStackTrace();
		}
		return ret;
	}
}


