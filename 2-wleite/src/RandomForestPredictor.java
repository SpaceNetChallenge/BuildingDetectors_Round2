
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class RandomForestPredictor {
	private final int[] roots;
	private final int[] nodeLeft;
	private final short[] splitFeature;
	private final float[] value;
	private int trees, free;

	public RandomForestPredictor(int trees, int nodes, boolean max) {
		if (max) {
			roots = new int[trees];
			int totalNodes = trees * nodes;
			nodeLeft = new int[totalNodes];
			splitFeature = new short[totalNodes];
			value = new float[totalNodes];
		} else {
			this.trees = trees;
			this.free = nodes;
			roots = new int[trees];
			nodeLeft = new int[nodes];
			splitFeature = new short[nodes];
			value = new float[nodes];
		}
	}

	public synchronized void add(ClassificationNode root) {
		int rt = roots[trees++] = free;
		free++;
		expand(root, rt);
	}

	public int size() {
		return trees;
	}

	private void expand(ClassificationNode node, int pos) {
		if (node.left == null) {
			nodeLeft[pos] = -1;
			splitFeature[pos] = -1;
			value[pos] = node.getValue();
		} else {
			int l = nodeLeft[pos] = free;
			free += 2;
			splitFeature[pos] = (short) node.splitFeature;
			value[pos] = node.splitVal;
			expand(node.left, l);
			expand(node.right, l + 1);
		}
	}

	public double predict(float[] features) {
		double ret = 0;
		for (int root : roots) {
			ret += classify(root, features);
		}
		return ret / roots.length;
	}

	private double classify(int pos, float[] features) {
		while (true) {
			int sf = splitFeature[pos];
			if (sf < 0) return value[pos];
			if (features[sf] < value[pos]) pos = nodeLeft[pos];
			else pos = nodeLeft[pos] + 1;
		}
	}

	public void save(File file) throws Exception {
		OutputStream out = new GZIPOutputStream(new FileOutputStream(file), 1 << 20);
		byte[] bytes = new byte[trees * 4 + 8];
		int cnt = 0;
		bytes[cnt++] = (byte) ((trees >>> 24) & 0xFF);
		bytes[cnt++] = (byte) ((trees >>> 16) & 0xFF);
		bytes[cnt++] = (byte) ((trees >>> 8) & 0xFF);
		bytes[cnt++] = (byte) ((trees >>> 0) & 0xFF);
		for (int i = 0; i < trees; i++) {
			int ri = roots[i];
			bytes[cnt++] = (byte) ((ri >>> 24) & 0xFF);
			bytes[cnt++] = (byte) ((ri >>> 16) & 0xFF);
			bytes[cnt++] = (byte) ((ri >>> 8) & 0xFF);
			bytes[cnt++] = (byte) ((ri >>> 0) & 0xFF);
		}
		bytes[cnt++] = (byte) ((free >>> 24) & 0xFF);
		bytes[cnt++] = (byte) ((free >>> 16) & 0xFF);
		bytes[cnt++] = (byte) ((free >>> 8) & 0xFF);
		bytes[cnt++] = (byte) ((free >>> 0) & 0xFF);
		out.write(bytes, 0, cnt);
		for (int i = 0; i < free; i++) {
			cnt = 0;
			int si = splitFeature[i];
			bytes[cnt++] = (byte) ((si >>> 8) & 0xFF);
			bytes[cnt++] = (byte) ((si >>> 0) & 0xFF);
			int ni = nodeLeft[i];
			bytes[cnt++] = (byte) ((ni >>> 24) & 0xFF);
			bytes[cnt++] = (byte) ((ni >>> 16) & 0xFF);
			bytes[cnt++] = (byte) ((ni >>> 8) & 0xFF);
			bytes[cnt++] = (byte) ((ni >>> 0) & 0xFF);
			int fi = Float.floatToIntBits(value[i]);
			bytes[cnt++] = (byte) ((fi >>> 24) & 0xFF);
			bytes[cnt++] = (byte) ((fi >>> 16) & 0xFF);
			bytes[cnt++] = (byte) ((fi >>> 8) & 0xFF);
			bytes[cnt++] = (byte) ((fi >>> 0) & 0xFF);
			out.write(bytes, 0, cnt);
		}
		out.close();
	}

	public static RandomForestPredictor load(File file) throws Exception {
		InputStream in = new GZIPInputStream(new FileInputStream(file), 1 << 20);
		byte[] bytes = new byte[4];
		in.read(bytes);
		int trees = (((bytes[0] & 0xFF) << 24) + ((bytes[1] & 0xFF) << 16) + ((bytes[2] & 0xFF) << 8) + ((bytes[3] & 0xFF) << 0));
		int[] t = new int[trees];
		for (int i = 0; i < trees; i++) {
			in.read(bytes);
			t[i] = (((bytes[0] & 0xFF) << 24) + ((bytes[1] & 0xFF) << 16) + ((bytes[2] & 0xFF) << 8) + ((bytes[3] & 0xFF) << 0));
		}
		in.read(bytes);
		int nodes = (((bytes[0] & 0xFF) << 24) + ((bytes[1] & 0xFF) << 16) + ((bytes[2] & 0xFF) << 8) + ((bytes[3] & 0xFF) << 0));
		RandomForestPredictor predictor = new RandomForestPredictor(trees, nodes, false);
		System.arraycopy(t, 0, predictor.roots, 0, trees);
		bytes = new byte[10];
		for (int i = 0; i < nodes; i++) {
			int cnt = in.read(bytes);
			while (cnt != 10) {
				int add = in.read(bytes, cnt, 10 - cnt);
				cnt += add;
			}
			predictor.splitFeature[i] = (short) (((bytes[0] & 0xFF) << 8) + ((bytes[1] & 0xFF) << 0));
			predictor.nodeLeft[i] = (((bytes[2] & 0xFF) << 24) + ((bytes[3] & 0xFF) << 16) + ((bytes[4] & 0xFF) << 8) + ((bytes[5] & 0xFF) << 0));
			predictor.value[i] = Float.intBitsToFloat((((bytes[6] & 0xFF) << 24) + ((bytes[7] & 0xFF) << 16) + ((bytes[8] & 0xFF) << 8) + ((bytes[9] & 0xFF) << 0)));
		}
		in.close();
		return predictor;
	}
}