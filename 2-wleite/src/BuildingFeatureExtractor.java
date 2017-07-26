public class BuildingFeatureExtractor {
	public static final int numFeatures = 68;

	public static float[] getFeatures(MultiChannelImage image, int sx, int sy, int w, int h) {
		float[] ret = new float[numFeatures];
		int k = 0;
		for (int i : new int[] {0,1,3,5}) {
			int rx = sx - i;
			int ry = sy - i;
			int rs = i * 2 + 1;
			int p = i == 0 ? 1 : 3;
			System.arraycopy(rectStatFeatures(image.h, rx, ry, rs, rs, w, h), 0, ret, k, p);
			k += p;
			System.arraycopy(rectStatFeatures(image.s, rx, ry, rs, rs, w, h), 0, ret, k, p);
			k += p;
			System.arraycopy(rectStatFeatures(image.gray, rx, ry, rs, rs, w, h), 0, ret, k, p);
			k += p;
			System.arraycopy(rectStatFeatures(image.edge, rx, ry, rs, rs, w, h), 0, ret, k, p);
			k += p;
			if (i > 0) {
				p = 2;
				System.arraycopy(text(image.gray, rx, ry, rs, rs, w, h), 0, ret, k, p);
				k += p;
				System.arraycopy(text(image.edge, rx, ry, rs, rs, w, h), 0, ret, k, p);
				k += p;
			}
		}
		int w8 = image.w8;
		int h8 = image.h8;
		int rx = sx / 4;
		int ry = sy / 4;
		int p = rx + ry * w8;
		if (ry < 0 || rx < 0 || rx >= w8 || ry >= h8) {
			for (int i = 0; i < 16; i++) {
				ret[k++] = -1;
			}
		} else {
			for (int i = 0; i < 8; i++) {
				ret[k++] = image.extraBands[i][p];
				ret[k++] = image.extraEdges[i][p];
			}
		}
		return ret;
	}

	private static float[] text(int[] arr, int rx, int ry, int rw, int rh, int width, int height) {
		final int[] cnt1 = new int[256];
		final int[] cnt2 = new int[256];
		final int[] dxy = new int[] {-width - 1,-1,width - 1,width,width + 1,1,-width + 1,-width};
		int max1 = 0;
		int max2 = 0;
		int ret1 = -1;
		int ret2 = -1;
		for (int y = ry; y < ry + rh; y++) {
			if (y < 1 || y >= height - 1) continue;
			int yw = y * width;
			for (int x = rx; x < rx + rw; x++) {
				if (x < 1 || x >= width - 1) continue;
				int m = 0;
				int s = 0;
				int c = yw + x;
				int ac = arr[c];
				int aq = arr[c + dxy[7]];
				for (int i = 0; i < 8; i++) {
					int ap = arr[c + dxy[i]];
					if (ap > aq) m |= 1 << i;
					if (ap > ac) s |= 1 << i;
					aq = ap;
				}
				if (++cnt1[m] > max1) max1 = cnt1[ret1 = m];
				if (++cnt2[s] > max2) max2 = cnt2[ret2 = s];
			}
		}
		return new float[] {ret1,ret2};
	}

	private static float[] rectStatFeatures(int[] a, int rx, int ry, int rw, int rh, int width, int height) {
		double sum = 0;
		double sumSquares = 0;
		double sumCubes = 0;
		int cnt = 0;
		for (int y = ry; y < ry + rh; y++) {
			if (y < 0 || y >= height) continue;
			for (int x = rx; x < rx + rw; x++) {
				if (x < 0 || x >= width) continue;
				int off = y * width + x;
				double p = a[off];
				sum += p;
				double pp = p * p;
				sumSquares += pp;
				sumCubes += pp * p;
				cnt++;
			}
		}
		float[] ret = new float[3];
		if (cnt > 0) {
			double k3 = (sumCubes - 3 * sumSquares * sum / cnt + 2 * sum * sum * sum / cnt / cnt) / cnt;
			double k2 = (sumSquares - sum * sum / cnt) / cnt;
			ret[0] = (float) (sum / cnt);
			ret[1] = (float) k2;
			ret[2] = (float) (k2 == 0 ? 0 : k3 / Math.pow(k2, 1.5));
		}
		return ret;
	}
}
