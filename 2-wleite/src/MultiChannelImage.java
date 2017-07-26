import java.awt.Color;
import java.awt.image.BufferedImage;
import java.util.Arrays;

public class MultiChannelImage {
	int[] h, s, l, r, g, b, gray, edge;
	int[][] extraBands, extraEdges;
	int width, height, w8, h8;

	public MultiChannelImage(BufferedImage imgPan, BufferedImage img3band, BufferedImage img8band) {
		width = img3band.getWidth();
		height = img3band.getHeight();
		int length = width * height;
		r = new int[length];
		g = new int[length];
		b = new int[length];
		h = new int[length];
		s = new int[length];
		l = new int[length];
		float[] hsl = new float[3];
		int[] rgbPixels = new int[width * height * 3];
		img3band.getRaster().getPixels(0, 0, width, height, rgbPixels);

		int rmax = 1780;
		int gmax = 1630;
		int bmax = 1080;
		int rmin = 0;
		int gmin = 100;
		int bmin = 150;
		int ptr = 0;
		for (int i = 0; i < length; i++) {
			r[i] = rgbPixels[ptr++];
			g[i] = rgbPixels[ptr++];
			b[i] = rgbPixels[ptr++];
		}
		for (int i = 0; i < length; i++) {
			r[i] = range((r[i] - rmin) * 255 / (rmax - rmin));
			g[i] = range((g[i] - gmin) * 255 / (gmax - gmin));
			b[i] = range((b[i] - bmin) * 255 / (bmax - bmin));
		}
		gray = new int[length];
		imgPan.getRaster().getPixels(0, 0, width, height, gray);

		/////////////////////////BLUR/////////////
		for (int channel = 0; channel <= 3; channel++) {
			int[] a = channel == 0 ? r : channel == 1 ? g : channel == 2 ? b : gray;
			int[] c = a.clone();
			for (int y = 1; y < height - 1; y++) {
				int yw = y * width;
				int p1 = c[yw - width];
				int p2 = c[yw];
				int p3 = c[yw + width];
				int p4 = c[yw - width + 1];
				int p5 = c[yw + 1];
				int p6 = c[yw + width + 1];
				for (int x = 1; x < width - 1; x++) {
					int p7 = c[yw - width + x + 1];
					int p8 = c[yw + x + 1];
					int p9 = c[yw + width + x + 1];
					a[yw + x] = (p1 + p3 + p7 + p9 + ((p2 + p4 + p6 + p8) << 1) + (p5 << 2)) >>> 4;
					p1 = p4;
					p2 = p5;
					p3 = p6;
					p4 = p7;
					p5 = p8;
					p6 = p9;
				}
			}
		}

		for (int i = 0; i < length; i++) {
			int rr = r[i];
			int gg = g[i];
			int bb = b[i];
			Color.RGBtoHSB(rr, gg, bb, hsl);
			h[i] = (((int) Math.round(hsl[0] * 999)) + 500) % 1000;
			s[i] = (int) Math.round(hsl[1] * 999);
			l[i] = (int) Math.round(hsl[2] * 999);
		}
		edge = new int[length];
		h8 = img8band.getHeight();
		w8 = img8band.getWidth();
		extraBands = new int[8][w8 * h8];
		extraEdges = new int[8][w8 * h8];
		int[] extraPixels = new int[w8 * h8 * 8];
		img8band.getRaster().getPixels(0, 0, w8, h8, extraPixels);
		ptr = 0;
		int pos = 0;
		for (int i = 0; i < h8; i++) {
			for (int j = 0; j < w8; j++) {
				for (int k = 0; k < 8; k++) {
					extraBands[k][pos] = extraPixels[ptr++];
				}
				pos++;
			}
		}
		for (int i = 0; i < 8; i++) {
			Arrays.fill(extraEdges[i], -1);
			int[] p = extraBands[i];
			int[] e = extraEdges[i];
			for (int y = 1; y < h8 - 1; y++) {
				int off = y * w8 + 1;
				for (int x = 1; x < w8 - 1; x++, off++) {
					int p1 = p[off - 1 - w8];
					int p2 = p[off - 1];
					int p3 = p[off - 1 + w8];
					int p4 = p[off - w8];
					int p5 = p[off + w8];
					int p6 = p[off + 1 - w8];
					int p7 = p[off + 1];
					int p8 = p[off + 1 + w8];
					if (p1 < 0 || p3 < 0 || p6 < 0 || p8 < 0) continue;
					int vert = Math.abs(p1 + 2 * p4 + p6 - p3 - 2 * p5 - p8);
					int horiz = Math.abs(p1 + 2 * p2 + p3 - p6 - 2 * p7 - p8);
					e[off] = (int) Math.sqrt((vert * vert + horiz * horiz) / 2);
				}
			}
		}
		Arrays.fill(edge, -1);
		for (int y = 1; y < height - 1; y++) {
			int off = y * width + 1;
			for (int x = 1; x < width - 1; x++, off++) {
				/*
				 * p1 p4 p6
				 * p2    p7
				 * p3 p5 p8
				 */
				int p1 = gray[off - 1 - width];
				int p2 = gray[off - 1];
				int p3 = gray[off - 1 + width];
				int p4 = gray[off - width];
				int p5 = gray[off + width];
				int p6 = gray[off + 1 - width];
				int p7 = gray[off + 1];
				int p8 = gray[off + 1 + width];
				if (p1 < 0 || p3 < 0 || p6 < 0 || p8 < 0) continue;
				int vert = Math.abs(p1 + 2 * p4 + p6 - p3 - 2 * p5 - p8);
				int horiz = Math.abs(p1 + 2 * p2 + p3 - p6 - 2 * p7 - p8);
				edge[off] = (int) Math.sqrt((vert * vert + horiz * horiz) / 2);
			}
		}
	}

	public int getRGB(int x, int y) {
		int p = y * width + x;
		return (r[p] << 16) | (g[p] << 8) | b[p];
	}

	private static final int range(int v) {
		return v < 0 ? 0 : v > 255 ? 255 : v;
	}
}