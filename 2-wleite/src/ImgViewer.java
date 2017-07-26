import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Insets;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;

public class ImgViewer extends JFrame {
	private static final long serialVersionUID = -5820105568092949073L;
	private static final Object lock = new Object();
	private static int off = 0;
	private List<BufferedImage> images = new ArrayList<BufferedImage>();
	private List<String> ids = new ArrayList<String>();
	int idx = 0;

	public ImgViewer(final BufferedImage img) {
		this(img, "");
	}

	public ImgViewer(final BufferedImage img, String title) {
		setTitle(title);
		final ImageViewPanel panel = new ImageViewPanel();
		getContentPane().add(panel);
		synchronized (lock) {
			setBounds(0 * 1920 + off, off, 1620, 860);
			off += 24;
			if (off > 300) off = 0;
		}
		setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		setVisible(true);
		setExtendedState(JFrame.MAXIMIZED_BOTH);
		try {
			Thread.sleep(100);
		} catch (InterruptedException e) {
		}
		panel.setImage(img);
		images.add(img);
		ids.add(title);

		panel.setFocusable(true);
		panel.requestFocusInWindow();
		panel.addKeyListener(new KeyAdapter() {
			public void keyPressed(KeyEvent e) {
				if (e.getKeyCode() == KeyEvent.VK_RIGHT) {
					if (++idx >= images.size()) idx = 0;
					setTitle(ids.get(idx));
					panel.setImage(images.get(idx));
				} else if (e.getKeyCode() == KeyEvent.VK_LEFT) {
					if (--idx < 0) idx = images.size() - 1;
					setTitle(ids.get(idx));
					panel.setImage(images.get(idx));
				}
			}
		});
	}

	public void add(final BufferedImage img, String title) {
		images.add(img);
		ids.add(title);
	}
}

class ImageViewPanel extends JPanel {
	private BufferedImage image = null;
	private static final long serialVersionUID = 5489808946126559253L;
	private JScrollPane scrollPane;
	private double zoomFactor;
	private JPanel imgPanel;

	public ImageViewPanel() {
		super(new BorderLayout());
		imgPanel = new JPanel() {
			private static final long serialVersionUID = -7607117377739877804L;

			public void paintComponent(Graphics g) {
				super.paintComponent(g);
				Graphics2D g2 = (Graphics2D) g;
				g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
				g2.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
				g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
				if (image != null) {
					int w = (int) (image.getWidth() * zoomFactor);
					int h = (int) (image.getHeight() * zoomFactor);
					g2.drawImage(image, (getWidth() - w) / 2, (getHeight() - h) / 2, w, h, null);
				}
			}
		};
		scrollPane = new JScrollPane(imgPanel, JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED, JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
		add(BorderLayout.CENTER, scrollPane);

		imgPanel.addMouseListener(new MouseAdapter() {
			public void mousePressed(MouseEvent evt) {
				if (image == null) return;
				if (evt.getButton() == MouseEvent.BUTTON1) changeZoom(1.25, evt.getPoint());
				else if (evt.getButton() == MouseEvent.BUTTON3) changeZoom(1 / 1.25, evt.getPoint());
			}
		});
	}

	public boolean setImage(BufferedImage img) {
		image = img;
		zoomFactor = 1;
		if (image != null) {
			updateZoomFactor();
			scrollPane.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
			scrollPane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED);
		} else {
			Insets insets = scrollPane.getInsets();
			int sh = ImageViewPanel.this.getHeight() - insets.top - insets.bottom;
			int sw = ImageViewPanel.this.getWidth() - insets.left - insets.right;
			imgPanel.setPreferredSize(new Dimension(sw, sh));
			scrollPane.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
			scrollPane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_NEVER);
			imgPanel.setCursor(Cursor.getDefaultCursor());
		}
		imgPanel.revalidate();
		return image != null;
	}

	public void changeZoom(double factor, Point reference) {
		Point pt = (reference == null) ? new Point((int) imgPanel.getVisibleRect().getCenterX(), (int) imgPanel.getVisibleRect().getCenterY()) : reference;

		int sbx = scrollPane.getHorizontalScrollBar().getValue();
		int sby = scrollPane.getVerticalScrollBar().getValue();

		int iw = (int) (image.getWidth() * zoomFactor);
		int ih = (int) (image.getHeight() * zoomFactor);
		pt.x -= (imgPanel.getWidth() - iw) / 2;
		pt.y -= (imgPanel.getHeight() - ih) / 2;

		if (pt.x < 0) pt.x = 0;
		if (pt.y < 0) pt.y = 0;
		if (pt.x > image.getWidth() * zoomFactor) pt.x = (int) (image.getWidth() * zoomFactor);
		if (pt.y > image.getHeight() * zoomFactor) pt.y = (int) (image.getHeight() * zoomFactor);

		double currFactor = zoomFactor;
		zoomFactor *= factor;
		if (zoomFactor > 10) zoomFactor = 10;
		if (zoomFactor < 0.1) zoomFactor = 0.1;
		double f = zoomFactor / currFactor;
		int w = image == null ? 0 : (int) (image.getWidth() * zoomFactor);
		int h = image == null ? 0 : (int) (image.getHeight() * zoomFactor);
		Dimension d = new Dimension(w, h);
		imgPanel.setPreferredSize(d);
		imgPanel.revalidate();

		for (int i = 0; i < 100; i++) {
			if (scrollPane.isValid()) break;
			scrollPane.validate();
			try {
				Thread.sleep(50);
			} catch (InterruptedException e) {
			}
		}
		sbx += (int) (pt.x * (f - 1));
		sby += (int) (pt.y * (f - 1));
		imgPanel.scrollRectToVisible(new Rectangle(sbx, sby, imgPanel.getVisibleRect().width, imgPanel.getVisibleRect().height));
		imgPanel.repaint();

		scrollPane.getVerticalScrollBar().setUnitIncrement(d.height / 50 + 1);
		scrollPane.getHorizontalScrollBar().setUnitIncrement(d.width / 50 + 1);
	}

	private void updateZoomFactor() {
		Container parent = ImageViewPanel.this;
		Insets insets = parent.getInsets();
		int sw = parent.getWidth() - insets.left - insets.right - 4;
		int sh = parent.getHeight() - insets.top - insets.bottom - 4;

		zoomFactor = Math.min(((double) sw) / image.getWidth(), ((double) sh) / image.getHeight());
		//if (zoomFactor > 1) zoomFactor = 1;

		Dimension d = new Dimension((int) (image.getWidth() * zoomFactor), (int) (image.getHeight() * zoomFactor));
		imgPanel.setPreferredSize(d);
		scrollPane.getVerticalScrollBar().setUnitIncrement(d.height / 50);
		scrollPane.getHorizontalScrollBar().setUnitIncrement(d.width / 50);
	}
}
