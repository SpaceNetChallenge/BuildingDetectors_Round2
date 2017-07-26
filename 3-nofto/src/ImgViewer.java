import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;

import javax.swing.JFrame;
import javax.swing.JPanel;

public class ImgViewer extends JFrame {
    private static final long serialVersionUID = -5820105568092949073L;
    private static final Object lock = new Object();
    private static int off = 0;

    public ImgViewer(final BufferedImage img) {
    	this(img,"");
    }
    
    public ImgViewer(final BufferedImage img, String title) {
    	setTitle(title);
        JPanel imagePanel = new JPanel() {
            private static final long serialVersionUID = -7037900892428152902L;

            public void paintComponent(Graphics g) {
                super.paintComponent(g);
                ((Graphics2D) g).setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                int w = img.getWidth();
                int h = img.getHeight();
                if (getWidth() * h > getHeight() * w) {
                    h = getHeight();
                    w = img.getWidth() * getHeight() / img.getHeight();
                } else {
                    w = getWidth();
                    h = img.getHeight() * getWidth() / img.getWidth();
                }
                ((Graphics2D) g).drawImage(img, 0, 0, w, h, null);
            }
        };
        getContentPane().add(imagePanel);
        synchronized (lock) {
            setBounds(0 + off, off, 1620, 860);
            off += 24;
            if (off > 300) off = 0;
        }
        setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        setVisible(true);
    }
}