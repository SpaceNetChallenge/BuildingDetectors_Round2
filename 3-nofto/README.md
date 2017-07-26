**Marathon Match - Solution Description**

1. **1.**** Introduction**

- **●●** Handle: nofto

1. **2.**** Solution Development**

After the first round, in which I finished 7th, I was sure I cannot beat wleite&#39;s winning solution using my previous approach. Having some experience with wleite&#39;s code from the Lung Cancer Round 2, it was natural to start with his solution also in this match. My idea was to improve it. Sadly, almost nothing I tried was beneficial. Here is the list of things which _did not_ work (i. e., did not show improvement in the provisional score)

- **●●** Changing the returned shapes from convex hulls to non-convex
- **●●** Adding non-convex shapes to the convex hulls (that is, doubling the number of shapes which are processed in the second training phase)
- **●●** Training on multiple cities, that is, trying to build a single model for all cities.
- **●●** Adding new shape features (ratio of original area and convex hull area, distance from the image border, number of detected shapes in the current image, ratio of original perimeter and convex hull perimeter)
- **●●** Adding new &quot;global&quot; pixel features (the original methods rectStatFeatures() and text() applied to the entire image frame)

1. **3.**** Final Approach**

I will describe only the differences between the original wleite&#39;s and my solution. All these are only minor changes (or changes implied by the difference in data format between both rounds).

- **●●** In the first round, there were 2 kinds of images, denoted by _3band_ and _8band_. The problem statement says that in the second round, _RGB-PanSharpen_ corresponds to _3band_ and _MUL_ corresponds to _8band_. However, I found out that better score is obtained when _MUL_ is replaced with _MUL-PanSharpen_ (around 10% gain). Since the resolution of _MUL-PanSharpen_ images is the same as the resolution of _RGB-PanSharpen_ (which was not the case for _8band_ and _3band_ images), it is necessary to adjust lines 28-29 in BuildingFeatureExtractor.java and lines 151-152 in PolygonFeatureExtractor.java.
- **●●** The original solution uses entropy as impurity function for splitting nodes in random forest. I replaced it with sqrt(x\*(1-x)) – it should be computationally less intensive and in the past I got better results in several problems with this function when compared to entropy and/or Gini impurity. In this problem, the results were only slightly better with my function (1% gain), and I only compared it on one city.
- **●●** The original solution uses 60% of data for the first phase of the training, 30% for the second phase of the training and 10% for offline testing. There is no sense to leave any data for offline testing in the final tests, so my split is 65/35/0 instead of 60/30/10.
- **●●** I believe there is a small bug on line 89 of the original Util.java (line 92 of my version) – there should be &quot;buildingsPolyBorder2&quot; instead of &quot;buildingsPolyBorder&quot;.
- **●●** I change the code so that it fulfils the requirements for final testing.

1. **4.**** Open Source Resources, Frameworks and Libraries**

My solution does not use any library or open source resource different from the one used by wleite in the first round. The jar files located in the &quot;lib&quot; directory are probably needed only to work with TIFF format, and I do not know its origin – I downloaded it as a part of the first-round winning solution.

1. **5.**** Potential Algorithm Improvements**

I have no clear idea how the approach may be improved. I tried several things (mentioned above) which did not help.

1. **6.**** Algorithm Limitations**

The results are good only if you test on the model which was trained on the same city. If you will detect buildings on a new city without training data, only with one of the four city models, the results will be poor.

1. **7.**** Deployment Guide**

Follow the guide from the wleite&#39;s solution of the first match. I did not add any new source files, I only changed some of the original files.

1. **8.**** Final Verification**

1. **1.** Create directory structure as in the zip file in [https://www.dropbox.com/s/iov4wsgmutxt7ko/nofto-docker.zip?dl=1](https://www.dropbox.com/s/iov4wsgmutxt7ko/nofto-docker.zip?dl=1)
2. **2.** Execute &quot;BuildingDetectorTrainer &lt;directory&gt;&quot;, which will produce a serialized random forest files named &quot;rfBuilding.dat&quot;, &quot;rfBorder.dat&quot; and &quot;rfDist.dat&quot; in the corresponding models/city&lt;n&gt; directory.
3. **3.** Execute &quot;PolygonMatcherTrainer &lt;directory&gt;&quot;, which will produce a serialized random forest file named &quot;rfPolyMatch.dat&quot; in the corresponding models/city&lt;n&gt; directory.
4. **4.** Execute &quot;BuildingDetectorTester &lt;directory&gt; &lt;output file&gt;&quot;, which will produce the expected CSV file.