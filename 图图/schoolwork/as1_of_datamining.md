# AS1
                                                                  Editor: Wang Peng

                                                                  UID:3035420027
***
## Question 1
a) $threshold=3$
- Pass 1:

    |  | a | b | c | d | e |
    |--|---|---|---|---|---|
    |counter|5|5|2|3|2|
    $frequent -items: a,b,d$

- Pass 2:

  We will not consider c and e this time.

  ||ab|ad|bd|
  |-|-|-|-|
  |counter|4|2|3|
  $frequent -items: ab, bd$

- Pass 3:

  No result because abd contains ad, which is not frequent item according to Pass 2.

b) $confidence = \frac{s(b\bigcup d)}{s(b)} = 3/5 = 0.6, sufficient$

c) $threshold = 2$

- Pass 1:


  ||1|2|3|4|5|6|7|                  
  |-|-|-|-|-|-|-|-|
  |counter of item|2|2|2|2|1|1|2|  

  ||0|1|2|
  |-|-|-|-|
  |counter of bucket|4|3|1|
  $frequent-items: 1, 2, 3, 4, 7$
- Pass 2:

  We don't consider anything about 5 and 6 as they are not frequent; Also, pair of (1,4) is out because their hash value is 2, which is not frequent.

  ||(1,3)|(3,4)|(2,7)|
  |-|-|-|-|
  |counter of item|1|1|2|
  |hash value|1|1|0|
  $frequent -items: (2,7)$
- Pass 3:

  Apparently, nothing in this step
## Question 2
1. K-Means
- Step 1

    $centroids: p1, p4$

    $d(p1,p2)=1/2, d(p1,p3)=\sqrt{5}/2,d(p1,p5)=4,d(p1,p6)=\sqrt{17},d(p1,p7)=\sqrt{26}$

    $d(p4,p2)=\sqrt{5}/2,d(p4,p3)=1/2,d(p4,p5)=\sqrt{10},d(p4,p6)=3,d(p4,p7)=4$

    $cluster A:p1,p2$

    $cluster B:p4,p3,p5,p6,p7$
- Step 2

    $centroids:A:=\frac{p1+p2}{2}=(0,1/4),B:=\frac{p3+p4+p5+p6+p7}{5}=(3,7/10)$

    $d(A,p1)=1/4,d(A,p2)=1/4,d(A,p3)=\sqrt{17}/4,d(A,p4)=5/4,d(A,p5)=\sqrt{257}/4,d(A,p6)=\sqrt{265}/4,d(A,p7)=\sqrt{634}/4$

    $d(B,p1)=3.08,d(B,p2)=3.01,d(B,p3)=2.01,d(B,p4)=2.02,d(B,p5)=1.22,d(B,p6)=1.04,d(B,p7)=2.02$

    $clusterA:p1,p2,p4$

    $clusterB:p3,p5,p6,p7$

- Step 3

    $centroids:A:=\frac{p1+p2+p4}{3}=(1/3,1/2),B:=\frac{p3+p5+p6+p7}{4}=(7/2,5/8)$

    $d(A,p1)=0.6,d(A,p2)=0.33,d(A,p3)=0.67,d(A,p4)=0.83,d(A,p5)=3.7,d(A,p6)=3.7,d(A,p7)=4.69$

    $d(B,p1)=3.56,d(B,p2)=3.5,d(B,p3)=2.5,d(B,p4)=2.53,d(B,p5)=0.8,d(B,p6)=0.625,d(B,p7)=1.55$

    $clusterA:p1,p2,p3,p4$

    $clusterB:p5,p6,p7$
- Step 4

  $centroids:A:=\frac{p1+p2+p3+p4}{4}=(2/4,2/4),B:=\frac{p5+p6+p7}{3}=(13/3,2/3)$

  $d(A,p1)=0.71,d(A,p2)=0.5,d(A,p3)=0.5,d(A,p4)=0.71,d(A,p5)=3.54,d(A,p6)=3.54,d(A,p7)=4.53$

  $d(B,p1)=4.38,d(B,p2)=4.34,d(B,p3)=3.34,d(B,p4)=3.35,d(B,p5)=0.75,d(B,p6)=0.47,d(B,p7)=0.75$

  $clusterA:p1,p2,p3,p4$

  $clusterB:p5,p6,p7$

  Now, the algorithm terminates and centroids will not change.

2. Empty Cluster

    Suppose there are 7 points:  

    $p1=(1,11/4),p2=(1,5/2),p3=(1/2,0),p4=(2,3),p5=(5,3),p6=(6,2),p7=(1,3)$

    Use K-means to classfy these data. This time, $K=3$, initate centroids as p3, p5, p6 and we will ignore the details of calculation.

  - Step 1

    $centroids: p3,p5,p6$

    $clusterA: p1,p2,p3,p7$

    $clusterB: p4,p5$

    $clusterC: p6$

  - Step 2

    $centroids: A:=\frac{p1+p2+p3+p7}{4}=(7/8,33/16), B:=\frac{p4+p5}{2}=(7/2,3),C:=p6=(6,2)$

    $clusterA: p1,p2,p3,p4,p7$

    $clusterB: \emptyset$

    $clusterC: p5,p6$


    Now the algorithm terminates and an emptyset occurs.
## Question 3
