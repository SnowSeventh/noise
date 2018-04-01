<script type="text/javascript" async
src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
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

b) $confidence = s(b\bigcup d)/s(b) = 3/5 = 0.6, sufficient$

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

  apparently, nothing in this step
## Question 2
1.
- Step 1

    $centroids: p1, p4$

    $d(p1,p2)=1/2, d(p1,p3)=\sqrt{5}/2,d(p1,p5)=4,d(p1,p6)=\sqrt{17},d(p1,p7)=\sqrt{26}$

    $d(p4,p2)=\sqrt{5}/2,d(p4,p3)=1/2,d(p4,p5)=\sqrt{10},d(p4,p6)=3,d(p4,p7)=4$

    $cluster A:p1,p2$

    $cluster B:p4,p3,p5,p6,p7$
- Step 2

    $centroids:A:=\frac{p1+p2}{2}=(0,1/4),B:=\frac{p3+p4+p5+p6+p7}{5}=(3,7/10)$
