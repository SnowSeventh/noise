# Interim report of Wang Peng -- Application of Deep Learning in pricing Structured Products
> Editor: Wang Peng
UID: 3035420027

## Preface
All the work I have done can be approached at github:  [project](https://github.com/qiyexue777/pricing_structured_product). It not only contains all the project code but also have a ***readme file which demonstrates what I have done step by step in details***

## Part 1: About the topic
There are some financial derivatives called ***Structured Products***, which are heavily traded in investment banks. Normally, pricing them is the key process and the accuracy and speed when doing this really mean a lot to these financial institutions. However, most traditional calculation method such as ***Monte Carlo simulation*** cannot meet the demand of the market as they are too slow and the fluctuation are also hard to hedge. So we need a new method and a totally different way to fulfill such kind of work. That is why I want to use deep learning method to price the products.

## Part 2: Outline of my project
The work can be divided into several ***stages***:
1. Key modules of pricing;
2. Implementation of traditional pricing method;
3. Building DBS and realizing train_set and test_set;
4. Recursive neural network;
5. Predicting the underlying price with LSTM;
6. GPU test;
7. Implementation of Web Application;

## Part 3: What I have finished
Now I have finished ***stage 1 - 4***. The accuracy of the neural network calculator is 93.75% when the tolerable error rate is under 5%. Details can be found at [readme](https://github.com/qiyexue777/pricing_structured_product).

## Part 4: Schedule of work to be done
Acutally, the most important work have been done especially in ***stage 4***. But I still want to try some different networks such as LSTM, which is excellent in dealing with time series. Anyway, my plan is as follow:
- Finishing part of pricing and the implementation of webpage in September;
- Designing test case and writing paper, which should be done by the end of October;
