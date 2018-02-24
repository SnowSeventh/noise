# Topic of Dissertation
## Background of myself
My name is Wang Peng from MSc Computer Science. I graduate from Nanjing University as an undergraduate and have studied Physics during that period till 2016. Between 2016 and 2017, I work for a Fin-Tech startup company in Shanghai for almost one year and is assigned to do some work about pricing model of financial derivatives such as options. The goal of my former company is to develop a software or a platform that can help some big financial company in the mainland to do some trade management. It is just this experience making me realize that the idea of computer science and coding ability are really important. I should do some thing to catch up because I know I do need it in the future. So I come to HKU.
## Application of NLP in Finance
### - Scenario A: Helping the Investment Bank to analyse documemnts (Background and Objectives)
The income of Investment Bank may mainly come from two parts as far as I know. One is the commission when they help other company list the securities on the stocks market(简而言之：上市佣金);The other part seems more complicated and please allow me to explain with a small example:

**Suppose company *M* has been allowed to list *1000* shares on the stocks market by Financial Regulation Authority with the help of Investment Bank *q*. Notice that it is not the end of the story because the retail investors still cannot buy or sell the stocks of company *M* in the market. According to the rules, the new stocks of one company should be sold to the third party(usually big financial company such as Investment Banks) first and then retail investors can buy this stocks from the third party(中文中称之为做市商). In this example, the third party contains 4 Investment Banks:*q,w,e and r*. Details are shown below in the table:**

| company| Shares   |
| ------ |:--------:|
| q      | 400      |
| w      | 300      |
| e      | 200      |  
| r      | 100      |
In the mainland, if a company can pass the rules and be listed in the stocks market, it usually means stocks of this company are valuable, which are worth fighting for by many Investment Banks. The Investment Banks will earn a lot during the first several trading days as long as they are holding the stocks of this company. Besides this profit, the Investment Bank can also get commission as compensation of risk because Investment Banks mush use their own money to buy all the shares when being the third party. The No. of shares they want to hold depends on how they look as the risk of this company. In the example we talked above, Investment Bank *q* apparently thinks highly of this stock while Investment Bank *r* may have totally different view from *q*. Now, we are clear about how Investment Banks earn money.
Everyday there are many company under the process of being listed while more companys are fighting for the opportunity to be listed. We are hard to get such kind of information but it is easy for the Investment Banks. Among these companys, the Investment Banks should find out their potential customers accuratly and fast because those are their chance to make money and they should make preparation. Here comes the question: **How to determine whether a company can be listed successfully or not?**

Nowadays, such kind of work is done by experienced stuffs of Investment Banks. They usually read the Application Documemnts of these companys which contain hundreds of pages and cover all the information since one company is started. With the professional knowledge, the stuffs of Investment Bank can often find the answer of the question. However, as the workload is getting higher and higher as well as more and more repetitive work, most of them say that it is enough of it. They really need some useful tools to assist them analysing so many documemnts.
#### - Solution
In this case, I want to use NLP to analyse the documemnts of *CSRC(China Securities Regulatory Commission)*:

![Authority](authroity.png)

All the documemnts are writen in Chinese like the picture below:

![React](react.png)

As the documemnts are usually long, it is not possible to analyse all the information of them. To solve this issue, I may ask some of friends who are working for Investment Bank now to give me a ** Dictionary that contains some key words used by professional stuffs for judgement**. Then use RNN to analyse some words before and after the key word. The target is to get a conclusion for reference.

### - Scenario B: Helping the trader to analyse the news of underlying (Background and Objectives)
Suppose a trader from hedge fund is holding 100 different stocks in the market which are traded heavily most time and contains some real estate stocks. It is not hard to imagine what it is like when he is working: staring at 6 or 9 screens dealing with some buy and sell orders with no time doing anything else. Suddenly at 10:00 a.m., here comes a news about banks: **The central bank is now forcing commerical banks to increase borrow rate of loan by 10bp**. Obviously, the price of all real estate stocks will drop sharply with this news and every trader knows how to react at that time if knowing this piece of news. However, the trader in our example does nothing and lose a lot because he has no time to get and analyse this piece of news.
#### - Solution
In this case, there are 2 goals we need to fulfill:
- Post the news of related stocks to traders at trade time quickly;
- It is far from enough finishing the goal above. Because the trader mostly do not have time to read and analyse the news, we should also give them a reference conclusion of the trend: up, down or keep still.

Same as Scenario A, I also want to use NLP to analyse some Chinese character. However, text is much much shorter than before like the picture shown below:

![news](news.png)

And all the news can be found at some websites like:

[财联社](https://www.cailianpress.com/);

[华尔街见闻](https://wallstreetcn.com/live/global?from=navbar);

### - Deliverables
For both Scenarios, I intend to build an online platform that all the functions and data are stored in a web sever while the user interface is a webpage like this:

![webpage](Structure of Webpage.png)

- For Scenario A, users can upload documemnts to this platform and read the documemnts in '1.Display'. They can also input some key words in '2.input'. Then, the result is shown in '3.Output' after click a button.
- For Scenario B, trader can input the name of stocks in '2.Input'. Then the news will be displayed in '1.Display' and the analyse can be found at '3.Output'.

### - Challenges
Actually, there so many challenges:
1. Firstly it is myself. As I mentioned above, computer science is my weak point. To be honest, it is just 3 weeks since I understand the conception of NN. To fulfill this task, I definitely need lots of time and do much self-study as well as asking for help from others.
2. Besides, there may be some problems of technology. For example, what if the documents contain a table for Scenario A. And is there a better way to get the news immediately rather than resolving from the webpage with Python. Last but not least, the time and space complexity can also drive me crizy.


## **PS**
- I have to say thank you so much if you can reach here patiently and do not delete this file.
- Actually, before I start to write or even after I finish this, I still feel a little confused. According to my background, I should do something about financial application. So I just write these topic down to you. However, when I read the topic description of you as well as other teachers, I find that almost all the titles emphasize 'financial computing' which I suppose had better contain some trade strategy and algo for trade. I do not know whether this feeling of mine is true or not. If so, I can also switch to some other topics in my mind about trade and discuss with you.
- I do not have too much work experience and not have deep understanding about this area. But the problems in the 2 scenarios mentioned above do occur more and more frequently in the mainland everyday which really worth for attention especially when there are almost no soluions done by others before for them.
- At last, thanks so much again. Hope that I am fortunate enough and can be your student.
