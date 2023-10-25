(Class1)=
# Classification I: The geometric view

In machine learning we need to follow a rigorous methodology to ensure that our deployed solutions work as expected. In our {ref}`Meth1` chapter we discussed three key machine learning tasks, namely test, training and validation. Needless to say, the three tasks require datasets, which are used as surrogates of our target population. The essential role of datasets in machine learning can make us feel that as long as we have datasets, we are good to go. This is however not the case. To illustrate why just having datasets might not take us far, no matter how big our datasets are, we will start this chapter revisiting the famous 1936 *Literary Digest* poll blunder, from which we will extract our fourth top tip.

% The test task is used to assess the future deployment performance of an already built model, training allows us to build solutions and finally validation can help us assess the potential of different families of models.

This chapter is devoted to classification problems and their solutions, which are known as classifiers. As you will remember, classification belongs together with {ref}`Reg` to the family of supervised learning problems defined in {ref}`Intro3`. In this chapter we will follow a geometric approach to explore classification, whereas in the chapter {ref}`Class2` we will follow a complementary approach based on probability concepts. First, we will formulate classification problems and will learn to visualise classifiers. Then, we will present three families of classifiers, namely linear classifiers, decision trees and nearest neighbours. When discussing each family of classifiers we will investigate the mechanisms that they implement to operate on a set of predictors to produce a predicted label. Finally, we will learn how to use datasets to train each family of classifiers.

Before we submerge ourselves in classification, what happened to the *Literary Digest*?


## The demise of the Literary Digest

% Human landscapes are the byproduct of history piling up, folding in and breaking through layers of human activity over the course of years and centuries, even millennia. To the curious mind landscapes are abundant in clues that when read properly, will take us back to the lives of peoples that are long gone. From the shape of the fields surrounding a village, to the name of a river, a street or a town, through the different tonalities displayed by the bricks in an old wall, the trained eye can find plenty of opportunities to gain access to the past, by identifying the many messages that are encoded in our surroundings.

% in its material surrounding.

Bricked-up windows are one of those architectural oddities that we can spot in many 18th and 19th century buildings in Britain. If you find yourself visiting the Science Museum or the Natural History Museum in London, take a moment to walk around the neighbourhood, the London Borough of South Kensington, and look up. You will discover splendid residences that exhibit bricked-up windows next to conventional ones, as in {numref}`WindowTax`. What moved the owners of these residences to boast assumed windows? Was there any aesthetical principle in their minds?


```{figure} images/WindowTax_1.jpg
---
name: WindowTax
scale: 70%
---
Bricked-up windows in the London Borough of South Kensington.
```


To find an explanatation to the existence of bricked-up windows, we need to move away from aesthetics and look somewhere else. In particular we need to look deep inside the machinery of His Majesty's Treasury, where at the end of the 17th century a new charge on wealth was devised. This new charge became known as the Window Tax. The rationale and workings of the Window Tax can be explained easily. Properties of wealthy individuals have many windows, many more the wealthier those individuals are. With this in mind, His Majesty's Treasury realised that to tax wealth they could visit a property, count the number of windows, assign the correct charge based on the window count, and move to the next property. Simple - and effective. Unfortunately, for His Majesty's Treasury at least, soon after the Window Tax was introduced, a corresponding tax-avoidance scheme would follow: the more windows you brick up, the less taxes you pay. Simple - and effective.



The Window Tax sought to solve two problems with one stone - taxing individuals and assessing their wealth. Deciding how much each invidual should pay based on their wealth does not present great difficulties. Wealth assessment, however, is not as trivial a task as it might sound, both because of the difficulty of quantifying what we mean by wealth and because of the costs of conducting it on large populations of individuals, such as an entire country. Counting windows, although imperfect, was a practical way to solve both difficulties. Wealth is not the only information that governments and other agencies have been gathering from individuals for many years to understand the social, economical or political realities of human populations, or to gain insight into their opinions and preferences. The process of gathering information from a group of people to understand better the population that they belong to, is known as *polling* or *surveying*. Amongst the many applications of polling, election forecasting is one of the most exciting and widely known, with US elections undoubtedly attracting the most attention worldwide. Indeed it would be hard to imagine US elections devoid of polls, however not all of them have had the same historical prominence. If there is such a thing as a famous US elections poll, then the one conducted by the *Literary Digest* in 1936 will top the charts.

The 1936 US elections saw the incumbent Democratic candidate Franklin D. Roosevelt facing the Republican challenger Alfred Landon in the middle of the Great Depression. Which of the two candidates were the bets on? To answer this question, the weekly magazine *Literary Digest* conducted one of the **largest polls ever** and based on 2.4 million responses, predicted that Landon would obtain 54% of the popular vote, while Roosevelt would get 41%. Backed by a good track record in election forecasting, the *Literary Digest* poll results sent a clear message: Americans were about to see a change in government. With the results of the *Literary Digest* poll still warm in their hands, the shock felt by many after election day could not have been greater: Roosevelt went on to receive 61% of the vote, while Landon got 39%. This constituted a huge 20% forecast error, despite the large number of respondents. What went so wrong with the *Literary Digest* poll? One interpretation was that the *Literary Digest*'s sampling strategy was ill conceived, as it realied heavily on telephone directories and registers of automobile owners to send ballots. i.e. higher income groups that *supposedly* leant towards the Republican party. A second view is that Republican voters were more willing to participate in the *Literary Digest* poll as they saw it as a protest channel against the government behind the New Deal. Both views, although different, find the reason for such catastrophically wrong prediction in the **lack of representativity** of the otherwise huge sample used by the *Literary Digest*.

In machine learning we use datasets to build and assess solutions, in much the same way as the *Literary Digest* used response ballots to predict the outcome of the 1936 US elections. However as we have just seen, data, no matter how large, should never be taken for granted. So here is our fourth top tip:


```{admonition} Our fourth top tip is:
:class: tip
<h3 style="text-align: center;"><b>Know your data!</b></h3>
```

By this we not only mean that we need to understand what our data samples represent or how they are formatted. In addition to this, it is essential to understand how our datasets have been extracted from our target population. Dataset-first views of machine learning can mislead us into thinking that as long as we have a dataset, we are good to go. Deployment-first views of machine learning encourage us to think about the relationship between our datasets and our populations. Ideally we should be the ones creating our datasets and if this is not possible, at least we should know how our datasets have been created, so that we can assess how far they can take us. Good practice dictates that we should understand our datasets before we start training models. Otherwise we will risk building solutions that appear to **work well when tested**, but actually **perform poorly during deployment**.

Deployment is indeed the final test: if our solutions do not work when deployed, we are out of business. The *Literary Digest*'s reputation suffered heavily after the 1936 US election forecast fiasco. Two years later, they shut down.


% Reference: Survey Methods, Groves et al

% https://www.cambridge.org/core/services/aop-cambridge-core/content/view/E360C38884D77AA8D71555E7AB6B822C/S014555320001035Xa.pdf/president-landon-and-the-1936-literary-digest-poll-were-automobile-and-telephone-owners-to-blame.pdf


## Formulating classification problems

Classification is a supervised learning family of problems where we seek to build a model, known as classifier, that predicts the value of a discrete label using a set of predictors. Each of the values that a label can take on is known as a *class*, hence classification can also be described as the process of assigning a sample whose predictors we know to a class. Note that classification problems are also known as *decision* or *detection* problems in some scientific fields. This will justify some of the terminology that we will come across later in the book.

Examples of classification problems include predicting whether the salary of an individual will be higher or lower than a certain figure based on known demographic attributes, identifyind a hand-written digit in an image or recognising potitive or negative emotions in a fragment of text. Machine learning uses datasets to build such classifiers. Open datasets that can be used to solve classification problems include the [Adult Data Set](http://archive.ics.uci.edu/ml/datasets/Adult) to predict the salary level of individuals based on demographic attributes, the [MNIST database](http://yann.lecun.com/exdb/mnist/) for digit recognition and the [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) for emotion detection in fragments of texts.

Consider the problem of predicting whether the salary of an individual of a known age is greater than, for instance, 50,000 units of a certain currency. In this problem, *age* is the continuous predictor and *salary>50K* the discrete label that we intend to predict. Specifically, *salary>50K* is a binary label whose values can be *Yes/No*, or *True/False*. {numref}`AgeVsSalaryDiscrete` shows a toy dataset consisting of samples described by two attributes, *age* and *salary>50K* that we could use to build a solution for this problem.


```{list-table} A toy dataset registering the age and salary level of a group of individuals
:header-rows: 1
:name: AgeVsSalaryDiscrete

* - ID
  - Age
  - Salary > 50K
* - $S_1$
  - 37
  - True
* - $S_2$
  - 18
  - False
* - $S_3$
  - 66
  - True
* - $S_4$
  - 25
  - False
* - $S_5$
  - 26
  - False
```

### Mathematical notation

Our mathematical notation for classification problems is very similar to the notation that we developed for {ref}`Reg`. Specifically, in a classification problem we have:
- **Set of predictors**: $\boldsymbol{x}$.
- **Label**: $y$.
- **Model**: $f$.
- **Prediction**: $\hat{y}=f(\boldsymbol{x})$.
- **Dataset**: $\{(\boldsymbol{x}_i,y_i): 1\leq i \leq N \}$, where $N$ is the number of samples and $i$ is the sample identifier. The predicted label for item $i$ is $\hat{y}_i$.

{numref}`SupervisedDiagram` illustrates a classifier $f$ as a system that takes a predictor value $x_i$ as an input and produces a prediction $\hat{y}_i$ as an output.

```{figure} images/SupervisedDiagram.svg
---
name: SupervisedDiagram
width: 70%
align: center
---
Machine learning model $f$ as a system that takes a predictor $x_i$ as an input and produces a predicted label $\hat{y}_i$ as an output.
```
The **discrete nature** of labels sets classification appart from regression in a fundamental way: label values are **not ordered**, in other words, we cannot say that one label value is higher or lower than another. We can say whether two discrete values are **equal or not**, but not define an error quantity like the one we used in regression.

%Therefore, to assess the quality of a model on a dataset, we will establish whether the actual label $y_i$ and the predicted label $y_i$ are equal or not.






### Classifiers in the predictor space

In our {ref}`Reg` chapter we visualised datasets and models in the **attribute space**, i.e. in a coordinate system where each axis corresponded to one of the attributes, including predictors and label. By convention, the vertical axis represents the chosen label attribute. Can we visualise datasets used for classification problems in a similar way? To do so, we need to map each one of the values of the discrete label to a numerical value, so that they can be represented on the vertical axis. Let us explore two toy datasets consisting of samples with a discrete attribute that plays the role of label in a classification problem.

The dataset shown in {numref}`BinaryClass1` consists of samples that are described by two attributes, namely *age* (continuous) and *salary>50K* (discrete). The *salary>50K* attribute is the label and can take on two values, *True* or *False*. For the purpose of visualisation, *True* has been mapped to the numerical value 1, whereas *False* has been mapped to 0.


```{figure} images/BinaryClass1.jpg
---
name: BinaryClass1
width: 80%
align: center
---
Visualisation of the *salary>50K* vs *age* dataset in the attribute space, where *salary>50K* is the label and is therefore represented in the vertical axis.
```

{numref}`MulticlassClass1` represents the second toy dataset, which consists of samples described by three attributes, *age* (predictor), *salary* (predictor) and *opinion* (label). Predictors *age* and *salary* are continuous, whereas *opinion* is discrete and can take on the three values *positive*, *indifferent* or *negative*. In this representation, *negative* is mapped to the numerical value 0, *indifferent* to the numerical value 1, and positive to the numerical value 2.


```{figure} images/MulticlassClass1.jpg
---
name: MulticlassClass1
width: 80%
align: center
---
Visualisation of the *opinion* vs *age* and *salary* dataset in the attribute space. The *opinion* atttribute is the label in this case.
```

Representing datasets used for classification as we have done in {numref}`BinaryClass1` and {numref}`MulticlassClass1` can be misleading, as by mapping label values to numerical values we are imposing an irrelevant ordering that does not exist. For instance, *positive* is not greater than *indifferent* or *negative*, however 2 is greater than 1 and 0. An alternative and more convenient visualisation of datasets consists of representing each sample in the **predictor space** instead, using a **different symbol for each label value**. In {numref}`BinaryClass2` the samples from the *salary > 50K* vs *age* dataset are represented using two different symbols that indicate whether *salary > 50K* is *True* or *False*, on a predictor space consisting of one axis representing the *age* predictor.


```{figure} images/BinaryClass2.jpg
---
name: BinaryClass2
width: 80%
align: center
---
Visualisation of the *salary>50K* vs *age* dataset in the 1D predictor space. Label values are represented using different symbols.
```

{numref}`MulticlassClass2` corresponds to the *opinion* versus *age* and *salary* dataset. The predictor space is 2D, as we have two predictors (*age* and *salary*) and *opinion* values are represented using three symbols for the values *positive*, *indifferent* and *negative*.

```{figure} images/MulticlassClass2.jpg
---
name: MulticlassClass2
width: 80%
align: center
---
Visualisation of the *opinion* vs *age* and *salary* dataset in the 2D predictor space. The *opinion* atttribute is represented using three different symbols.
```

Representing datasets in the predictor space will lead to our geometric definition of a classifier. Before providing this definition, try to answer the following question.


```{admonition} Question for you
:class: question1

Consider the problem of predicting the opinion of an individual whose *age* is 50 and *salary* is 60K. Using the dataset plotted in {numref}`MulticlassClass2`, would yo say this individual's opinion is *positive*, *indifferent* or *negative*?

Submit your response here: <a href="https://forms.office.com/e/t3YwtKmssu" target = "_blank">Your Response</a>

```

As usual, the important aspect of this toy question is not the actual answer, but *how you have arrived* to it. One possible answer is as follows. When we identify the location in the predictor space of this individual, we realise that this location is very close to the group of individuals with a *negative* opinion. Hence if we had to bet on our answer, we could say that the individual has a *negative* opinion too. We can follow this process for any new individual: as long as we know their age and salary, i.e. **their location in the predictor space**, we could guess what their opinion could be by looking at the distribution of dataset samples in the predictor space. There might also be *undecidable* locations where you will not be able to decide how to label a new sample, but anywhere else, we would be able to assign samples to one of the three classes with no hesitation.

It is important to note that we have just come up with a **mechanism to classify our samples using a dataset**, in other words, we have just built a *machine learning* classifier. Our classifier is essentially assigning a label to each point in the predictor space. Therefore, if we colour the predictor space according to the label that will be assigned to each point, we could obtain a representation such as the one shown in {numref}`boundariesEx`. This visualisation shows entire regions in the predictor space where all the samples will receive the same label. We could therefore classify a new individual by identifying the region where this new individual is. Incidentally, we have just **visualised a classifier**.


```{figure} images/MulticlassClass2Boundaries.jpg
---
name: boundariesEx
width: 80%
align: center
---
Partition of the prediction space of the *opinion* vs *age* and *salary* into three decision regions. Samples in the same decision region are assigned the same label.
```

So this is our geometric definition of a classifier. A classifier is a **partition of the predictor** space into **decision regions** separated by **decision boundaries**:
- Each decision region is made up of points that are assigned the same label.
- Decision boundaries separate decision regions and by definition, do not belong to any region.

With this definition in mind we can see classification as a problem where we ask ourselves what the **best partition of the predictor space** is. In machine learning, we use datasets to obtain such partitions. The question is, what do we mean by the *best* partition? In order for an answer to this question to be meaningful, we need to provide a notion of quality.


### A basic quality metric

In contrast to regression, where we had different options available to define the quality of a single prediction $\hat{y}_i$ (e.g. the quadratic error $e_i^2$ or the absolute error $|e_i|$), in classification all we can do is check whether a label $y_i$ and its prediction $\hat{y}_i$ are equal or different:

- If $y_i=\hat{y}_i$, we call $\hat{y}_i$ a **true prediction**.
- If $y_i\neq \hat{y}_i$, we call $\hat{y}_i$ a **false prediction**.


Based on this notion of quality for a single prediction, we can define different quality metrics on an entire dataset. The simplest ones are the **empirical accuracy** $\hat{A}$ and the **empirical error** (or misclassification) **rate** $\hat{E}$, that are defined as

$$
\hat{A} = \frac{\text{Number of true predictions}}{N}
$$(eqEmpA)

and

$$
\hat{E} = \frac{\text{Number of false predictions}}{N}
$$(eqEmpE)

where $N$ is the number of samples. Empirical accuracy and error rates take on values between 0 and 1 and are equivalent, as we can obtain one from the other as $\hat{A}= 1 - \hat{E}$.

As an example, {numref}`AgeVsSalaryDiscrete2` shows the prediction $\hat{y}_i$ of a classifier that assigns a sample of predictor $x_i$ to one of the classes following this simple rule:
- If $x_i>50$, $\hat{y}_i$= True
- Otherwise, $\hat{y}_i$= False

The comparison between an actual label $y_i$ and a predicted label $\hat{y}_i$ is also shown in {numref}`AgeVsSalaryDiscrete2`.

```{list-table} The *salary > 50K* vs *age* dataset is represented using the sybols $y$ and $x$. Predictions from a simple model are added, and the comparison between actual and predicted labels are shown.
:header-rows: 1
:name: AgeVsSalaryDiscrete2

* - ID
  - $x$
  - $y$
  - $\hat{y}$
  - $y=\hat{y}$
* - $1$
  - 37
  - True
  - False
  - True
* - $2$
  - 18
  - False
  - False
  - False
* - $3$
  - 66
  - True
  - True
  - True
* - $4$
  - 25
  - False
  - False
  - True
* - $5$
  - 26
  - False  
  - False
  - True
```

The empirical accuracty of this classifier is $\hat{A} = 4/5 = 0.8$ and its empirical error $\hat{E} = 1/5 = 0.2$. As expected, $\hat{A} + \hat{E} = 0.8 + 0.2 = 1$.


Note that we have used the *hat* notation to indicate that both empirical quantities $\hat{A}$ and $\hat{E}$ are estimations of quantities defined on the population. Both quantities are the **true accuracy** $A$ and the **true error rate** $E$:
- The true accuracy $A$ is the fraction of samples from the population that are classified correctly on average.
- The true error rate $E$ is the fraction of samples from the population that are misclassified.

True accuracy $A$ and error rate $E$ are also quantities that take on a value from 0 to 1 and are related by the simple expression $A= 1-E$.


## Basic classifiers

In this section we are going to explore three basic classifiers, namely linear classifiers, decision trees and nearest neighbours. As we are about to see, linear classifiers and decision trees label samples by identifying the decision region where they lie. The nearest neighbours method is different in that it does not use the notion of decision region to classify new samples, although it still produces a partition of the predictor space into decision regions.

Rather than looking at how to train each classifier, in this section our emphasis is on describing the mechanism that these classifiers implement to produce a label based on a set of input predictors. Nearest neighbours is a peculiar method as strickly speaking it is not trained, yet it uses the entire training dataset every time it predicts a label. In a later section, we will describe how to fit linear classifers and decision trees to a training dataset.



### Linear classifiers

Consider a binary classification problem, where labels can take on two values. The simplest decision boundary that we can think of is the linear boundary. In a 1D predictor space a linear boundary is just a single point ({numref}`LinearBoundaries1D`), also known as a threshold, in a 2D predictor space a straight line ({numref}`LinearBoundaries2D`) and in a 3D predictor space a plane ({numref}`LinearBoundaries3D`). In higher dimensions, linear boundaries are called hyperplanes and needless to say, cannot be visualised.

```{figure} images/LinearClass1D.jpg
---
name: LinearBoundaries1D
width: 80%
align: center
---
A linear decision boundaries in a 1D predictor space is one single point.
```

```{figure} images/LinearClass2D.jpg
---
name: LinearBoundaries2D
width: 80%
align: center
---
In a 2D predictor space, a linear decision boundary is a straight line.
```

```{figure} images/LinearClass3D.jpg
---
name: LinearBoundaries3D
width: 80%
align: center
---
In a 3D predictor space, a linear decision boundary is a plane.
```

%Given a linear decision boundary, one of its sides is the $\textcolor{red}{\text{o}}$ decision region, whereas the other side is the $\textcolor{blue}{\text{o}}$ decision region. Therefore, g

Given a linear boundary defining two decision regions, a simple mechanism to classify a new sample is to **identify the decision region where the sample lies**. How can we do this?

Let us first develop the mathematical notation that we need to describe linear decision boundaries. This notation can be used for any number of predictors $K$, i.e. for predictor spaces of any number of dimensions. A linear boundary in the predictor space is defined by the equation

$$
w_0 + w_1 x_{1} + w_2 x_{2} + \dots + w_K x_{K}=0
$$(eqLinearBoundary)

where $x_1$, $x_2$, $\dots$, $x_K$ are the predictors and $w_0$, $w_1$, $w_2$, $\dots$, $w_K$ are the parameters of the linear boundary. These parameters determine the exact location of the linear boundary. What this equation means is, if we are given a set of predictors $x_1$, $x_2$, $\dots$, $x_K$ and their linear combination using $w_0$, $w_1$, $w_2$, $\dots$, $w_K$ is zero, then this set of predictors corresponds to a sample that lies exactly on the linear boundary. Note that $x_1$ refers, using this notation, to the first predictor, rather than the value of the predictor of the first sample. Using bold font notation for sets of predictors will allow us to avoid potential notation ambiguities like this one.

For instance, consider a classification problem with just one predictor $x_1$. The predictor space is 1D in this case and a linear boundary is defined by the equation $w_0 + w_1 x_{1} = 0$. Solving for $x_1$ reveals that when $x_1 = -w_0/w_1$ the equality $w_0 + w_1 x_{1} = 0$ holds. In other words, the single point $x_1 = -w_0/w_1$ is the linear boundary or threshold in this 1D predictor space. Values of $x_1$ greater than $-w_0/w_1$ belong to one class, less than $-w_0/w_1$ to the other class.

In a 2D predictor space, a linear boundary is defined as

$$
w_0 + w_1 x_1 + w_2 x_2 = 0
$$(eqLinearBoundary2D)

Solving for $x_2$ we obtain the more familiar equation for the straight line

$$
x_2 = -\frac{w_0}{w_1} - \frac{w_2}{w_1} x_1
$$(eqLinearBoundary2Dsol)

where $-w_0/w_1$ is the intercept and $-w_2/w_1$ the gradient. Samples on one side of this straight line are assigned to one class, on the other side to another class. The question arises, given a sample with predictor values $x_1$ and $x_2$, how do we know which side it belongs to without plotting it?


Before answering this question, let us obtain an alternative and more compact way to define linear boundaries using basic matrix algebra. Let us start defining the by now familiar-looking parameter vector $\boldsymbol{w}$ and the predictor vector $\boldsymbol{x}$:

$$
\boldsymbol{w}= \begin{bmatrix}
w_0\\
w_{1}\\
w_{2}\\
\vdots \\
w_{K}
\end{bmatrix}, \quad
\boldsymbol{x}= \begin{bmatrix}
1\\
x_{1}\\
x_{2}\\
\vdots \\
x_{K}
\end{bmatrix}
$$(eqVectors)

Using this vector notation, a linear boundary can be expressed as:

$$
\boldsymbol{x}^T\boldsymbol{w}=0
$$(eqVectorsProd)

Note that in addition to being simple, this expression is valid for any number of predictors and eliminates the effort of having to enumerate all of the predictors each time. Any sample described by the set of predictors $\boldsymbol{x}_i$ belongs to the linear boundary defined by the parameters $\boldsymbol{w}$ if and only if $\boldsymbol{x}_i^T\boldsymbol{w}=0$. So what happens to points that are not along the boundary?

By definition, if sample $\boldsymbol{x}_i$ is *not* on the boundary, $\boldsymbol{x}_i^T\boldsymbol{w}\neq 0$. Hence either $\boldsymbol{x}_i^T\boldsymbol{w}> 0$ or $\boldsymbol{x}_i^T\boldsymbol{w}< 0$. Crucially, all the points on the same side of the linear boundary are such that when calculating $\boldsymbol{x}_i^T\boldsymbol{w}$ we always obtain a number of the same sign. In other words:
- All the samples such that $\boldsymbol{x}_i^T\boldsymbol{w}> 0$ lie on one side of the boundary.
- All the samples such that $\boldsymbol{x}_i^T\boldsymbol{w}< 0$ lie on the other side of the boundary.
- All the samples such that $\boldsymbol{x}_i^T\boldsymbol{w} = 0$ lie on the boundary.


This simple observation, which is illustrated in {numref}`LinearBoundaries2DxtW`, will allow us to implement a linear classifier defined by a set of parameters $\boldsymbol{w}$.


```{figure} images/LinearClass2DxTw.jpg
---
name: LinearBoundaries2DxtW
width: 80%
align: center
---
A linear boundary defined by the parameters $\boldsymbol{w}$ separates the predictor space into two regions, such that $\boldsymbol{x}_i^T\boldsymbol{w}> 0$ on one region, $\boldsymbol{x}_i^T\boldsymbol{w}< 0$ on the other region and $\boldsymbol{x}_i^T\boldsymbol{w}= 0$ on the boundary.
```


Our implementation of a linear classifier defined by a set of parameters $\boldsymbol{w}$ is illustrated in {numref}`LinearClassifier` and consists of the following steps. Given a predictor vector $\boldsymbol{x}_i$:
- We first compute $\boldsymbol{x}_i^T\boldsymbol{w}$.
- Then compare this quantity against 0. This comparison is denoted by $\lessgtr 0$.
- If $\boldsymbol{x}_i^T\boldsymbol{w}>0$, the prediction $\hat{y}_i$ is the label value associated with the positive decision region, if $\boldsymbol{x}_i^T\boldsymbol{w}<0$, then $\hat{y}_i$ takes on the value associated to the negative decision region. If $\boldsymbol{x}_i^T\boldsymbol{w}=0$, the sample lies on the decision boundary.

Note that we have just described how a linear classifier assigns labels to new samples using the boundary parameters $\boldsymbol{w}$. What we have not discussed is how the values of these parameters are determined, in other words, how to use a training dataset to tune $\boldsymbol{w}$. A method for traning linear classifiers will be discussed in the {ref}`train_class` section.


```{figure} images/LinearClassifier.svg
---
name: LinearClassifier
width: 70%
align: center
---
Linear classifier
```


Depending on the ability of linear boundaries to *separate* samples from different classes, it is common to describe datasets as **linearly separable** and **non-separable**. {numref}`SeparableDataset` shows an example of a dataset that is linearly separable, as we can find multiple linear boundaries that separate samples from one class from samples from the other class. By definition, linear classifiers can achieve the empirical accuracy $\hat{A}=1$ when applied to a linearly separable dataset.



```{figure} images/LinearlySeparableRegions.jpg
---
name: SeparableDataset
width: 80%
align: center
---
Linearly separable dataset.
```


By contrast, {numref}`NonSeparableDataset1` and {numref}`NonSeparableDataset2` are linearly non-separable datasets. No matter how hard we try, we will never be able to find a straight line that will separate both classes. Consequenyly, we will never find a linear model that will achive the maximum empirical accuracy, and therefore $\hat{A}<1$ or equivalently $\hat{E}>0$. In general, samples from a population will be non-separable if samples overlap in the predictor space (as in {numref}`NonSeparableDataset1`) or when the distribution of samples is complex (as in {numref}`NonSeparableDataset2`). In either case, we need to accept that we will not always achieve perfect results. Remember to **embrace de error!**



```{figure} images/NonLinearlySeparable1Regions.jpg
---
name: NonSeparableDataset1
width: 80%
align: center
---
Linearly non-separable dataset, where samples from different classes overlap in the perdictor space.
```

```{figure} images/NonLinearlySeparable2Regions.jpg
---
name: NonSeparableDataset2
width: 80%
align: center
---
Linearly non-separable dataset due to a complex distribution of samples in the predictor space.
```

In summary, linear classifiers identify the decision region that a sample $\boldsymbol{x}_i$ belongs to by finding out which side of the boundary the sample lies, and this is determined by looking at the sign of the result of the operation $\boldsymbol{x}_i^T\boldsymbol{w}$. What if instead of a binary classification problem we have a multiclass classification problem, what could we do? One option would be to express a multiclass classification problem as multiple binary problems operating in parallel. The decision regions will be more complex than the two semi-planes of a binary classifier, but they will still be separated by straight segments.



### Decision trees

Similarly to linear classifiers, decision trees label samples by identifying the decision regions where they lie. Decision regions in trees are however defined differently. To illustrate decision trees, let us look at the decision regions that we created for the *opinion* vs *salary* and *age* problem ({numref}`boundariesEx`). A simple mechanism to label an individual of known *age* and *salary* using the already defined decision regions would be:
- If the *salary* is greater than 50K, the *opinion* is $\hat{y}_i=\text{negative}$.
- If the *salary* is not greater than 50K we look at the *age*:
  - If the *age* is greater than 40, then $\hat{y}_i=\text{indifferent}$.
  - Otherwise $\hat{y}_i=\text{positive}$.

A useful representation for this simple mechanism is the tree-like structure shown in {numref}`TreeEx`. This structure consists of two types of nodes, namely **branching nodes** and **leaves**:
- In a branching node, the value of **one of the predictors is compared** against a threshold. The result from this comparison determines which node to visit next.
- A leaf node represents a **decision region**. When a leaf node is visited, a label is assigned to the sample, in other words, it produces a prediction.


```{figure} images/TreeEx.jpg
---
name: TreeEx
---
Tree structure for the classifier consisting of the decision regions shown in {numref}`boundariesEx`.
```

In the tree shown in {numref}`TreeEx` there are two branching nodes, one for the *salary* with threshold $50K$ and another one for the *age*, with threshold 40. There are also three leaves, each one representing one of the decision regiones in {numref}`boundariesEx`.

To parse a decision tree, we start from the top branching node (the root), and sequentially navigate the nodes as indicated by the intermediate branching nodes until we reach a leaf. Note that in this navigation we use the value of the input predictors. One peculiarity of decision trees is that their **decision boundaries are axis-parallel**, since branching nodes use one predictor only. In {numref}`boundariesEx`, this can be seen as a horizontal decision boundary and a vertical one. As a consequence of this, the decision regions created by a tree are rectangular-shaped in 2D predictor spaces, cuboids in 3D predictor spaces or, in general, hypercuboids.

The simplest decision tree that we can create is one consisting of one branching node and two leaves. This would produce a partition of the predictor space into two decision regions. Replacing one of the leaves by a branching node followed by two leaves, results in two new decision regions where previously there was only one. As we add more branching nodes, the **complexity of the decision tree increases** and so does the complexity of the partiotioning of the predictor space. The deeper the structure of a tree, the more decision regions we can create in the predictor space. For instance, a tree consisting of one branching level results in 2 decision regions, two levels in up to 4 decision regions, three levels in up to 8, and so on. In other words, deep trees are more flexible than shallow trees.

{numref}`TreeEx2` shows a decision tree consisting of two levels and four leaf nodes defined on a 2D predictor space. The four decision regions defined by this tree are visualised in {numref}`TreeEx2Regions`.


```{figure} images/TreeEx2.jpg
---
name: TreeEx2
---
Two level decision tree consisting of three branching nodes and four leaf nodes.
```

```{figure} images/TreeEx2Regions.jpg
---
name: TreeEx2Regions
width: 80%
align: center
---
Decision regions defined by the tree in {numref}`TreeEx2`.
```

In {numref}`TreeEx3` a three level decision tree resulting from replacing two of the lead nodes of {numref}`TreeEx2` with two branching nodes is shown. The decision regions defined by this tree are visualised in {numref}`TreeEx2`.

```{figure} images/TreeEx3.jpg
---
name: TreeEx3
---
Three level decision tree resulting from replacing two leaf nodes in {numref}`TreeEx2Regions` with two branching nodes.
```

```{figure} images/TreeEx3Regions.jpg
---
name: TreeEx3Regions
width: 80%
align: center
---
Decision regions defined by the tree in {numref}`TreeEx3`.
```


Once again, it is important to highlight that so far we have discussed how trees define decision regions and can be used to classify samples. What we have not discussed is how we can train decision trees, i.e. how we can build decision trees using a training dataset. {numref}`TreeExNonSep` shows the decision boundaries defined by a tree that has been fitted to a linearly non-separable dataset. As we can see, a decision tree that is not much more complex than a linear classifier, can separate well training samples from two classes in an example where a linear classifier would perform poorly. In the {ref}`train_class` section we will discuss how to train decision trees and explore trees of different degrees of flexibility.



```{figure} images/NonLinearlySeparableTree.jpg
---
name: TreeExNonSep
width: 80%
align: center
---
Decision trees can be used to create complex decision boundaries suitable for complex distributions of samples.
```


### Nearest neighbours

Linear classifiers and decision trees are *parametric* models, i.e. they are defined by a set of parameters. The decision regions generated by parameteric models can adopt a predefined range of shapes and by setting the value of their parameters, we materialise one of the predefined shapes. For instance, linear classifiers produce two decision regions separated by linear boundaries and decision trees can only produce rectangular decision regions.

The nearest neighbours family of models, also known as kNN ($k$ nearest neighbours) is by contrast *non-parametric* and because of this, it does not impose on decision regions of any predefined shape. Furthermore, even though like any other classifier kNN models partition the prediction space into decision regions, the mechanism by which samples are classified does not involve identifying the decision regions where samples lie. How do kNN models work?

The mechanism that kNN implements is very simple and in fact, we already suggested it when we considered the problem of guessing the *opinion* of an individual of *age* 50 and *salary* 60K, using the dataset shown in {numref}`MulticlassClass2`. Back then, we decided to classify a new sample based on its proximity to training samples from each class. This is precisely the strategy that kNN implements.

In kNN, $k$ refers to the number of closest training samples that we will use to classify a new sample. kNN proceeds as follows:
- Given a new sample, we calculate its distance to all the samples in the training dataset.
- Then, we identify the closest $k$ training samples (i.e. its $k$ nearest neighbours).
- Finally, we assign the label of the most popular class among its $k$ nearest neighbours.

kNN is a so-called *instance-based* method, as in order for it to be implemented it needs all the instances (or samples) in the training dataset. Consequently, kNN models need to memorise the entire training dataset. By contrast, linear classifiers and decision trees only need the training dataset during training;  once the model has been built the training dataset can be discarded, as all we need to classify a new sample is the set of tuned parameters.

One obvious question is, what is the role of $k$? For a start, the value of $k$ should be chosen to avoid ties, i.e. situations where there is not a majority class. For instance, in a binary classification problem, we would prefer $k$ to be odd, otherwise we would encounter many cases where half of the nearest neighbours of a sample belong to one class and the other half to the other class. The complexity of the decision regions produced by kNN is also related to the value of $k$. Specifically, low values of $k$ result in complex decision boundaries, whereas increasing the value of $k$ results in increasing the rigidity of the boundaries (see {numref}`kNN1`, {numref}`kNN3` and {numref}`kNN15`). Consequently, the risk of overfitting increases for low values of $k$, whereas the risk of underfitting increases for large values of $k$. As you will have concluded, $k$ is a hyperparameter in kNN: changing $k$ leads to solutions of different degrees of complexity. To choose $k$, we can run a validation task, where we compare the validation quality of kNN models for different values of $k$.

```{figure} images/kNN1.jpg
---
name: kNN1
width: 80%
align: center
---
Decision boundaries defined by a kNN model, using $k=1$.
```

```{figure} images/kNN3.jpg
---
name: kNN3
width: 80%
align: center
---
Decision boundaries defined by a kNN model, using $k=3$.
```

```{figure} images/kNN15.jpg
---
name: kNN15
width: 80%
align: center
---
Decision boundaries defined by a kNN model, using where $k=15$.
```

kNN implements a simple and intuitive mechanism to classify samples. However, compared to decision trees and linear classifiers, kNN models are computationally expensive and require to store the entire training dataset to make predictions during deployment.


(train_class)=
## Training classifiers

In {ref}`Meth1_train` section we saw that in machine learning, the parameters of a tunable model are set using a training dataset together with a cost function. We also disscussed how in general, the cost function might be different from our *target quality* during deployment. Our hope is that defining the right cost function will lead us to a model whose target quality during deployment is high.

Linear models and decision trees are tunable models defined by a set of parameters. The parameters that need to be tuned are respectively the parameters that define a linear boundary, and the thresholds defining each branching node in a decision tree. Accuracy $A$ and error rate $E$ can be used as target quality metrics during deployment. The question is, which cost functions could we use during training?

In this section, we first consider a method to train linear classifiers known as logistic regression. Reading the term *regression* in a classification context might feel confusing. As we will see, this method involves building a model for a continuous quantity, hence the term regression. Second, we present a method to train decision trees with a predefined structure and then discuss how to explore tree structures of different complexity.



### Logistic regression

Let us go back to the linearly separable dataset shown in {numref}`SeparableDataset`. We have already discussed that we can obtain many, in fact an infinite number of linear classifiers that will achive the highest empirical accuracty $\hat{A}=1$ on this dataset.

```{admonition} Question for you
:class: question1

Have a look at the three linear boundaries that we have plotted in {numref}`SeparableDatasetLines`. If you had to, which linear boundary would you choose?

Boundary with
1. solid-line
2. dashed-line
3. dotted-line

Submit your response here: <a href="https://forms.office.com/e/euB1HB1Aeq" target = "_blank">Your Response</a>

```


```{figure} images/LinearlySeparableBoundaries.jpg
---
name: SeparableDatasetLines
width: 80%
align: center
---
Linearly separable dataset and three solutions
```

Using the empirical accuracy $\hat{A}$ as our quality metric we should conclude that all of them are equally good, hence we would not have a preference. This is not a desirable situation, as at the end of the day we need to choose one of them. Without any additional criterion, we might just pick one of the solutions at random. You might have felt, however, that one of them looks somehow superior. If this is the case, what made you lean towards one over the other two?

If you felt that one of the solutions was better than the other two, regardless of the fact that they all achieve the highest empirical accuracy, you might have looked at them from a *generalisation* point of view. Specifically, you might have imagined new samples that could be generated during deployment. Two of the solutions are very close to training samples from each class and because of this, both feel a bit risky, as we could imagine that in the future one sample from the class that is close to the boundary could *jump* to the other side just by chance. One of the solutions seems to be half way between both classes and therefore the risk of having samples jumping accidentally to the 'wrong' side seems to be lower.

There is an equivalent way to look at this. Given a classifier, the closer a new sample is to the decision boundary, the less certain we are that the sample belongs to the decision region where it lies, as we would be worried it might have accidentaly jumped over. In contrast, if the sample is very far away from the boundary, our certainty that the sample is in the right decision region increases. From this angle, a linear classifier whose boundary is as far as possible from the training samples should be preferred over another one that is very close to the traning samples. We call this strategy the **keep that boundary away from me!** principle. The question arises, how do we encode this principle in a cost function that we can use to train a linear classifier?

To come up with our new cost function, let us first summarise the main points we have presented:
- A linear classifier is a partition of the predictor space into two decision regions separated by a linear boundary.
- To classify a new sample, we need to identify the side of the boundary where the samples lies.
- The farther away a sample is from the decision boundary, the more certain we are that the sample belongs to the decision region where it lies.
- The closer a sample is to the boundary, the less certain we are that the sample is on the right decision region.

As we can see, our approach is built around two concepts, namely the **distance** between a sample and the linear boundary, and the **certainty** that a sample has been correctly classified.

Let us start computing the distance $d_i$ between a sample $\boldsymbol{x}_i$ and a linear boundary defined by the parameters vector $\boldsymbol{w}$. You might be surprised to learn that this distance we can be computed as

$$
d_i = \boldsymbol{x}_i^T \boldsymbol{w}
$$(eqLinearDis)

You could prove this result using a bit of maths, but there is no need to do it here. The notion of distance encapsulated in {eq}`eqLinearDis` is illustrated in {numref}`LinearBoundaries2DxtWDistances`. As we already know, $d_i$ is positive if $\boldsymbol{x}_i$ is on one side of the boundary and negative if $\boldsymbol{x}_i$ is on the other side of the boundary. It is obviously zero if $\boldsymbol{x}_i$ is on the boundary. This is indeed the fact that we used to design the linear classifier shown in {numref}`LinearClassifier`. From now on, we will call the decision region where $\boldsymbol{x}_i^T \boldsymbol{w} > 0$ the **positive semi-plane**, and the decision region where $\boldsymbol{x}_i^T \boldsymbol{w} < 0$ the **negative semi-plane**. We will also use the label values $+1$ and $-1$ to denote the classes corresponding respectively to the positive and negative semi-planes.


```{figure} images/LinearClass2DxTwDistances.jpg
---
name: LinearBoundaries2DxtWDistances
width: 80%
align: center
---
A notion of distance between a sample $\boldsymbol{x}_i$ and a linear boundary defined by the parameters $\boldsymbol{w}$ can be defined as $d_i =\boldsymbol{x}_i^T \boldsymbol{w}$.
```

Let us now move on to the notion of certainty. A convenient way to quantify this notion is using a quantity in the range 0 to 1, where:
- 1 means we are certain that the sample has been **correctly classified**.
- 0 means we are certain that the sample has **not** been **correctly classified**; in other words, we are certain that the sample belongs to the *other class*.
- 0.5 means we are **uncertain** about the identify of the sample.

Now that we know how to quantify our notion of certainty, our job is to map the distance $d_i$ to this notion of certainty. A convenient choice is the so-called **logistic function**, which is defined as

$$
s(d_i)=\frac{e^{d_i}}{1+e^{d_i}} = \frac{1}{1+e^{-d_i}}
$$(eqLogistic)

The logistic function has a so-called sigmoid shape and is shown in {numref}`Logistic1`. As you can see:
- As the move away from the decision boundary on the positive semi-plane ($d_i \rightarrow \infty$), the value of the logistic function $s(d_i)$ gets closer to 1.
- When we are exactly on the decision voundary ($d_i=0$), $s(d_i)=0.5$.
- As the move away from the decision boundary on the negative semi-plane ($d_i \rightarrow -\infty$), the value of the logistic function $s(d_i)$ gets closer to 0.

```{figure} images/Logistic.jpg
---
name: Logistic1
width: 80%
align: center
---
Logistic function
```

Using the logistic function, we can define two certainties:

- $s(d_i)$ is the certainty that a sample belongs to the $+1$ class.
- $1-s(d_i)$ is the certainty that a sample belongs to the $-1$ class.

For instance, imagine a sample $\boldsymbol{x}_i$ such that $s(d_i)=0.85$. Then:

- Our certainty that $\boldsymbol{x}_i$ belongs to the $+1$ class is 0.85.
- Our certainty that $\boldsymbol{x}_i$ belongs to the $-1$ class is 0.15.

Note that by definition, $s(d_i) + [1-s(d_i)] = 0.85+0.15=1$. Obviously, if $s(d_i) = 0.5$, our certainty is 0.5 for both classes.

%Using the notion of certainty, we could say that **linear classifiers assign samples to the class with the highest certainty**.

 %For the sake of the argument, we will assume that there exist two labels $\textcolor{blue}{\text{o}}$ and $\textcolor{red}{\text{o}}$ and
%- Samples in the positive semi-plane are labelled as $\textcolor{blue}{\text{o}}$. Given a sample $\boldsymbol{x}_i$ whose label is $y_i=\textcolor{blue}{\text{o}}$, the certainty that this sample belongs to the $\textcolor{blue}{\text{o}}$ class is $p(d_i)$.
%- Samples in the negative semi-plane are labelled as $\textcolor{red}{\text{o}}$. Given a sample $\boldsymbol{x}_i$ whose label is $y_i=\textcolor{red}{\text{o}}$, the certainty that this sample belongs to the $\textcolor{red}{\text{o}}$ class is $1-p(d_i)$.

We are finally ready to come up with a cost function that we can use to train linear classifiers. Given a training dataset consisting of a collection of samples with labels $+1$ and $-1$, we know how to quantify how certain a linear classifier is about the actual label of each training sample. We can use the individual certainties to define an overall certainty for the entire training dataset. We call this overall certainty **likelihood** $L$. Since the training dataset consists of **independent samples**, we can compute the likelihood by multiplying the individual certainties for each training sample:

$$
L=\prod_{y_i=-1}\left(1-s(d_i)\right) \prod_{y_i=+1}s(d_i)
$$(eqLikelihood)

In this mathematical expression, the symbol $\prod_{y_i=-1}$ means we are multiplying only those factors $1-s(d_i)$ corresponding to samples whose label is $y_i=-1$ and similarly, $\prod_{y_i=+1}$ means we are multiplying factors $s(d_i)$ where $y_i=+1$. Crucially, the likelihood $L$ will be high if

- $s(d_i)$ is high for $+1$ samples, i.e. **$+1$ samples are in the positive semi-plane and away from the boundary**, and
- $1-s(d_i)$ is high for $-1$ samples, i.e. **$-1$ samples are in the negative semi-plane and away from the boundary**.

Therefore, the linear classifier that presents the highest likelihood $L$ is the one that separates training samples in a way that they are kept overall the **furthest away from the boundary**. We can use the **likelihood $L$ as our cost function** and define the optimal model as the one that produces the highest $L$ value on our training dataset.

In contrast to the empirical accuracy $\hat{A}$, there is only one model for which the likelihood $L$ is maximum. Therefore, the likelihood $L$ leads to a unique solution. From a computational angle, $L$ can be problematic. Since $L$ is defined as the product of many quantities between 0 and 1, $L$ is in general a very small number and can lead to underflow in computing environments, i.e. to our computers not being able to represent such small numbers. This can make $L$ unusuable from a practical point of view. Because of this, it is common to use the **log-likelihood** function $l$, which is the result of computing the logarithm of $L$:

$$
l = \log(L)=\sum_{y_i=-1}\log\left[1-s(d_i)\right] + \sum_{y_i=+1}\log\left[ s(d_i)\right]
$$(eqLogLike)

The solution that presents the highest log-likelihood $l$ turns to be the same as the one that has the highest likelihood $L$ and therefore, instead of using $L$ as our cost function, we can use $l$ without changing our notion of $best$ classifier. Another common choice is the **negative log-likelihood**, $-l$. In this case, the optimal model is the one with the minimum negative log-likelihood.



### Branching, growing and prunning in decision trees

The decision tree family provides models of arbitrary degrees of flexibility. Shallow trees, consisting of a few branching levels, are simple and partition the predictor space into a low number of decision regions. In contrast, deep trees have many branching levels and can partition the prediction space into many decision regions. Deep trees are therefore more flexible than shallow trees and can produce more complex partitions of the predictor space. On the flip side, the risk of overfitting to training datasets is higher in deep trees compared to shallow ones, precisely because of their greater flexibility.

Strictly speaking, training a decision tree involves defining the branching and leaf nodes of a tree whose structure, and thefore flexibility, has been established beforehand. We can also devise algorithms that build a decision tree by exploring tree structures of different complexity. This can be done by adding branching nodes to a current tree, which is known as **growing**, or by collapsing branching nodes into a single leaf node, which is known as **prunning**. Exploring different tree structures is a form of **model selection** and therefore requires validation tasks to identify the best tree structure. In this section, we use the term *tree training* in a loose sense, to refer to both training (in a strict sense) and tree selection. The following notation will be useful to describe approaches to training decision trees:
- $M$ is the **number of decision regions** (leaf nodes). For instance, in a tree consisting of one single branching nodes, $M=2$.
- **Decision regions** are denoted by $R_1$, $R_2$, ..., $R_M$.
- $N_m$ is the **number of training samples** in decision region $R_m$.
- $c_m$ is the **proportion** of training samples belonging to the **majority class** in region $R_m$.

Before discussing how to train a decision tree, let us first turn our attention to the leaves of a tree that has already been fitted to a training dataset. We already know that each leaf represents one individual decision region. The question is, which label value should we assign to this particular decision region? An obvious choice would be, the label value of the majority class among the training samples on this decision region.

If $c_m$ is very closse to 1, the majority of the traning samples in $R_m$ belong to the majority class. In other words, $R_m$ is very pure. Since the label value assigned to $R_m$ is determined by the majority class, a value of $c_m$ close to 1 could be used as an indication that a new sample on $R_m$ indeed belongs to the majority class. Combining all the proportions $c_m$ from all the decision regions, we can obtain a metric that quantifies the quality of the entire tree on the training dataset:

$$
C(R_1, R_2, ..., R_M) = N_1 c_1 + N_2 c_2 + ... + N_M c_M
$$

If the cost $C$ is high, the decision regions defined by this tree are pure. Therefore, we could define the best decision tree as the one with the highest $C$. Now that we have a notion of cost, we need to devise an optimisation procedure to train a decision tree of a given structure.

Let us consider a tree consisting of one single branching node for a binary classification problem on a $K$ dimensional predictor space, where $x_1$, $x_2$, $\dots$, $x_K$ denote the predictors. Training the single branching node involves deciding which predictor $x_k$ is used to partition the space and the value of the threshold $t_k$. In other words, we need to determine the values of $k$ and $t_k$. For each value of $k$ and $t_k$ we will obtain a different partitioning of the predictor space into two regions $R_1$ and $R_2$ with a quality

$$
C(R_1, R_2) = N_1 c_1 + N_2 c_2
$$

The best partition will be then the one that achieves the highest purity $C(R_1, R_2)$. Training this single branching node would involve first finding the best threshold $t_k$ along each predictor $x_k$. After this, we would have $K$ potential partitions, one for each predictor, with a different cost. Comparing the cost of each partition would allow us to select the best partition among the $K$ available choices.

Training a tree structure that has multiple branching levels can be computationally challenging, as it would require exploring all the partitions defined by every possible branching choice across all levels. **Greedy approaches** are optimisation strategies that can considerably reduce the number of choices that are explored, at the expense of producing solutions that might not be globally optimal. In the context of decision tree training, greedy approaches proceed by training each level step by step, starting from the root branching node and progressively training subsequent levels as we navigate the decision tree. When a leaf node is reached, the label value corresponding to the majotiry class is assigned to it. This approach is greedy in the sense that we train each branching node focusing exclusively on the quality of its own local partition, without considering the quality of the global partitioning of the space. This is why this process can end up producing solutions that are globally suboptimal.

There are a few extreme situations where a greedy approach might not find an otherwise trivial-looking solution. The so-called XOR problem, which is illustrated in {numref}`xorclass`, is one of them. In the XOR problem, even though a decision tree exists that creates pure decision regions (such as the two level tree shown in {numref}`xorclasstree`), a greedy approach will in general not be able to find it. The reason is simple: every axis-parallel partition at the root level, will produce two regions that contain approximately 50% of the training samples from each class, i.e. all of them achieve the lowest quality possible. Therefore, the threshold $t_1=5$ defining the root node in {numref}`xorclasstree` will be as bad as any other threshold value, and our greedy approach will be unable to discover this simple tree.

```{figure} images/XORTree.jpg
---
name: xorclass
width: 80%
align: center
---
XOR problem
```

```{figure} images/XORTreeSol.jpg
---
name: xorclasstree
---
XOR problem: tree solution
```

Greedy optimisation can be interpreted as progressive refinements of an already existing tree and naturally lead to tree growing strategies. Starting with a tree that can be as simple as one single branching node and two leaf nodes, we can recursively replace any leaf node with a branching node followed by two leaf nodes, which results in partitioning an existing decision region into two new decision regions. This process produces smaller decision regions of increased purity, eventually covering one training sample only. Growing trees increases therefore the risk of overfitting: the final decision regions are pure in the sense that they contain *training samples* that belong to the same class, however this does not imply that this purity will also be seen during deployment. We can also prune a decision tree by collapsing a branching node into a leaf. This action results in all the leaves hanging from this branching node merging into one leaf. Therefore, after prunning, many small decision regions merge to form one single large decision region. Severe prunning can lead to underfitting, i.e. to tree structures too rigid to model the underlying sample distribution. Tree growing and prunning can be controlled by using validation approaches, which allow us to assess whether an increase (respectively decrease) of complexity resulting from growing (respectively prunning) of a tree leads to improved generalisation.



## Summary and discussion

In a classification problem we seek to build a model, known as a classifier, that predicts the value of a discrete label using a set of predictors as its input. Classifiers can be geometrically described as partitions of the predictor space into decision regions, each of which is associated to one unique label value. Accordingly, samples that lie on the same decision region are always assigned the same label. In machine learning we use datasets to build classifiers. As in any other machine learning problem, we can use training, validation and test tasks in a classification context. Training allows us to define a classifier using a dataset, validation to compare different classifiers and testing to assess the deployment performance of a given classifier. In this chapter, we have presented the accuracy and error rate as target quality metrics, which quantify the proportion of samples that are correctly or incorrectly classified.

Each family of machine learning classifiers can be characterised by the types of decision region that it can produce together with the optimisation problem that it solves during training. Logistic regression produces decision regions separated by linear boundaries and uses a likelihood or equivalently log-likelihood cost function during training to tune the parameters of the linear boundary. Logistic regression is not the only approach to train linear boundaries. Linear support vector machines also produce decision regions separated by linear boundaries, however they solve a different optimisation problem to tune the parameters of the linear boundary. Decision trees produce rectangular-shaped partitions of the predictor space. During training, we set out to produce decision regions that are as pure as possible. In this chapter we have quantified the notion of purity using the proportion of training samples from the majority class in a decision region, but there are other popular options, such as the Gini index or the cross-entropy. Greedy optimisation strategies are commonly used to train decision regions as they can alleviate the computational burden during training. However, greedy approaches will in general produce suboptimal solutions, as they restrict the range of candidate solutions that are explored.

Linear classifiers and decision trees are parametric families of models. The range of shapes that linear classifiers and decision trees can produced are limited to, respectively, semi-planes and rectangles and setting their parameters fixes the  shapes of their decision regions. In contrast, nearest neighbours is a non-parametric family and consequently it is not defined by a set of parameters nor produces decision regions of specific shapes. In addition to this, even though nearest neighbours define implicitely decision regions, they do not classify new samples by identifying the decision regions where they lie. You might be asking yourself, is there a common framework to analyse classifier models as different as nearest neighbours and decision trees? The answer is yes, but we will need to wait until a second chapter on classification ({ref}`Class2`), where we will use probability concepts to put classifiers on solid ground. This probability view will also allow us to establish a connection between target quality metrics, such as accuracy and error rate, and cost functions used during training.

Finally, the notions of flexibility, complexity, generalisation, overfitting and underfitting are as important in classification problems as they were in regression scenarios. Flexible classifiers are capable of producing complex decision boundaries that are required whenever the underlying structure of the population is also complex. However, an excessive flexibility can lead to our models learning irrelevant details of a training dataset, especially when the number of training samples is reduced. Models that are too rigid or too flexible for a given classification problem cannot generalise well, in other words, their performance will be limited during deployment. The former models underfit to the training dataset and are uncapable of capture the underlying structure of the population. The latter models overfit to the training dataset and interpret as true structure details of the training dataset that are irrelevant. Linear models constitute an example of rigid models. In contrast, decision trees and nearest neighbours offer models of different degrees of flexibility. As in regression, we can assess the required flexibility for a given classification problem using validation tasks.



%Selection bias, nonresponse bias

%Market research agencies

%Market research

%Public opinion polling

%Facts vs feelings/opinions/attitudes/subjective states

%Census: entire population -> sample -> probability samples

%Training, test, validation, complexity, overfitting, underfitting, etc

%Gini index and cross-entropy for trees

%In summary, logistic regression is a method to train linear classifiers that uses as a cost function the likelihood or log-likelihood function. It is called *regression* because to build a classifier, we actually fit a continuous quantity, namely the *certainty*. It is *logistic regression* because we model the certainty using the logistic function. Logistic regression is not the only approach to train linear classifiers. We can come up with other cost functions. For instance...
