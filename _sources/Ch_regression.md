(Reg)=
# Regression

Regression is the first family of machine learning problems that we will study. As you might remember, we have already considered one regression problem, namely that of guessing the heart rate of an animal from its body mass. Our approach to solve it was however mostly intuitive and only allowed us to produce rough guesses by visually inspecting the available dataset. In this chapter we will introduce, formulate and discuss regression problems and techniques rigorously.

% even though we followed the main steps involved in solving regression problems,

The structure of this chapter is as follows. We will start off by offering our second top tip, using Revolutionary France as a backdrop. Then, we will formulate regression problems using mathematical notation. Our mathematical notation will allow us to explore some basic regression models and discuss how to use datasets to build solutions. Finally, we will explore the notions of model flexibility and complexity, and will connect them to the important machine learning concepts of interpretability and generalisation. This will allow us to define the fundamental notions of underfitting and overfitting.


(Reg1)=
## How far is the Equator from the North Pole?

In the last decade of the 18th century, a commision appointed by the French Académie des sciences decided that the distance from the Equator to the North Pole should be exactly 10,000 km. Yes, you have read it correctly, they *decided* it.

How can you decide how long an existing distance should be, surely it must be given? In reality, what the French commission did was define a much needed new unit of length, the **metre**,  and they did so taking the distance from the Equator to the Norh Pole as a reference. This is why in some old textbooks you might read, rather intriguingly, that *one metre is one ten-millionth of the meridian quadrant*. A meridian quadrant is precisely any segment that starts in the Equator and finishes in the North Pole (see {numref}`MeridianQuadrant`). Defining physical units requires stable references and back in the 18th century, the Earth was seen as the most suitable physical object to define a standard unit of length. The definition of the metre has changed over time though and since 1983, we define the metre using the speed of light in vacuum as our stable reference.

% https://en.wikipedia.org/wiki/History_of_the_metre
% https://www.npl.co.uk/resources/the-si-units/current-research

```{figure} images/meridian_quadrant.png
---
name: MeridianQuadrant
scale: 40%
---
The Paris meridian quadrant runs from the North Pole, through Paris, to the Equator and was used by the French Académie des sciences in 1791 to define a new unit of length: the metre.
```

Let us get back to the 1790s. Stating that one metre is one ten-millionth of a meridian quadrant was the easy part of the job entrusted to the French commission. It can be done, and most likely was done, from the comfort of an armchair. The challenge was to actually measure a meridian quadrant accurately. Think about it for a moment, how would you measure a meridian quadrant, let alone immersed in the social and political instability of 1790s Revolutionary France? One of the main concerns of the team appointed to measure the meridian quadrant was their limited ability to produce precise measurements, or in other words, to reduce the measurement *errors*. To overcome this obstacle, one obvious avenue was to improve the existing instruments. Better instruments, more precise measurements. It is in this atmosphere where a second, less-obvious idea to reduce the impact of measurement errors took shape. The great French mathematician Adrien-Marie Legendre used the following words to describe the essence of this idea: *By using this method, a sort of equilibrium is established between the errors which prevents the extremes from prevailing [...] [getting us closer to the] truth.* Legendre called this method least squares (*moindres carrés*, in French) and he published it in 1805. So what is Legendre trying to tell us?


% https://books.google.co.uk/books?id=Ia8WAAAAQAAJ&printsec=frontcover&source=gbs_ge_summary_r&cad=0#v=snippet&q=verite&f=false
% Legendre, A.M. (1805). Nouvelles méthodes pour la détermination des orbites et des comètes. Courcier, Paris. (Appendice sur la méthode des moindres carrés, pp. 72-80)

 What Legendre is suggesting is to deal with errors *mathematically*. Up until 1805, scientists dealt with errors *physically*, by improving their instrumentation. Legendre is telling us that by carrying out the right mathematical operations on our measurements, we can reduce the impact of errors on our final solution. From this point onwards, mathematics provided a second avenue to deal with errors and get closer to the *truth*. Note that to deal with errors mathematically, we need to explicitely account for them. Specifically, we need to represent them in our mathematical formulation. Only by including them in our mathematical formulation, we will be able to devise methods that can deal with them mathematically.

%So here is our second top tip:

```{admonition} So here is our second top tip:
:class: tip
<h3 style="text-align: center;"><b>Embrace the error!</b></h3>
```

%**Embrace the error**!

% we need to accept it. Accepting means that

We might not like them, but errors do exist and we should not pretend they are not there. In other words, errors are first-class citizens in our formulation. This idea constitutes a core principle in machine learning.

Least squares quickly became a cornerstone in science, and from its early applications to geodesy (the science of measuring Earth's geometry) and astronomy (e.g. to determine the orbit of a celestial object), it spread inexorably throughout every branch of science. Least squares is such an important method that science historians still debate today whether it should be credited to Legendre or to another great mathematician, Karl F. Gauss. Incidentally, if you are looking for the origins of machine learning, you will find them exactly here. Least squares is actually a regression method and we will cover it later in this chapter. However, even though we could regard least squares as the first machine learning method, what we want to highlight is not least squares itself, but the revolutionary idea that allowed Legendre to conceive this method. Remember this: if we do not embrace the error, there is no machine learning.

%what we want to reflect on is the actual idea that allowed Legendre to conceive least squares, rather than the method of least squares itself.

%It is everywhere, so much so that you will soon fail to appreciate it is even there.


(Reg2)=
## Formulating regression problems

Regression problems belong to the category of supervised learning problems, where we seek to predict a label using a set of predictors ({numref}`RegressionDiagram`). What distinguishes regression from the other family of supervised learning problems, i.e. classification, is that in regression the label takes on continuous values. Examples of problems where we are interested in predicting a continuous label include predicting the energy consumption of a household, the future value of a company stock, tomorrow's temperature or the probability of developing a specific health condition. As in any other machine learning scenario, regression problems belong to machine learning because their solutions are built using a dataset.

```{figure} images/regression_diagram_nq.svg
---
name: RegressionDiagram
width: 70%
align: center
---

In supervised learning we seek to find a model that predicts a label using a set of predictors. This model is the solution to a supervised learning problem.
```


To illustrate regression, let us consider the problem of predicting the salary of an individual who lives in Paris, of whom we know their age. If the salary of a Parisian was prescribed by their age according to some written law, our job would be finished. We would obtain the salary from the age simply using this law. Unfortunatelly there is not any such written law and hence the question is, is there any relationship between the salary and age of the Parisians? If this is the case, how can we discover this relationship? If such relationship exists, our goal is to build a mathematical model that represents it.

% In other words, using a dataset we can try to build a model that predicts salary from age.

Using a dataset recording the age and the salary of a collection of individuals from Paris, we can try to discover how Parisian salaries are related to ages. {numref}`AgeVsSalary` shows a made-up dataset created for this purpose. Note that this same dataset could have been created to build a model that predicts the age of an individual using their salary as the predictor. It is us who have to decide which attribute is the predictor and which is the label when we formulate a regression problem.

```{list-table} A toy dataset registering the age and salary of a group of individuals
:header-rows: 1
:name: AgeVsSalary

* - ID
  - Age
  - Salary
* - $S_1$
  - 37
  - 68,000
* - $S_2$
  - 18
  - 12,000
* - $S_3$
  - 66
  - 80,000
* - $S_4$
  - 25
  - 45,000
* - $S_5$
  - 26
  - 30,000  
```

### Mathematical notation

In machine learning, our first step is always to formulate our problem *mathematically*. This involves using mathematical notation to represent all the concepts in our problem and their relationships. Let us start with the basic mathematical notation needed to describe our population and dataset:

- **Predictor**: $x$.
- **Label**: $y$.
- **Number of samples** in our dataset: $N$.
- Dataset **sample identifier**: $i$.

Using this notation, the value of the predictor of the $i$-th sample in a dataset can be denoted by $x_i$ and its label by $y_i$. For instance, to report on the predictor and label of the third sample in the dataset shown in {numref}`AgeVsSalary`, we would write $x_3=66$ and $y_3=80,000$, respectively. Remember that when we formulated our problem, we decided that age was the predictor ($x$) and salary the label ($y$).

Furthermore, we can denote our entire dataset by $\{(x_i,y_i): 1\leq i \leq N \}$. Curly brackets '$\{$' and '$\}$' are used to represent the notion of *collection*. With this in mind, the mathematical expression $\{(x_i,y_i): 1\leq i \leq N \}$ should be read as *a collection of pairs of values $(x_i,y_i)$, where $i$ runs from 1 to $N$*. For instance, our dataset in {numref}`AgeVsSalary` can be expressed as a collection of $N=5$ pairs:

$\{(x_i,y_i): 1\leq i \leq 5 \} = \{(x_1,y_1), (x_2,y_2), (x_3,y_3), (x_4,y_4), (x_5,y_5)\}$,

specifically

$\{(x_i,y_i): 1\leq i \leq 5 \} = \{(37, 68000), (18, 12000), (66, 80000), (25, 45000), (26, 30000)\}$.



Now that we have agreed on how to represent basic population and dataset concepts, let us create the notation needed to describe our model:

- **Model**: $f$.
- **Prediction**: $\hat{y}$.

Using this notation, we express the idea of making a prediction as

$$
\hat{y} = f(x)
$$(eqModelNotation)

which should be read as *the model $f$ takes the predictor $x$ as an input and produces the prediction $\hat{y}$ as an output* (see {numref}`FunctionIOBox`).

```{figure} images/function_input_output_box.svg
---
name: FunctionIOBox
---
A supervised learning model $f$ takes a predictor $x$ as an input and produces a prediction $\hat{y}$ as an output. This can be represented as a block diagram and expressed mathematically as $\hat{y} = f(x)$.
```


 Note that our notation explicitly distinguishes between the actual value that we want to predict, $y$, and the prediction provided by our model, $\hat{y}$. This distinction is crucial to define our last concept:

- **Prediction error**: $e$.

The prediction error can be defined as $e= y-\hat{y}$, i.e. as the difference between the actual value that we want to predict and the value that our model predicts.

To consolidate our understanding of the mathematical notation that we have developed, let us apply it to the following example. Assume that we have collected the dataset shown in {numref}`AgeVsSalary` and are using the model $f(x) = 1,000x$ to predict the salary $y$ of an individual given their age $x$. This model simply predicts the salary of an individual to be 1,000 times their age. {numref}`RegresionNotation` provides a visual illustration of the mathematical notation that we have created. First, it represents in the attribute space the five samples, $S_1$ to $S_5$, of the dataset defined in {numref}`AgeVsSalary`. Note that the coordinates of each sample $S_i$ correspond to the values of its attributes $x_i$ and $y_i$. Second, the model $f(x) = 1,000 x$ is plotted as a solid line. The coordinates of each point in the line representing the model correspond to each age value $x$ and its predicted salary $\hat{y}=f(x)$. Finally, the prediction error $e_i$ is represented as a vertical line from each individual sample to the line representing the model, which corresponds to the difference $e_i=y_i-\hat{y}_i$.


```{figure} images/regression_notation.svg
---
name: RegresionNotation
---
Visualisation of the dataset defined in {numref}`AgeVsSalary` toghether with the model $f(x) = 1,000 x$ (solid line). The vertical dashed lines from each sample to the model represent the individual prediction errors.
```

{numref}`AgeVsSalary2` captures the dataset shown in {numref}`AgeVsSalary` together with the predicted labels and the prediction errors of the model $f(x) = 1,000 x$. The predictor value of the first sample in {numref}`AgeVsSalary` is $x_1 = 37$ and its actual label $y_1=68,000$. Using the model $f(x) = 1,000 x$, the predicted label is $\hat{y}_1= f(x_1) = 1,000 \times 37 = 37,000$ and the prediction error is $e_1 = 68,000-37,000=31,000$. You should be able to carry out this process with the remaining samples. In doing this, make sure you use our mathematical notation consistently.

```{list-table} Predictor $x$, actual label $y$, prediction $\hat{y}$ and error $e$ of our 5 individuals.
:header-rows: 1
:name: AgeVsSalary2

* - ID
  - $x$
  - $y$
  - $\hat{y}$
  - $e$
* - $1$
  - 37
  - 68,000
  - 37,000
  - 31,000
* - $2$
  - 18
  - 12,000
  - 18,000
  - -6,000
* - $3$
  - 66
  - 80,000
  - 66,000
  - 14,000
* - $4$
  - 25
  - 45,000
  - 25,000
  - 20,000
* - $5$
  - 26
  - 30,000  
  - 26,000
  - 4,000
```





### Quality metrics

Regression models can be represented using mathematical expressions that tell us how to calculate the predicted label from a predictor. For instance, the simple model $f(x) = 1,000 x$ predicts the salary of an individual as 1,000 times their age. {numref}`SalaryVsAge3models` shows in the attribute space a dataset consisting of the salary and age of a collection of individuals. Superimposed to the dataset are three curves that represent three candidate models that predict the salary of an individual given their age. Specifically, Model 1 represents a *linear* model such as $f(x) = 1,000 x$.



```{figure} images/salaryVage3sols_label.svg
---
name: SalaryVsAge3models
width: 70%
align: center
---
Toy dataset consisting of the salary and age of 200 individuals in the attribute space, together with three candidate models that predict the salary of individuals from their age.
```

```{admonition} Question for you
:class: question1

Given the dataset and candidate models shown in {numref}`SalaryVsAge3models`, which model would you say is the *best*, *Model 1*, *Model 2* or *Model 3*?

Submit your response here: <a href="https://forms.office.com/e/XagZJFmuLx" target="_blank">Your Response </a>

```

Did you identify the *best* model? Which one did you choose? It turns out that each of the models can potentially be the best model. The reason is that to talk about the best model, we need to agree on what we mean by *best* first. In other words, we need to agree on a notion of **model quality**. If we are looking for the simplest model, Model 1 would be the best, as it represents a very simple, linear relationship between salary and age. If we want our model to make predictions that reduce the prediction error overall, Model 3 would be the best. Finally, if we want our model not to make predictions that are always greater than the actual label, Model 2 would be the best. In summary, asking for the *best* model does not make sense until we decide what we mean by *best*. Or mathematically speaking, until we decide what our chosen **quality metric** is.

The **quadratic** or **squared error** $e^2$ is a common quantity used in regression to encapsulate the notion of **single prediction quality**. Given a sample $i$, the closer $e_i^2$ is to zero, the closer is the predicted label $\hat{y}_i$ to the actual label $y_i$. Using the squared error as our notion of single prediction quality, good models are those that lead to small squared errors across a collection of samples. What quantities can we define that give us an idea of how good a model is on a collection of samples, rather than just on one individual sample?

%In machine learning we do not have access to the entire population, but to a dataset consisting of $N$ samples extracted from the population. Using our dataset, we can define quantities that could give us an idea as to how good our model is.

One such quantity is the **sum of squared errors** (SSE) (also known as the residual sum of squares), which is defined as the sum of all the squared errors produced by our model on the dataset:

$$
SSE = e_1^2 + e_2^2+\dots+e_N^2
$$(eqSSE1)

or using the summation symbol $\Sigma$ (*sigma*)

$$
SSE &= \sum_1^N e_i^2 \\
&= \sum_1^N (y_i-\hat{y}_i)^2 \\
&= \sum_1^N (y_i-f(x_i))^2 \label{eq-sse2}
$$(eqSSE2)

%TEST- A link to an equation directive: {eq}`eqSSE1`


The SSE is a metric that can be used to quantify the overall quality of a model on a given dataset. The lower the SSE, the closer the model predictions are to the actual labels on the dataset. The performance of two models can then be compared by comparing their respective SSE values.

We can define a second, related quality metric that describes how good a model is at predicting a label on average. This quantity is known as the **mean squared error** (MSE) and can be obtained on a dataset by simply averaging the squared errors:

$$
MSE = \frac{1}{N}(e_1^2 + e_2^2+\dots+e_N^2) = \frac{1}{N}\sum_1^N e_i^2
$$(eqMSE1)

%TEST- A link to an equation directive: {eq}`eqMSE1`

As an example, the SSE of model $f(x) = 1,000 x$ on the dataset shown in {numref}`AgeVsSalary2` is:

$$
SSE = 31,000^2+(-6,000)^2+14,000^2+20,000^2+4,000^2 = 1,609,000,000
$$

and its MSE is

$$
MSE = \frac{1,609,000,000}{5} = 321,800,000
$$

Note that SSE and MSE seem to be very similar quantities. Specifically, MSE can be calculated as SSE divided by the number of samples $N$. The interpretation is however slightly different. We will come back to this idea in the next chapter. For now, let us just use both as two in principle equivalent quality metrics.


```{admonition} Question for you
:class: question1

Given a dataset, is it possible to find a model such that $\hat{y}_i = y_i$ for every
sample $i$ in the dataset, i.e. a model whose error is exactly zero (SSE=0 and MSE = 0)?

Submit your response here: <a href="https://forms.office.com/e/vemZER0DWJ" target = "_blank">Your Response</a>

```

A model such that SSE=0 on a dataset can be visualised in the attribute space as a curve that goes through every single sample. Therefore, the question as to whether there exists such a model for any dataset can be rephrased as, can we draw a curve that goes through every single sample in the dataset? At first the answer seems to be yes - after all, we can draw as wiggly a curve as we want to so that it goes through each one of the samples. There is, however, one restriction. Models produce one prediction per predictor and therefore, visually they are not allowed to go through two samples that have the same predictor. Thus, if our dataset has two samples that have the same predictor and different labels, no model will be able to predict both labels and therefore the error will never be zero. Think of the problem of predicting the salary of a Parisian. If our dataset of Parisians has two individuals of the same age but different salaries, then no matter how hard we try, our model will predict one and only one salary and therefore will produce the wrong prediction for at least one of these two individuals. In summary, it is never guaranteed that if we are given a dataset we will be able to find a zero error model.




% In other words, it will not always predict the exact salary of a Parisian. Why not? The reason is simple: in our target population, Paris, we will surely be able to find two individuals of the same age with different salaries. No matter how hard we try, our model will predict one and only one salary and therefore will produce the wrong prediction for at least one of these two hypothetical individuals. Remember to embrace the error, nothing is perfect.



### Regression as an optimisation problem (Take 1)

You might be wondering whether we have forgotten to remove the text *(Take 1)* from the heading. We have not. In this section we present our first formulation of regression problems. In the next chapter, we will refine this formulation. To present the refined version, we need to consolidate some basic understanding first.

In a regression problem, we have three main components:
- A dataset, $\{(x_i,y_i): 1\leq i \leq N \}$.
- A collection of candidate models, $f$.
- A quality metric.

Our *Take 1* definition of regression is as follows. We define regression as the process of identifying the best model from a set of candidate models, where the best model is the one that exhibits the highest quality *on the available dataset*. If we use the SSE as our quality metric, the best model is the one that has the lowest SSE value on the dataset. A mathematician would write:

$$
f_{best} = \underset{f}{\operatorname{argmin}} \sum_1^N (y_i-f(x_i))^2
$$(eqfbest_1)

which might look scary but simply reads *the best model, $f_{best}$, among all the candidate models, $f$, is the one that has the lowest (argmin) SSE on our dataset, where the SSE is calculated as $\sum_1^N (y_i-f(x_i))^2$*. In machine learning, we say that we are **training** a model or **fitting** a model to a dataset when we use a dataset to identify the best model among a family of candidate models. Accordingly, we call the dataset that we are fitting the model to the **training dataset**.

This process, where we aim at identifying the model that produces the lowest error, is what we call in mathematics an **optimisation** problem. Incidentally, using the SSE as our quality metric we have just formulated the classical **least squares** problem that Legendre and others, including Karl F. Gauss, came up with more than two centuries ago. Note that the best model according to the SSE metric is the same as the best model accrding to the MSE metric, as we have defined the latter as the former divided by $N$.

% This process is what we call in mathematics an **optimisation** problem, where we aim at identifying the object (in our case, the model) that produces the lowest cost (in our case, the lowest SSE).

```{admonition} Question for you
:class: question1

Consider the following three models:

$f_1(x) = 1,000x$

$f_2(x) = 999x$

$f_3(x) = 1,000  + 1,000x$

Using the SSE as your quality metric and the dataset in {numref}`AgeVsSalary`, identify the best model among the three candidate models $f_1$, $f_2$ and $f_3$.

Submit your response here: <a href="https://forms.office.com/e/etKdZmRC37" target = "_blank">Your Response</a>

```

The idea of regression looks very simple: we have a collection of models, each with an associated quality obtained using a dataset. Our task is to identify the one with the highest quality. If we have a few candidate models this is easy. We use the training dataset to compute their quality (for instance, SSE), rank them according to this quality and choose the one at the top. This is what you must have done to solve the previous question. However, what if we have an infinite number of candidate models? We cannot possibly compute each individual quality! Note that this is not an extreme situation. On the contrary, it is the most common case. Think about models $f_1(x) = 1,000x$ and $f_2(x) = 999x$. They look almost the same, yet using the coefficient $1,000$ or $999$ makes them different. We have in fact an infinite choice of values for this coefficient and therefore, we could consider an infinite number of candidate models. Optimisation theory will provide us with useful approaches to operate in such scenarios.


(Reg3)=
## Basic regression models

To build a machine learning solution for a given regression problem, we need to identify a family of candidate models. In this section we introduce the **linear** and **polynomial** families of regression models. We will distinguish between **simple regression**, in which there is is only one predictor, and **multiple regression**, which considers two or more predictors. One example of a simple regression problem is that of predicting the salary of an individual knowing their age. Predicting a salary knowing the age and the height of an individual is one example of multiple regression. At the end of this section we present the **least squares** solution, which can be used to identify exactly the best model within a family of linear or polynomial models.

### Simple linear regression

The family of **linear models** for **simple regression problems** prescribe a linear relationship between the predictor and the label:

$$
f(x) = w_o + w_1 x
$$(eqSLMs1)

Simple linear models have two **parameters**, namely the intercept ($w_0$) and the gradient or slope ($w_1$). Changing the value of either parameter leads to different models. Hence, if we use the family of linear models to build our solution, finding the best model is equivalent to identifying the values for the intercept and the gradient that yield the highest quality. This is why we sometimes refer to model training as **parameter tuning**, since training involves changing or tuning the values of the parameters of the model. Note that we have an infinite number of choices for both parameters, i.e. any real number between minus infinity and infinity. Hence, the number of models belonging to this family is infinite too.


{numref}`SalaryVsAgeLinear` shows the result of fitting a linear model to our salary vs age toy dataset, using the SSE as our quality metric. The solution is the linear model that produces the lowest SSE on our dataset. In other words, it is the *least squares linear solution*. Note that we have not yet explained how to find this model, e.g. how we have fitted the model to the dataset. For now, you can assume that you have an optimisation genie that does this for you.

```{figure} images/salaryvsageSolMSE.svg
---
name: SalaryVsAgeLinear
---
Linear solution for the salary vs age toy dataset, using the SSE on the training dataset as our quality metric.
```

### Simple polynomial regression

Visually, you might have concluded that the linear solution that we have obtained does not capture well the relationship between salary and age. It is indeed the best linear model, but it does not seem to be good enough. Unfortunatelly changing the intercept and gradient of our linear models, we will not be able to produce a curve that represents adequately the relationship that we want to discover. We need a family of models that is less rigid than the linear family and allows us to produce more complex curves.

One such family is the family of **polynomial models**. In polynomial regression, we use models that follow the mathematical expression:

$$
f(x) = w_0 + w_1 x + w_2 x^2+ \dots + w_D x^D
$$(eqSPMs1)

where $D$ is known as the degree of the polynomial. Linear models are of course a subfamily of the polynomial family, where $D = 1$. Depending on our chosen value for $D$, we can define different families. When $D=2$, we have the quadratic family:

$$
f(x) = w_0 + w_1 x + w_2 x^2
$$(eqSPMs_D2)

when $D=3$, the cubic family:

$$
f(x) = w_0 + w_1 x + w_2 x^2 + w_3 x^3
$$(eqSPMs_D3)

and so on. {numref}`SalaryVsAgeQuadratic`, {numref}`SalaryVsAgeCubic` and {numref}`SalaryVsAge5` show the quadratic, cubic and degree 5 least squares solutions. As you can see, increasing the degree of the polynomial $D$ gives us more flexibility to produce models that are fitted better to the dataset.

```{figure} images/salaryvsageSolsMSEQuadratic.svg
---
name: SalaryVsAgeQuadratic
---
Quadratic solution for the salary vs age toy dataset, using the SSE on the training dataset as our quality metric.
```

```{figure} images/salaryvsageSolsMSECubic.svg
---
name: SalaryVsAgeCubic
---
Cubic solution for the salary vs age toy dataset, using the SSE on the training dataset as our quality metric.
```

```{figure} images/salaryvsageSolsMSE5.svg
---
name: SalaryVsAge5
---
Solution for the salary vs age toy dataset for $D=5$, using the SSE on the training dataset as our quality metric.
```

Once again, we have not discussed yet how to obtain these solutions. At this stage, what is important is to understand how to express polynomial models mathematically and reflect on the solutions that they can produce once fitted to a training dataset.


### Multiple linear regression

So far we have considered **simple regression** problems, which are problems where there is only one predictor. **Multiple regression** involves two or more predictors. For instance, in the multiple regression problem where we want to predict the salary of in individual from their age and height, age and height are the two predictors. A toy dataset that we could use to build machine learning solutions for this multiple regression problem is shown in the attribute space in {numref}`SalaryVsAgeVsHeight`.

```{figure} images/SalaryVsAgeVsHeight.svg
---
name: SalaryVsAgeVsHeight
---
Toy dataset consisting of the salary, age and height of 200 individuals in the attribute space.
```
A linear model for this multiple regression problem could be expressed mathematically as follows:

$$
SALARY = w_0 + w_{a} \times AGE + w_{h} \times HEIGHT
$$(eqSalary)

where the coefficients $w_0$, $w_{a}$ and $w_{h}$ are the model's parameters. If we fit this linear model to the dataset in {numref}`SalaryVsAgeVsHeight`, using again our optimisation genie, we will obtain as our solution the plane shown in {numref}`SalaryVsAgeVsHeightSurface`. This plane dictates how to predict the salary of an individual, based on their age and height.

```{figure} images/SalaryVsAgeVsHeightSurface.svg
---
name: SalaryVsAgeVsHeightSurface
---
The plane surface represents the linear solution for the toy salary vs age and height dataset, using the SSE on the training dataset as our quality metric.
```

Linear models in multiple regression, such as {eq}`eqSalary`, are defined as the *sum of a constant (the intercept) plus each predictor multiplied by a coefficient*. The constant and coefficients are precisely the parameters of the linear model that we need to tune. To formulate multiple regression mathematically and obtain general solutions, we need to develop a few last pieces of mathematical notation, starting with a symbol for

- **Number of predictors**: $K$.

As an example, if we want to predict the salary of an individual from their age and height, $K=2$. A small toy dataset for this problem is shown in {numref}`AgeHeightVsSalary`. As you can see, each individual in the dataset is described by three attributes, two of which are used as the predictors (age and height) and the third one as the label (salary).

```{list-table} A toy dataset registering the age and salary of a small group of individuals
:header-rows: 1
:name: AgeHeightVsSalary

* - ID
  - Age
  - Height [cm]
  - Salary
* - $S_1$
  - 18
  - 175
  - 12,000
* - $S_2$
  - 37
  - 180
  - 68,000
* - $S_3$
  - 66
  - 158
  - 80,000
* - $S_4$
  - 25
  - 168
  - 45,000
* - $S_5$
  - 26
  - 190
  - 30,000  
```

We can extend our mathematical notation to identify each of the predictors of each sample. We will denote the $k$-th predictor of sample $i$ by $x_{i,k}$. Accordingly, $x_{i,1}$ is the first predictor of sample $i$, $x_{i,2}$ is the second predictor and so on. For instance, if age is our first predictor and height our second predictor, using the dataset shown in {numref}`AgeHeightVsSalary` we would write $x_{1,1}=18$, $x_{1,2}=175$, $x_{2,1}=37$ and so on.

The next piece of notation allows us to pack all the predictors together in a column vector which we represent using **bold font**:

$$
\boldsymbol{x}_i= \begin{bmatrix}
1\\
x_{i,1}\\
x_{i,2}\\
\vdots \\
x_{i,K}
\end{bmatrix}
$$(eqxi1)

As you can see, vector $\boldsymbol{x}_i$ contains all the predictors of sample $i$. There is an additional entry, the number 1, whose role will be clear very soon.

%We can use $\boldsymbol{x}_i$ to refer to all the predictors of sample $i$, without needing to specify each one of them %individually. In addition to being a compact way to do it, we can use the same symbol $\boldsymbol{x}_i$ in any multiple %regression problem, without needing to specify how many or which predictors we have.

At this stage, we can abstract away what each predictor means and how many predictors there are by simply using the symbol $\boldsymbol{x}_i$ to denote all the predictors of sample $i$. Using this notation, we can write

$$
\hat{y}_i = f(\boldsymbol{x}_i)
$$(eqyi1)

which should be read as *model $f$ takes as an input all the predictors of sample $i$, which are packed in $\boldsymbol{x}_i$, and produces the prediction $\hat{y}_i$*. Note that {eq}`eqyi1` can be used to respresent any multiple regression problem, irrespective of the number of predictors that it defines. In addition, {eq}`eqyi1` looks almost identical to {eq}`eqModelNotation`. The only difference is that the input to the model is now a set of predictors instead of a single one, which for convenience we highlight by using bold font instead of normal font.

We will also pack the parameters of a multiple linear regression model in a vector. These parameters are a constant ($w_0$) and the coefficients that multiply each predictor. We denote this vector by $\boldsymbol{w}$ and define it as:

$$
\boldsymbol{w}= \begin{bmatrix}
w_0\\
w_{1}\\
w_{2}\\
\vdots \\
w_{K}
\end{bmatrix}
$$(eqw1)

Note that there are $K+1$ parameters in a linear model for a multiple regression problem with $K$ predictors. Using this notation and a bit of vector algebra, we can express *any* multiple linear regression model as  

$$
f(\boldsymbol{x}_i) &= w_0 + w_1 x_{i,1} + w_2 x_{i,2} + \dots + w_K x_{i,K}\\
&= \boldsymbol{x}_i^T\boldsymbol{w}
$$(eqfx1)

where $T$ denotes vector transposition and $\boldsymbol{x}_i^T\boldsymbol{w}$ is the vector multiplication of vector $\boldsymbol{x}_i^T$ and vector $\boldsymbol{w}$. Pretty neat, isn't it? The role of the entry of value 1 in the extended vector $\boldsymbol{x}_i$ should be now clearer: it multiplies the coefficient $w_0$ and allows us to build the compact expression $\boldsymbol{x}_i^T\boldsymbol{w}$.

Let us take our vector notation one step further and define the **design matrix** $\boldsymbol{X}$ and the **label vector** $\boldsymbol{y}$. Given a dataset consisting of a collection of $N$ samples described by $K$ predictors and one label, **the design matrix encapsulates all the predictor values** and is defined as

$$
\boldsymbol{X}= \begin{bmatrix}
1 & x_{1,1}& x_{1,2}& \dots & x_{1,K}  \\
1 & x_{2,1}& x_{2,2}& \dots & x_{2,K}  \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{N,1}& x_{N,2}& \dots & x_{N,K}  \\
\end{bmatrix}\nonumber
$$(eqX1)

The all-ones first column in {eq}`eqX1` play the same role as the 1 entry in {eq}`eqxi1`. The **label vector $\boldsymbol{y}$ contains all the labels** of the $N$ samples in our dataset and is defined as

$$
\boldsymbol{y}= \begin{bmatrix}
y_{1}\\
y_{2}\\
\vdots \\
y_{N}
\end{bmatrix}
$$ (eqy1)

<!-- TEST - equation link {eq}`eqy1` -->

The design matrix design matrix $\boldsymbol{X}$ and the label vector $\boldsymbol{y}$ pack all the values in our dataset. For instance, if we are considering the problem of predicting the salary of an individual from their age and height, and assume that age is our first predictor and height our second predictor, {numref}`AgeHeightVsSalary` would lead to the following design matrix and label vector:

$$
\boldsymbol{X}= \begin{bmatrix}
1 & 18 & 175  \\
1 & 37 & 180 \\
1 & 66 & 158 \\
1 & 25 & 168 \\
1 & 26 & 190
\end{bmatrix}\nonumber
\quad  \quad
\boldsymbol{y}= \begin{bmatrix}
12,000\\
68,000\\
80,000 \\
45,000\\
30,000
\end{bmatrix}
$$(eqXy1)



We also need to define a new vector $\hat{\boldsymbol{y}}$ that, given a model, contains **all the predicted labels**:

$$
\hat{\boldsymbol{y}} = \begin{bmatrix}
\hat{y}_{1}\\
\hat{y}_{2}\\
\vdots \\
\hat{y}_{N}
\end{bmatrix}
$$

Applying basic matrix algebra, given a linear model defined by a coefficients vector $\boldsymbol{w}$, we can express $\hat{\boldsymbol{y}}$ as the product of the design matrix $\boldsymbol{X}$ and the coefficients vector $\boldsymbol{w}$:

$$
\hat{\boldsymbol{y}}&=\boldsymbol{X}\boldsymbol{w}\\
&= \begin{bmatrix}
1 & x_{1,1}& x_{1,2}& \dots & x_{1,K}  \\
1 & x_{2,1}& x_{2,2}& \dots & x_{2,K}  \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{N,1}& x_{N,2}& \dots & x_{N,K}  \\
\end{bmatrix}\nonumber
\begin{bmatrix}
w_{0}\\
w_{1}\\
w_{2}\\
\vdots \\
w_{K}
\end{bmatrix}
$$(eqY_hat1)

Finally, we can define vector $\boldsymbol{e}$, which contains **all the prediction errors**:

$$
\boldsymbol{e} = \begin{bmatrix}
{e}_{1}\\
{e}_{2}\\
\vdots \\
{e}_{N}
\end{bmatrix} =
\begin{bmatrix}
{y}_{1}-\hat{y}_{1}\\
{y}_{2}-\hat{y}_{2}\\
\vdots \\
{y}_{N}-\hat{y}_{N}
\end{bmatrix} =
\boldsymbol{y}-\hat{\boldsymbol{y}}
$$(eqErr1)

Even though it might look complicated at first, vector notation makes it easier for us to describe our problems and build solutions. Essentially, when discussing multiple regression problems we will use the mathematical symbols $\boldsymbol{x}_i$, $\boldsymbol{X}$, $\boldsymbol{y}$, $\boldsymbol{w}$, $\hat{\boldsymbol{y}}$ and $\boldsymbol{e}$, irrespective of whether we have 2 predictors or 1,000 predictors, 10 samples or 100,000 samples. This is the power of our mathematical notation: we can abstract details away and focus instead on the essence of the problem.


### The least squares solution

Now that we have developed our mathematical notation, we are ready to show you how to fit a multiple linear model to a given dataset using the SSE, or equivalently the MSE, as our quality metric. We will have to wait until the next chapter to understand fully where this solution comes from. For now, you will just need to trust us.

The coefficients $\boldsymbol{w}$ of the multiple linear model with the lowest SSE on a training dataset characterised by a design matrix $\boldsymbol{X}$ and a label vector $\boldsymbol{y}$, can be calculated as

$$
\boldsymbol{w}_{best} = \left(\boldsymbol{X}^T \boldsymbol{X}\right)^{-1} \boldsymbol{X}^T \boldsymbol{y}
$$(eqW_best1)

This is the **least squares** solution. The calculation defined by {eq}`eqW_best1` consists of simple matrix operations (multiplication, inversion and transposition) that can be easily implemented in computing engines equipped with linear algebra capabilities. In addition to having the lowest SSE *on the training dataset*, the model with coefficients $\boldsymbol{w}_{best}$ also has the lowest MSE *on the training dataset*.

As you would expect, this solution can also be used for simple linear regression as simple linear regression problems can be formulated as a multiple linear regression problem where $K=1$. What might be susrpising at first is to know that we can also use the least squares solution for multiple linear regression to solve polynomial regression problems. Let us see how to solve simple polynomial regression.

In simple polynomial regression, the predicted label $\hat{y}_i$ is calculated as

$$
\hat{y}_i = w_0 + w_1 x_i + w_2 x_i^2 + \dots + w_D x_i^D
$$(eqY_hat2)

This expression is similar to the multiple linear regression expression {eq}`eqfx1`, where instead of a linear combination of predictors, we have a **linear combination of the powers of one predictor**. The trickt consists of treating the powers of the predictor as predictors themselves, i.e. $x_i$ is the first predictor, $x_i^2$ the second predictor and so on. Accordingly, we can create a design matrix $\boldsymbol{X}_P$ where each column corresponds to a power of the only predictor

$$
\boldsymbol{X}_P=
\begin{bmatrix}
1 & x_{1,1}& x_{1}^2& \dots & x_{1}^D  \\
1 & x_{2,1}& x_{2}^2& \dots & x_{2}^D  \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{N,1}& x_{N}^2& \dots & x_{N}^D  \\
\end{bmatrix}\nonumber
$$(eqXp1)

Using this design matrix in the least squares solution {eq}`eqW_best1`, we can obtain the coefficients of the best polynomial model of degree $D$, assuming the SSE on the training dataset is our quality metric.



(Reg4)=
## Flexibility, interpretability and generalisation

Linear and polynomial models have a set of parameters that we can tune. Changing the values of these parameters allows us to generate different shapes, for instance setting the gradient of a linear model to 0 we obtain a horizontal straight line, while setting it to 1 we obtain a straight line that forms a 45 degree angle with the horizontal axis.

The range of shapes that a model can produce is in general related to the number of parameters that it has. We sometimes refer to the number of parameters of a model as its degrees of freedom. A quadratic model, for instance, has three parameters (3 degrees of freedom) and can produce a wider range of shapes than a linear model, which has two parameters (2 degrees of freedom); a cubic model has four parameters (4 degrees of freedom) and produces more shapes than a quadratic model, and so on. The ability of a model to generate different shapes by changing the value of its parameters is known as the model's **flexibility**. Accordingly, a linear model is very rigid, as it can only generate straight lines. In comparison, a cubic model is flexible, as it can generate many more curves, including straight lines. Flexible models are also said to be more **complex** in that they can generate curves of greater complexity.

The expected quality of a model on the training dataset is related to its flexibility. Since flexible models can generate a wider range of shapes than rigid ones, we should expect flexible models to be able to produce solutions that are better fitted to our training dataset, compared to rigid ones. For instance, {numref}`SalaryVsAgeLinear`, {numref}`SalaryVsAgeQuadratic`, {numref}`SalaryVsAgeCubic` and {numref}`SalaryVsAge5` show that as we increase the degree of the polynomial, our solution follows better the observed pattern. Indeed, the SSE and MSE values of the best linear model on this toy dataset is the highest, whereas the SSE and MSE values of the best polynomial of order 5 is the lowest. This implies, following our *take 1* definition, that the quality of a polynomial of order 5 is higher than the quality of a linear model. On the flip side, flexible models are harder to interpret than rigid ones. Model **interpretability** is crucial for us, as humans, to make sense in a qualitative manner how a predictor is mapped to a label. Linear models, which are very rigid, are easier to interpret. For instance, we could describe the best linear model shown in {numref}`SalaryVsAgeLinear` by simply saying, *the older you are, the higher your salary*. In contrast, describing {numref}`SalaryVsAge5` would be much more dificcult.

If flexible and complex models are expected to have a higher quality on the training dataset than rigid and simple models, should we always be using complex models? The answer is no. To understand why, we need to remind ourselves that our ultimate goal is to *deploy* the model that we have built. In other words, we ultimately want to put our model to work. Accordingly, our goal should not be to produce models that have a high quality on the training dataset, but models that have a **high quality when deployed**. It turns out that training and deployment qualities behave quite differently.

{numref}`TrainDeployment` illustrates what happens to the training and deployment quality as we increase the flexibility of our models. Initially, as we increase the flexibility, the quality improves both during training and deployment, as the MSE is decreasing. This means that by initially increasing the flexibility, our models make better predictions. However, beyond a certaing degree of flexibility, the training quality keeps improving whereas the deployment quality deteriorates, as the deployment MSE starts to increase. Therefore, a model that appears to be working very well on the training dataset, can in reality perform very poorly during deployment. The question is, why are training and deployment qualities behaving so differently?


```{figure} images/train_test_MSE2.jpg
---
name: TrainDeployment
---
Training and deployment MSE of models of increasing flexibility. The flexibility of a model can be defined as its degrees of freedom.
```

{numref}`under_over_dataset` shows a simple dataset consisting of 8 samples randomly extracted from the dataset shown in {numref}`SalaryVsAge3models`. Note that the pattern that we could visually identify in {numref}`SalaryVsAge3models` cannot be discerned anymore, due to having only a few samples.

% Visually, these samples seem to be following a pattern with some small deviations. These deviations are due to chance alone and are hence, irrelevant.


```{figure} images/under_over_dataset.svg
---
name: under_over_dataset
---
Small dataset consisting og 8 samples.
```

If we fit a rigid linear model to this small dataset, we obtain the straight line shown in {numref}`under_over_linearsol`. Overall, this straight line follows the general pattern, however we could wonder if we could reduce the prediction error on the training dataset further. To do so, we need to use models of higher complexity.

```{figure} images/under_over_linearsol.svg
---
name: under_over_linearsol
---
Linear model fitted to the small dataset.
```

{numref}`under_over_6ordercsol` shows the result of fitting a polynomial model of order 6 to the training dataset. This polynomial predicts almost without errors the label of every single sample in the training dataset. Accordingly, the MSE on the training dataset is close to zero and we could be tempted to conclude that this model is close to perfect.


```{figure} images/under_over_6ordersol.svg
---
name: under_over_6ordercsol
---
Polynomial model of order 6 fitted to the small dataset.
```

Finally, {numref}`under_over_cubicsol` shows a cubic fit. This model is more complex than the basic linear one, but not as complex as the polynomial model of order 6. Its MSE on the training dataset is lower than the linear model, but higher than the polynomial of order 6.

```{figure} images/under_over_cubicsol.svg
---
name: under_over_cubicsol
---
Cubic model fitted to the small dataset.
```

Linear, cubic, order 6, which one is the right model? Frustraitingly the answer is, we cannot tell just by looking at their performance *on the training dataset*. The only way for us to decide which one is the best is by **assessing the quality of our models during deployment**. One way to assess the quality of our models during deployment, without actually deploying them, is to use a separate dataset. For instance, if we superimpose the linear, cubic and order 6 solutions to the dataset shown in {numref}`SalaryVsAge3models`, we would conclude that the order 6 solution is actually performing very poorly, in spite of performing so well on the training dataset. A model that captures the right pattern will be able to perform well when presented with new samples that it has not been exposed to before. We would say that this model is **generalising** well.

For the sake of the argument, assume that the real pattern underlying the training dataset shown in {numref}`under_over_dataset` is a cubic one. Then, the linear model would be too rigid, the polynomial of order 6 would be too complex and the cubic model would be the right one. These three behaviours can be identified in {numref}`TrainDeployment` and we have three terms to refer to them:
- **Underfitting** models produce large errors during training and deployment. The flexibility of these models is too low and are unable to reproduce the underlying pattern. They occupy the left-hand side of {numref}`TrainDeployment`.
- **Overfitting** models are too flexible and perform extremely well on the training dataset at the expense of their generalisation ability. Consequently, they produce very small errors during training and large errors during deployment. They occupy the right-hand side of {numref}`TrainDeployment`.
- **Just right** models produce small errors during training and deploymnet. They have the right complexity and are capable of reproducing the underlying pattern. They are situated between the underfitting and overfitting regions in {numref}`TrainDeployment`.

According to these terms, if we assume that the underlying pattern in {numref}`under_over_dataset` has a cubic complexity, the linear model is underfitting and it produces large errors during training and deployment. The polynomial model of order 6 is overfitting, as its error is practically zero during training, but would be very high during deployment.


(Reg5)=
## Summary and discussion

In regression we set out to build a model that predicts the value of a continuous label using a set of predictors. There are many real-world problems that can be formulated as regression problems. One example of a problem that could be tackled using regression approaches would be that of predicting the energy consumption of a household, given the location of the house, the household size and the income. In this case the energy consumption is the continuous label and location, size and income the predictors.


The basic ingredients of any regression problem are a **training dataset**, a family of **candidate models**, a **quality metric** and an **optimisation method**. The training dataset is a collection of samples extracted from the population against which we will deploy our model. We use a training dataset because we do not know the true relationship between the label and the predictors. Our hope is to discover such relationship in the training dataset. An example of a real-world problem that we could formulate as a regression problem but we would not, is that of predicting the distance driven by a vehicle, using speed and journey duration as predictors. In this case we know very well the relationship between distance, speed and duration, hence there is no need to extract this relationship from a dataset.

Building a model involves selecting the best one among a family of candidate models. In this chapter we have covered linear and polynomial models, but there are many others, including exponential models, sinusoidal models, radial basis functions, spline, the logistic model and many more. These models have a set of parameters that can be tuned. Hence, finding the best model can be seen as finding the best values for the model's parameters.

It is important to highlight that to talk about a best model, we need to first agree on a notion of quality. In this chapter we have used two related quality metrics, namely the SSE and the MSE. Other quality metrics that you might come across include the root mean squared error

$$
RMSE = \sqrt{\frac{1}{N}\sum{e_i^2}}
$$(eqRMSE1)

the mean absolute error

$$
MAE = \frac{1}{N}\sum{|e_i|}
$$(eqMAE1)

or the R-squared metric

$$
R^2 = 1 -\frac{\sum{e_i^2}}{\sum{(y_i-\bar{y})^2}}, \quad \text{where} \quad \bar{y}=\frac{1}{N}\sum{y_i}
$$(eqRSq1)

Note that in general, the best model according to one metric will be different from the best model according to another metric. So which one is the right quality metric? Machine learning will not automatically answer this question. It is the job of the machine learning expert who formulates the problem to decide which metric is the most appropriate. Remember our first top tip: **Know Thy Domain!** Only by knowing our domain we will be able to define a suitable quality metric.

% to choose the quality metric that is more closely aligned with our real-world goals. Once again, it is the machine learning expert who has

You might still be asking yourself, why can I not expect my models to always produce perfect predictions? Why do models make errors? We can identify several factors that contribute to this. First, our family of candidate models might not be flexible enough to reproduce the correct pattern. We sometimes call this model bias. Second, there might be factors that determine the value of the label, but are not included among the chosen predictors. For instance, no matter how hard we try, we will never be able to predict the salary of an individual using their age as the only predictor. Other factors contribute to someone' salary, for instance education or family background. Third, we might not have enough samples to reveal a very complex underlying pattern. Finally, unrelated random factors might be contributing to the final value of a label. By their nature, these random factors cannot be predictied deterministically.

The final basic element in any regression problem is an optimisation method, which allows us to identify the best model among a family of candidate models. In this chapter we have presented the least squares solution for linear and polynomial models. Least squares is an exact solution obtained using basic optimisation theory. In general, exacts solutions will not be available and we will need to implement other optimisation approaches to identify the best model.

At the end of this chapter, we discovered a disturbing fact about our approach: the best model for our training dataset, migth not be the best model during deployment. We hoped that finding the best model on our training dataset would be sufficient, but then we observed that in fact, a perfect model on our training dataset might be a terrible model when deployed. This does not mean that we need to discard everything we have learnt so far. Instead, we need to reformulate our regression problem slightly. We have defined regression as the process of identifying the best model from a set of candidate models, where the best model is the one with highest quality *on the available dataset*. This is our *Take 1* definition. From now on, we will define the best model as the one with the highest quality when deployed, in other words, *on the target population*. This is our *Take 2* and final definition. The question is, how do we identify this model, if all we have about our population is a dataset? This is precisely one of the main questions that we will be addressing in the next chapter.


%Compare SSE and MSE when we use different datasets of different sizes. It is harder to use the SSE to compare models built using datasets, MSE is easier though.

%The take 2 version of regression includes training and test. In this chapter, we only discussed train

%a *population of individuals* described by two *attributes*, namely their age and salary. With this population in mind, we could formulate two different regression problems. The first problem would be predicting the salary of one individual based on their age. The second, predicting the age based on their salary. Both of them are valid regression problems. Let us choose the problem of predicting the salary based on the age. By making this choice, *we have decided* that the salary attribute is our label, and the age attribute our predictor.
