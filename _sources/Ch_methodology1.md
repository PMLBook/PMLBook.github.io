(Meth1)=
# Methodology I: Three basic tasks

In the {ref}`Reg` chapter we explored our first family of machine learning problems. We defined regression as a problem where we seek to predict the value of a continuous label based on a set of predictors. Crucially, we highlighted that what makes regression a machine learning problem is the use of datasets to build our solutions. However, while analysing several cases we discovered a disturbing reality: solution models that appear to work well on a training dataset, might actually perform poorly when deployed. The main take-home message was that we cannot tell how well a model will work when deployed by simply looking at its performance on the training dataset. This is very concerning, as we would like to deploy only those models that we know will work well.

Do these findings reveal an intrinsic limitation of machine learning? The answer is no. What they indicate is that the machine learning methodology that we have used so far is limited. In this chapter, we will focus on developing a rigorous machine learning methodology. The principles that we will present in this chapter are general and applicable to any machine learning problem, be it a regression, a classification or an unsupervised learning one. Our starting point will be a discussion around the fundamental concepts of population and datasets. This discussion will be guided by our deployment-first view of machine learning. Then, we will be covering three main machine learning tasks, namely the test task, the training task and the validation task. As you should expect, in all three tasks we will be using datasets.

Before we immerse ourselves into developing our machine learning methodology, let us travel back in time and revisit one of the greatest scientific achievements of the 20th century: the decipherment of the ancient Linear B script. Out of this story, seemingly unrelated to machine learning, we will extract our third top tip.






## Ventris' decisive check

Crete is a mountainous island in the estearn Mediterranean, with the Aegean Sea to the North and the Lybian Sea to the South. There are traces of human settlements in Crete dating back to the Paleolithic and ever since, a multitude of cultures and civilisations have inhabited uninterruptedly its shores, mountains and valleys. In classical times, Crete was renowned for having been ruled by King Minos, who every nine years would send seven girls and seven boys to be devoured by the fearsome Minotaur, a part-man, part-bull creature trapped inside an elaborate stone maze, the Labyrinth. To say that Crete is one of the epicentres of the classical era is no overstatement.

The historical prominence of Crete attracted by the end of the 19th century robbers, anticuarians, archaelogists and scholars - if they could be distinguished. Among those individuals was the British archaelogist Arthur Evans who, in 1900, started excavating the ruins of the ancient city of Knossos, in central Crete. In Knossos, Evans believed to have discovered the Minotaur Labyrinth and the palace of King Minos, and during his excavations he dug up many fine artifacts, including a bull-shaped drinking cup and two snake goddess statuettes. Perhaps most intriguing of all, Evans found thousands of small, palm-sized clay tablets written in an unknown script (see {numref}`LinearBTablet`). Evans called this script 'Linear Script of Class B', to distinguish it from a different script also found in Crete, the 'Linear Script of Class A' and went on to spend his life trying to decipher it. Evans died in 1941 having failed to decipher the Linear B script. That honour would fall to another Briton, Michael Ventris, an architect who announced his solution to this enigma in 1952.

```{figure} images/LinearBTablet.jpg
---
name: LinearBTablet
scale: 90%
---
Clay tablet written in Linear B. Photo taken in The British Museum, London.
```

You might be asking yourself what took the world so long to decipher Linear B, but the right question should be instead, how did we manage to decipher Linear B at all? Those who were trying to decipher Linear B were looking at thousands of fragments of text of unknown contents, written in an unknown script, encoding an unknown language. Think of it for a moment: Is this not an impossible proposition? Despite this, Ventris got to crack Linear B. Ventris' reflections on the methods that he used to decipher Linear B contain authentic scientific jewels that can be easily overlooked. Let us bring back to life one of these gems, **Ventris' decisive check**.

According to Ventris, any decipherment process consists of three separate stages, the third of which is

*a decisive check, preferably with the aid of virgin material, to ensure that the apparent results are not due to fantasy, coincidence or circular reasoning*.

Ventris is telling us that no matter how promising our solution looks, our work should not end until we have *checked* it. Crucially, this final check should be done using material that we have not seen while we were building our solution. The history of decipherment abounds with examples of decipherers announcing mutually incompatible solutions to the same riddle, following what they usually describe as eureka moments. Most of these decipherers are in reality fooling themselves as they assume that for a solution to be valid, it has to work on the material that they have used to build it. Good decipherers are not only capable of providing promising solutions: they also know how to rigorously check them.



```{admonition} Our third top tip follows on from Ventris' advice:
:class: tip
<h3 style="text-align: center;"><b>Don't fool thyself!</b></h3>
```

Ventris' reflections should by now sound familiar to you. In machine learning, as in decipherment, we solve problems using some recorded material, in our case, datasets. Machine learning also shares with decipherment the risk of building solutions that work well on the recorded material, but are actually wrong. In machine learning lingo, decipherers that fool themselves build solutions that *overfit* to their material. What is worse: they are unaware of it. In machine learning, as in decipherment, being able to check rigorously our solutions is as essential as being able to build them.

In this chapter we will develop a rigurous machine learning methodology that will help us reduce the risk of fooling ourselves. We will call our final, decisive check, the **test task**. On a final note, Ventris announced his solution in July 1952, but its confirmation had to wait until May 1953, when an american archaeologist, Carl Blegen, who was excavating a site in Pylos, Greece, found a new clay tablet written in Linear B and using Ventris' solution was able to read it. This was Ventris' decisive check to the letter.









## Populations and datasets

We briefly introduced the concept of population in the {ref}`Intro` chapter, where we defined it as an entity from which samples are extracted. What did we mean by this? And how are datasets, which are collections of samples, related to populations? In this section we will explore the relationship between populations and datasets in detail.

### The notion of population

In its original meaning, a population is a collection of individuals who inhabit a particular place, such as a town, a region or a country. Statistics as a discipline originated out of the need to understand such -human- populations, but as time went by the field of statistics grew to encompass a wider range of problems. This resulted in the term *population* abandoning its first, concrete sense and being used to refer to any entity that can be studied using the same methods developed to study human populations.

% extract samples following some mechanism or law


%What is then a population in machine learning?

Given a machine learning problem, a population is the abstract entity that consists of all the possible samples that the problem can refer to. A population in machine learning can produce samples, which is the reason why we sometimes use the term *data source* instead. Machine learning problems always have a target population and any machine learning solution is meant to work on samples extracted from that same target population. Let us reflect on the notion of population using two machine learning examples. First, consider the regression problem of guessing the salary of an individual of whom we know their age. When we formulate this problem, we need to specify which group of individuals we are targetting, as it would be reasonable to expect the relationship between salary and age to be dependent on the time when and place where the individuals live. For instance, we would expect the relationship between age and salary in today's city of Heraklion, in the island of Crete, to be different from that in 19th century Paris. If we chose Heraklion, our target population would consist of the salary and age of every Heraklian. One sample from this population would therefore be the age and the salary of one single Heraklian.

%We can abstract away its material existence.

In our first example it is relatively easy to imagine a population as a group of concrete, physical items, i.e. humans. However, most of the time such means to imagine populations will not be available. For our second example, consider the problem of translating into English a fragment of text written in Linear B. What would be our population? The answer would be, every possible fragment of Linear B text, together with its English translation, whether they physically exist or not. In fact, most of the samples in this population do not exist physically. We could pretend that they all exist buried somewhere in the Mediterranean coast and have not been yet discovered, but we do not need to, as we can abstract away their physical existence. The same could be said about the target population in the machine learning problem of translating a fragment of text written in Hindi into Spanish. This population should include every possible fragment of Hindi text and its Spanish translation, whether they have already been written or not.

%Existing fragments and their translations could be seen as materialisations of samples from an otherwise unmaterial population.

```{admonition} Question for you
:class: question1

Consider the problem of predicting the distance driven by a car moving at constant speed, using its speed and journey duration as predictors. What would be our target population?

1. This problem does not have a target population.

2. The values of the distance, speed and duration of every possible car journey.

3. All the cars that have existed and will ever exist.

Submit your response here: <a href="https://forms.office.com/e/gxdeFpu2ek" target = "_blank">Your Response</a>

```
% This is a regression problem, as the label takes on continuous values.

In the moving car problem, we are seeking to build a model that takes speed and journey duration as input predictors and produces a distance as the output label. It might be tempting to conclude that the population consists of all past, present and future cars. However, this would be wrong. To identify our population we need to first recognise the structure of one individual sample. Samples are abstractions described by a set of attributes. In the moving car problem we are considering samples that have three attributes, namely distance, speed, journey duration. Cars are therefore not samples and our target population is not a collection of cars. Our target population is instead the collection of triplets consisting of the values distance, speed and duration, of every possible car journey.

We have already discussed that, even though the moving car problem is a valid regression problem, we would never use machine learning to solve it. The reason for this is simple: we already know that we can compute the distance by simply multiplying speed and journey duration. The moving car problem is an example of a problem where we have a **perfect description of the population**, as we know the relationship between its attributes exactly. Using a perfect description of our target population, we can identify the model with the highest *deployment quality*. It is when we lack a perfect description of our population when it makes sense for us to use machine learning approaches. In fact, we could say that machine learning approaches use datasets to build an approximate description of our target population.




### From populations to datasets

The process by which samples are extracted from a population is known as **population sampling**. Understanding population sampling is very important in machine learning, as our starting point is that we lack a perfect description of our target population and because of it, we have no choice but to resort to datasets extracted from it. We sometimes say that machine learning treats datasets as population *surrogates*, which indirectly provide an imperfect description of our target population. The question is then, how can we use datasets to learn something useful about our target population?

% we consider problems where we lack a perfect description of our target population and all we have are datasets extracted from it.



<!-- ```{figure} images/PopulationSampling.jpg
---
name: PopulationSampling
---
Datasets are created by sampling a population, in other words, by extracting a collection of individual samples from it.
``` -->

```{figure} images/PopulationSampling.svg
---
name: PopulationSampling
---
Datasets are created by sampling a population, in other words, by extracting a collection of individual samples from it.
```


Let us discuss population sampling in the context of an already familiar problem, namely that of predicting the salary of an individual from their age. To build a regression model that predicts the salary of a Heraklian based on their age, we could sample the city of Heraklion by recording the salary and age of a group of its inhabitants. The collection of all the salaries and ages that we have recorded would form our dataset. Needless to say, our ultimate goal would be to build a model that predicts accurately the salary of any inhabitant of Heraklion picked at random, not just the salary of the Heraklians that we have included in our dataset. In other words, our goal is to be able to **generalise** what we have learnt from the dataset, to the entire population.


In order for us to be able to build solutions that generalise well, datasets need to be **representative** of our target population. First we have to ensure that all the samples in our datasets come from the same target population, e.g. no 19th century Parisians should be included in our Heraklian dataset. In statistical lingo, when our samples are extracted from the same population we say that they are **identically distributed**. Second, we need the samples in our dataset to be extracted following the same mechanism that will operate when the model is deployed against the target population. For example, rather than creating a dataset using the salaries and ages of Heraklians that belong to the same family or live in the same neighbourhood, we need the salary and age of Heraklians that have been **randomly** and **indepedently** drawn from the population. In other words, when extracting samples from our population, we must not impose any relationship between the samples that are being extracted. When we create a dataset following these two rules, we say that the samples in our dataset are **independent and identically distributed** or **IID**. Finally, our datasets need to have a **sufficiently large number of samples** so that we can reduce the risk of representing only partial segments of our population.



## The test task

In our {ref}`Intro` chapter we presented a basic machine learning model lifecycle consisting of two basic stages, namely the learning stage and the deployment stage. In addition to understanding what each stage is about, it is worth asking ourselves *who* is involved in each stage, i.e. who builds a model ready to be deployed and who will deploy a model that has already been built. In many scenarios this will be the same person or team, but in general we should not expect this to be the case. We can in fact see the machine learning world as an ecosystem consisting of multiple actors that produce, distribute and deploy models. In this ecosystem, whoever is responsible for deploying a model, whether they have built the model or not, needs to be confident that the model is good enough to be deployed.

The **test task is arguably the most important task in machine learning**, as it allows us to estimate the future deployment quality of any given model. Whether we create our own models and would like to quantify how good they are, or we are interested in deploying models that have already been built by others, we will always need to run a test task. In this section we will discuss the principles behind machine learning test tasks and how to interpret correctly the results of machine learning testing.


### True and empirical qualities

In our {ref}`Reg` chapter we defined regression as an optimisation problem where the best model was the one with the highest quality *on the training dataset*. We found this *Take 1* definition to be naive, as we soon discovered that models that work really well on a training dataset might perform poorly when deployed. Let us update our definition. From now on, **the best model will be defined as the one with the highest quality on the target population**, i.e. in the environment that our model is exposed to when deployed. This is our **Take 2** and final definition.

We will call the quality of a model on our target population its **true quality**. Given a model, its true quality is the metric that we would really like to know. Unfortunately, to know the true quality of a model we need a perfect description of our target population. In machine learning we assess the quality of a model using a dataset consisting of samples extracted from the target population. We call this second metric the **empirical quality** of our model, as it is obtained from indirect observations of our population, i.e. from a dataset. The question arises, what is the relationship between the true quality and the empirical quality of a model? The answer is, **the empirical quality of a model is an *estimation* of its true quality**. We sometimes reflect this relationship in our mathematical notation. For instance, if we define the notion of quality of a regression model using the MSE, we would denote the true quality of the model as $MSE$ and its empirical quality as $\widehat{MSE}$.

Once we have understood that an empirical quality should be interpreted as an estimation of a true quality, it is easier to establish whether a proposed quality metric is suitable or not. Let us look at the MSE and SSE used in regression. Back in our {ref}`Reg` chapter we suggested that we could consider both to be equivalent, but hinted at some differences in interpretation. Now we are in a position to explain what we meant. The notion of MSE can be defined both in a population and a dataset, as it makes sense to talk about an average squared error across all the samples of the population, and an average squared error across the samples of a dataset. By contrast, even though we can compute the SSE on a dataset, the SSE does not always make sense when considering the target population, as the population could potentially be infinite. Furthermore, whereas we can compare meaningfully MSE values computed on different datasets, the same cannot be said about the SSE. For instance if one dataset consists of a low number of samples and another of a large number of samples, the SSE computed on the former will be expected to be lower than SEE computed on the latter. This difference will not be due to differences in quality, but simply because of the difference in the number of samples. From now on, our quality metrics should be such that we will be able to define them both on a dataset and a population. Consequently, we will prefer the MSE to the SSE.


%the notion of SSE does not make sense for a population.
% The idea is simple: if a notion of quality on a dataset does not have an equivalent notion on the population, it should be avoided.


%We have already seen that we can use datasets to train models. What else can we do with datasets in machine learning?


### Testing as quality estimation

The test task in machine learning is the process by which we **assess the true quality** of a given model using datasets. Note that during testing we are not interested in knowing how the model was built or how it works internally, we are only interested in assessing how well it will perform when deployed. In fact, we can test models that have been built using non machine learning approaches. For testing purposes, models can be treated as **black boxes**.

A dataset used for testing purposes is known as a **test dataset** and the empirical quality of the model being tested is known as the **test quality** (see {numref}`PopulationSamplingTest`). Test datasets need to be **representative** and crucially **independent from datasets used for model training** (remember Ventris' decisive check?). As an empirical quantity, the test quality of a model is different from the true quality that we actually want to estimate. But how different?

% How should we interpret the value of a test quality?

```{figure} images/PopulationSamplingTest.svg
---
name: PopulationSamplingTest
---
A test task uses a dataset extracted from the target population (a *test dataset*) to assess the true quality of a model. The estimation that a test task produces is known as the *test quality*. The true quality can only be obtained if we have a perfect description of the population.
```

To correctly interpret the value of a test quality, we need to understand its **random nature**. We have already discussed that machine learning datasets consist of samples that are extracted randomly from the population. Because of this, any value computed on a machine learning dataset will be itself random. It is important to emphasise that the term *random* should not be interpreted in a colloquial sense, but strictly in a statistical one. By *random value* we do not mean just *any* value (colloquial sense). Instead, a random value should be seen as a range of values that can be obtained with some probability (statistical sense). A test quality is always computed on machine learning datasets and is therefore a random value. The specific value of the test quality that we obtain will be different from one dataset to another and in general, it will be different from the true quality. How close a test quality is to the underlying true quality will depend on how representative our dataset is.

%To correctly interpret a test quality, we need to remember that its value will in general be different from the true quality and will also be different from one dataset to another.


% Test and in general empirical metrics are random as the datasets from which they are obtained consist of samples extracted randomly from the population. 



% In other words, the numerical value of a test quality will not be just *any* value (colloquial sense), but we can expect it to lie within a range of values with some probability (statistical sense).


{numref}`PopulationSamplingTest3` illustrates the idea that a test quality is a random value. Consider a regression problem where we use the MSE as our notion of quality. Imagine we are given an already built model whose true quality is $MSE = 10$. The true quality is what we would like to know, however in machine learning we do not have direct access to it, so let us pretend we do not know this. If we extracted three separate IID datasets from the population, we would obtain three different test qualities, for instance, $\widehat{MSE}_1=9.5$, $\widehat{MSE}_2=11$ and $\widehat{MSE}_3=10.9$. These values illustrate the idea that we should never expect the different test qualities to be identical to one another nor equal to the true quality, although we should expect their values to be related to the true quality. 


```{figure} images/PopulationSamplingTest3.svg
---
name: PopulationSamplingTest3
---
Datasets are extracted randomly from the target population. Consequently, a test quality is also a random quantity. If we test the same model on three different datasets, the test quality will be different for each dataset.  
```


The random nature of the test quality has crucial implications when we use test tasks to compare different models.

```{admonition} Question for you
:class: question1

We are trying to decide which model to deploy out of three candidate models built by three separate teams, $A$, $B$ and $C$. We compute the test quality of each model on the same test dataset, as shown in {numref}`PopulationSamplingTest3model`, and obtain $\widehat{MSE}_A=100$, $\widehat{MSE}_B=99.9$ and $\widehat{MSE}_C=200$. Which model would we choose, the model built by team $A$, $B$ or $C$?

Submit your response here: <a href="https://forms.office.com/e/CuAFZEn5v7" target = "_blank">Your Response</a>

```



```{figure} images/PopulationSamplingTest3model_ABC.svg
---
name: PopulationSamplingTest3model
---
Models can be compared by estimating their deployment qualities using a test task. When comparing their test qualities, we need to remember they are random quantities.
```

According to the test quality, model $B$ appears to be the best, as it has the lowest test MSE. Is it however, the best? By now, we know that the test MSE is different from the true MSE and also that the test MSE, as an empirical quantity, is random. Could it be that model $B$ appears to be better than $A$ just by chance? In other words, when is a difference between two random values *significant*? Machine learning practitioners need to always be aware that random effects can mislead them. One of the best illustration of this is the so-called **Infinite Monkey Theorem**. According to this tongue-in-cheek theorem, if you have an unlimited number of monkeys typing in a room for long enough, at some point one of them will type the entire works of William Shakespeare. Would you say that a monkey that types a poem is in fact a poet? Would you not say that the poem has been entirely produced by chance? In machine learning, as in the Infinite Monkey Theorem room, randomness is always trying to trick you.


(Meth1_train)=
## The training task

Training tasks are the heart of model-building. As we already saw in our {ref}`Reg` chapter, training allows us to set the parameters of a tunable model using a dataset which we call the **training dataset**. Specifically, training uses optimisation approaches to find the *best* values for a model's parameters, according to a notion of quality defined *on the training dataset*.

As we know, populations and datasets can lead to different quality metrics and consequently, the best parameters according to a dataset and the population it comes from can be very different. In this section we will discuss in more detail model training and the crucial role of optimisation theory. Once again, we will think of our datasets as imperfect images, or surrogates, of the target populations from which they have been extracted.


### The error surface

In an optimisation problem we seek to find the best model, known as the **optimal model**, among a family of candidate models. An example of such a family is the family of linear models, which consists of all the straight lines that can be generated changing the values of the intercept and gradient parameters.

The notion of quality in optimisation theory is defined by a mathematical function known as the **cost function**, **loss function** or **error function**. We will use the more familiar term **error function** and will denote it by $E\{\boldsymbol{w}\}$, which could be read as *the error $E$ associated to the model with parameters $\boldsymbol{w}$*. The MSE of a regression model on a dataset is an example of an error function: for every model defined by a set of parameters $\boldsymbol{w}$, there is one error value $E\{\boldsymbol{w}\}$.

%, although you should note that optimisation does not require error functions to be defined on datasets.

%We just need the error function $E\{\boldsymbol{w}\}$ to associate an error to each model that we might consider.

Using mathematical notation, we can define the optimal model as the model whose parameters $\boldsymbol{w}_{opt}$ are

$$
\boldsymbol{w}_{opt} = \underset{\boldsymbol{w} \in W}{\operatorname{argmin}} E\{\boldsymbol{w}\}
$$(eqMinimE)

In words, the optimal model is the one that has the lowest error among the family of models $W$. To understand optimisation methods, it is sometimes useful to visualise the error function as an **error surface**. Although we can only visualise error surfaces for simple models that have one or at most two parameters, the intuition that we can extract from this low-dimensional visualisation is essential for understanding more complex scenarios.

{numref}`ErrorSurface1` shows the error surface of a model defined by two parameters $w_0$ and $w_1$. The horizontal plane formed by axes $w_0$ and $w_1$ is known as the **parameter space**. A point of coordinates $(a,b)$ in the parameter space corresponds to one single model, whose parameters are $w_0=a$ and $w_1=b$. The elevation above each point in the parameter space corresponds to its error. In this representation the error values are also colour coded.



```{figure} images/TrueError.svg
---
name: ErrorSurface1
---
Error surface for a model defined by two parameters $w_0$ and $w_1$.
```

An alternative representation of the error function consists of colour-coded or countour maps on the parameter space. {numref}`ErrorSurfaceMap` and {numref}`ErrorSurfaceContour` represent the same error surface as {numref}`ErrorSurface1` as respectively a colour-coded map and a contour map.


```{figure} images/TrueErrorColourOpt.svg
---
name: ErrorSurfaceMap
---
Error surface for a model defined by two parameters $w_0$ and $w_1$ represented as a colour-coded map. The optimal model, which has the lowest error, is identified by the symbol $\times$.
```

```{figure} images/TrueErrorContourOpt.svg
---
name: ErrorSurfaceContour
---
Error surface for a model defined by two parameters $w_0$ and $w_1$ represented as a contour map. The optimal model is identified by the symbol $\times$.
```

Moving slowly across the parameter space can be interpreted as changing gradually the values of the parameters of a tunable model. In {numref}`ErrorSurfaceContourWalk` we visit a sequence of models by keeping $w_1$ constant and changing $w_0$ in small steps. As we step from one location in the parameter step to the next, we visit models that have a different error.


```{figure} images/TrueErrorWalk.svg
---
name: ErrorSurfaceContourWalk
---
By keeping $w_1$ constant and increasing $w_0$ in small steps, different locations in the parameter space are visited, which correspond to different models. The error of each model is represented as a vertical line.
```

Visualising the error surface allows us to easily identify the values of the parameters that define the optimal model. However, we must not be misled into thinking that this is how we identify the optimal model in optimisation. Given a model with parameters $\boldsymbol{w}$, we can use the error function $E\{\boldsymbol{w}\}$ to obtain its associated error. However this does not mean that we have the error of every single model at hand, nor a visualisation of the error surface. In other words, in general optimisation assumes we do not *see* the error surface, let alone the optimal model. The question is then, how can we use what we know about $E\{\boldsymbol{w}\}$ to find an optimal model?

(optimalGradient)=
### Looking for the optimal model


As we move across the parameter space, we visit different models with parameters $\boldsymbol{w}$ whose error is given by $E\{\boldsymbol{w}\}$. We can ask ourselves, how much will the error change if we change the values of $\boldsymbol{w}$ slightly, or visually, if we take a small step in any direction in the parameter space? This is what we know as a **directional derivative** and can be computed using the error surface $E\{\boldsymbol{w}\}$. There is one direction along which the error increases the most, and this is known as the  **error gradient**, denoted by $\nabla E\{\boldsymbol{w}\}$. The gradient of an error surface can also be obtained from the error funcion $E\{\boldsymbol{w}\}$ but for now, do not worry too much about how to do this and let us simply assume that we can compute it for every set of parameters $\boldsymbol{w}$.

If we happen to be at the location of the **optimal model** in the parameter space, $\boldsymbol{w}_{opt}$, and move in any direction, the error will always increase as by definition, the optimal model is the one with the lowest error. As a consequence of this, the gradient at the optimal model is always zero

$$
\nabla E\{\boldsymbol{w}_{opt}\}=0
$$(GradZero)

Therefore, if we can easily identify which models have zero gradient, one of them will be the optimal model. Unfortunately, in general we will not be able to find an exact solution for {eq}`GradZero` that will return the parameters of the optimal model $\boldsymbol{w}_{opt}$. The question is then, how can we find the parameters of a model whose gradient is zero? A popular method that takes advantage of the information provided by the error surface gradient is **gradient descent**. Gradient descent is a numerical optimisation method where we improve iteratively our model until we find a solution whose gradient is close to zero.

In gradient descent, we move across the parameter space following the direction along which the error decreases the most. This direction happens to be the opposite of the direction given by the gradient. Our hope is to eventually reach a point in the parameter space from which we cannot improve the error any further, in other words, to reach a point where the gradient is zero. In gradient descent we start from an initial model, which is usually chosen randomly. During each iteration, the parameters are updated using the following rule:

$$
\boldsymbol{w}_{new} = \boldsymbol{w}_{old} - \epsilon \nabla E\{\boldsymbol{w}_{old}\}
$$(eqGradientUpdate)

where $\boldsymbol{w}_{new}$ are the updated parameters, $\boldsymbol{w}_{old}$ are the previous parameters and $\epsilon$ is the step size or learning rate. The step size $\epsilon$ indicates how far from $\boldsymbol{w}_{old}$ we move following the direction opposite to the gradient. The choice of $\epsilon$ is important. Small values result in slow convergence to the optimal model, whereas large values risk overshooting the optimal model. An illustration of gradient descent is shown in {numref}`ErrorSurfaceContourGradientDescent`.


```{figure} images/TrueErrorGradOpt.svg
---
name: ErrorSurfaceContourGradientDescent
---
Gradient descent is an iterative process to find the optimal model defined by an error surface. Starting from a random location in the parameter space, the parameters are updated iteratively following the direction oposite to the error surface gradient.
```

In general, gradient descent might never reach a model whose gradient is exactly a zero, hence it is always necessary to include a stopping strategy. Common choices include setting a maximum number of iterations or longest processing time, reaching an acceptable error value and observing a small relative change in the error from iteration to iteration.

It is important to highlight that even though the error gradient is zero at the optimal model, a **zero error gradient does not guarantee that the model is the optimal**. {numref}`ErrorSurface2` shows a complex error surface for which there exist three models whose gradient is zero. Only one of them has the lowest error. This model is known as the **global optimum** whereas the other two models are **local optima**.


```{figure} images/TrueSurfaceComplex.svg
---
name: ErrorSurface2
---
Complex error surface that has two local optima and one global optimum. Local and global optima all have zero gradient.
```

If we use gradient descent on a complex surface such as the one shown in {numref}`ErrorSurface2`, we will progressively get closer to an optimum model, but there is no guarantee that this will be the global one. We describe this scenario saying that gradient descent can get stuck in a local optimum. Because of this it is common to run gradient descent multiple times from several randomly selected initial models, as shown in {numref}`ErrorSurface2Grad`. After visiting multiple optima, we can select the best among them. This does not guarantees that we might have reached the global optimum, but at least it reduces the risk of getting stuck in a bad local optimum.


```{figure} images/TrueErrorGradComplexOpt.svg
---
name: ErrorSurface2Grad
---
Running multiple times gradient descent on a complex surface increases the chances of finding a good local optimum.
```

### True and empirical error surfaces

We have previously distinguished between the quality of a model on the target population (the *true quality*) and the quality of the model on a dataset extracted from the population (the *empirical quality*). In the context of model training it is also useful to distinguish between the **true error surface** and the **empirical error surface**.

%Given a tunable model defined by a set of parameters, t

Given a family of models defined by a parameter vector $\boldsymbol{w}$, the **true error surface** corresponds to the error associated to each model when *deployed against the target population*. The optimal model defined on the true error surface is the model that we would like to find. In machine learning we do not have access to the true error surface, as this requires having a perfect description of our population. Using a dataset extracted from the population we can obtain the **empirical error surface**, which corresponds to the error of each model *on the dataset*. Using the empirical error surface, we can identify an optimal model, which is the model that performs the best on the dataset. The **empirical error surface can be seen as an *estimation* of the true error surface**. In general, we should expect them to be different, as illustrated in {numref}`TrueEmpiricalErrorSurface` and {numref}`TrueEmpiricalErrorSurfaceContour`.

```{figure} images/TrueAndEmpirical.svg
---
name: TrueEmpiricalErrorSurface
---
The true error surface (transparent) and the empirical error surface (opaque) are different. The more representative is our training dataset, the closer are both error surfaces.
```

```{figure} images/TrueAndEmpiricalContourOpt.svg
---
name: TrueEmpiricalErrorSurfaceContour
---
The optimal model defined by the true error surface ($\times$) is different from the one defined by the empirical error surface ($\star$). We are would like to find $\times$, but can only hope to find $\star$ as we are using a dataset to train our models.
```

Since the true error sutface and the empirical error surface are in general different, we should expect the optimal model according to the true error surface, which is the model that we *would like to* find, to be different from the optimal model according to the empirical error surface, which is the model that we *can* find. In general, the similarity between the empirical and the true error surfaces, and hence the closeness between their optimal solutions, depends on how representative our training dataset is.


### Optimisation on the empirical error surface

A training dataset together with a family of models characterised by a parameter vector $\boldsymbol{w}$, define an empirical error surface $E\{\boldsymbol{w}\}$. The simplest method that we could think of to identify the optimal model defined by the empirical error surface, consists in directly evaluating as many choices of $\boldsymbol{w}$ as possible and selecting the one with the lowest error. This method is called **exhaustive** or **brute-force** and is illustrated in {numref}`ErrorBrute`.

% defined by a parameter vector $\boldsymbol{w}$

```{figure} images/TrueErrorBrute.svg
---
name: ErrorBrute
---
By evaluating the empirical error of many models, brute-force methods hope to find one that provides a good solution. Evaluated models are identified in the parameter space using the symbol $\times$.
```

Brute-force methods are simple, but in most cases impractical, especially when the error surface is very complex and the number of parameters is large.

A second avenue to identify the optimal model on the empirical error surface is to directly solve the zero-gradient equation {eq}`GradZero`. This approach can be succesfully applied in some cases, for instance to find the optimal linear model using the MSE as the empirical error function. Let $\boldsymbol{X}$ and $\boldsymbol{y}$ be the design matrix and the label vector of the training dataset. As we know, the predicted label vector can be calculated from $\boldsymbol{X}$ and the parameter vector $\boldsymbol{w}$ as

%An example of an empirical error surface is the MSE of a family of linear models with parameters $\boldsymbol{w}$ on a training dataset described by a design matrix $\boldsymbol{X}$ and a label vector $\boldsymbol{y}$. Using vector notation, we can obtain a mathematical expression for this error function. First, we note that the predicted labels $\hat{\boldsymbol{y}}$ can be obtained as

$$
\hat{\boldsymbol{y}} =
\boldsymbol{X}\boldsymbol{w}
$$(eqPredLabels)

Using {eq}`eqPredLabels`, the prediction error $\boldsymbol{e}$ can be obtained as

$$
\boldsymbol{e} =
\boldsymbol{y}-\hat{\boldsymbol{y}} = \boldsymbol{y}-\boldsymbol{X}\boldsymbol{w}
$$(eqPredErr)

Let us denote the MSE empirical error function by $E_{\widehat{MSE}}\{\boldsymbol{w}\}$, where we are using the *hat* notation $\widehat{MSE}$ to indicate that as an empirical quantity, it is obtained from a dataset. We can express $E_{\widehat{MSE}}$ in terms of the error vector $\boldsymbol{e}$ as follows:

$$
E_{\widehat{MSE}}\{\boldsymbol{w}\} &= \frac{1}{N}\boldsymbol{e}^T\boldsymbol{e}\\
&= \frac{1}{N}(\boldsymbol{y}-\hat{\boldsymbol{y}})^T(\boldsymbol{y}-\hat{\boldsymbol{y}}) \\
&= \frac{1}{N}(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{w})^T(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{w})  
$$(eqMSEemp)

Note that {eq}`eqMSEemp` defines a computation such that, given a parameter vector $\boldsymbol{w}$ defining a model together with a dataset defined by a design matrix $\boldsymbol{X}$ and a label vector $\boldsymbol{y}$, one MSE value is returned. Using {eq}`eqMSEemp` we can derive the following mathematical expression for the gradient of the empirical error surface:

%Moreover, this computation involves processing the entire training dataset, which is represented in the design matrix $\boldsymbol{X}$ and label vector $\boldsymbol{y}$. Consequently, as we increase the number of predictors and the number of samples in our training dataset, the time taken to calculate the error of one model will also increase.

%The simplest way for us to find the optimal model defined by the empirical error surface $E_{\widehat{MSE}}\{\boldsymbol{w}\}$ would be to consider as many models in the parameter space as possible, compute their MSE and select the model that has the lowest MSE. This is known as an exhaustive approach and although simple, it is computationally costly and in general impractical. A second approach would be to try to identify the model that has a zero gradient. Using the mathematical expression for the error surface $E_{\widehat{MSE}}\{\boldsymbol{w}\}$, we can derive the following expression for its gradient:

$$
\nabla E_{\widehat{MSE}}\{\boldsymbol{w}\} = \frac{-2}{N}\boldsymbol{X}^T(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{w})
$$(eqMSEempGradient)

As we know, the gradient at the optimal model is zero. Fortunately, we can solve the equation $\nabla E_{\widehat{MSE}}\{\boldsymbol{w}_{opt}\}=0$, and use this solution to calculate the parameters $\boldsymbol{w}_{opt}$ of the optimal model defined by the empirical error surface. The following derivations will take us, step by step, from the zero gradient equation to the computation of $\boldsymbol{w}_{opt}$:

$$
\nabla E_{\widehat{MSE}}\{\boldsymbol{\boldsymbol{w}_{opt}}\} &= 0\\
\boldsymbol{X}^T(\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\boldsymbol{w}_{opt}}) &= 0 \\
\boldsymbol{X}^T\boldsymbol{y}-\boldsymbol{X}^T\boldsymbol{X}\boldsymbol{\boldsymbol{w}_{opt}} &= 0\\
\boldsymbol{X}^T\boldsymbol{X}\boldsymbol{\boldsymbol{w}_{opt}} &= \boldsymbol{X}^T\boldsymbol{y} \\
(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{X}\boldsymbol{\boldsymbol{w}_{opt}} &= (\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{y} \\
\boldsymbol{\boldsymbol{w}_{opt}} &= (\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}^T\boldsymbol{y}
$$(eqLeastSqDer)

The solution {eq}`eqLeastSqDer` is precisely the **least squares** solution that was presented in the {ref}`Reg` chapter. Note that we have obtained the least squares solution by solving the equation $\nabla E_{\widehat{MSE}}\{\boldsymbol{w}_{opt}\}=0$ **exactly**. Unfortunately, solving the zero-gradient equation {eq}`GradZero` exactly is only possible in a limited number of cases. In the majority of the scenarios we will need to use other optimisation methods, such as gradient descent.

To run gradient descent to train a model, it is necessary to calculate the gradient of the empirical error surface in every iteration. Unfortunately, calculating the gradient of the empirical error surface can be very costly when the training dataset is very large. One simple approach to reduce the computational time is to calculate the gradient using a random subset of the training dataset, known as a **batch**. The resulting gradient is in fact an *estimation* of the actual gradient of the empirical error surface. If the batch that we use is close to the entire training dataset, the estimated gradient will be close to the actual gradient; if it is very small, it will deviate from the it. In other words, using batches to estimate the gradient produce **noisy gradients**. For this reason, this is known as **stochastic gradient descent**.


### Overfitting and regularisation

%The notions of true and empirical surfaces give us another angle from which to look at overfitting. In general, the true and empirical error surfaces are different. Furthermore, since training datasets are extracted randomly from the target population, the empirical error surface will be different for different training datasets, and so will the final optimal model. These discrepancies are further exacerbated when the number of model parameters is large and the training dataset is small. By increasing the size of the training dataset, empirical surfaces get closer to the true error surface and the risk of overfitting decreases.

We can design approaches that reduce the risk of overfitting by exploiting our understanding of the error surface. **Regularisation** is one such approach and consists in modifying the error surface by adding a new term that effectively restricts the flexibility of our model. For instance the MSE empirical error surface can be modified as follows

$$
E_{\widehat{MSE}-R}\{\boldsymbol{w}\} = \frac{1}{N}\boldsymbol{e}^T\boldsymbol{e} + \lambda \boldsymbol{w}^T\boldsymbol{w}
$$(eqMSEReg)

where $\boldsymbol{w}^T\boldsymbol{w}$ is the sum of the squares of the model's parameters and the value of $\lambda$ controls the relative importance of the error term compared to the parameters term. Overfitting models tend to have large parameters $\boldsymbol{w}$. Since the term $\lambda \boldsymbol{w}^T\boldsymbol{w}$ in {eq}`eqMSEReg` effectively penalises models that have large parameters, by regularising the MSE empirical error surface models that overfit are penalised.

The optimal solution for the regularised error surface $E_{\widehat{MSE}-R}$ will strike a balance between reducing the prediction error on the training dataset and preventing the coefficients from taking on large values, which adds rigidity to our solutions. We can show that {eq}`GradZero` can be solved exactly when we use the regularised MSE as the empirical error surface, obtaining

$$
\boldsymbol{w}_{opt} = (\boldsymbol{X}^T\boldsymbol{X}+N\lambda\boldsymbol{I})^{-1}\boldsymbol{X}^T\boldsymbol{y}
$$

where $N$ is the number of samples in the dataset and $\boldsymbol{I}$ is the identity matrix. Note that when $\lambda=0$ we obtain the least squares solution without regularisation.



### Optimisation cost and target quality

Regularised error surfaces are used for training purposes only, i.e. to find the values of the parameters of a model. The regularised error does not represent, however, our notion of quality, as it includes a second term whose purpose is to control the complexity of the final solution. When we use the regularised MSE empirical error surface {eq}`eqMSEReg` to train a model, we are asking ourselves, *find the model that has the lowest MSE on this dataset, but make sure its coefficients are not too large*. In other words, we are adding constraints to our problem. After **training a regression model using a regularised MSE**, if we want to assess its future deployment performance we need to **test it using the conventional definition of MSE**. This should sound strange, as we are using a notion of quality during training that is different from our notion of deployment quality.

In future chapters we will come across other examples where the quantity that we are optimising does not directly correspond to our notion of quality during deployment. This could be due to including other constraints in our problem formulation, as in regularisations, or because it migth be difficult to formulate mathematically an optimisation problem using the notion of quality during deployment. To illustrate the latter scenario, imagine a company that plans to increase their sales volume using a machine learning model that segments their customers into different groups. Formulating a machine learning problem using the sales volume as the quality metric that needs to be optimised would be very difficult, if not impossible. Instead, they would formulate a different problem using a quality metric that they would hope is related to the sales volume.

To distinguish between the notions of quality during training and deployment, we will reserve the term **quality to describe our target quality during deployment**, and we will use the terms **cost, loss or error to refer to the quantity that we want to optimise during training**. Sometimes both quantities will be the same, but in general, this will not be the case. Our hope will be that the solution obtained during optimisation will also be the one that has the highest quality during deployment.




## The validation task

% In machine learning we assume that the attributes of a population follow an **underlying pattern** and suffer from deviations from this pattern that are considered to be irrelevant, i.e. **noise**. From this prespective, model training should discover the underlying pattern in a dataset, while ignoring those irrelevant deviations.

% Models need to be flexible enough to be able to capture the complexity of the underlying pattern. If our models are too rigid, they will not be capable of reproducing complex patterns. On the other hand, if they are too flexible, the risk of memorising irrelevant details, i.e. of overfitting to our dataset, will increase. Ideally, we should know the complexity of the underlying pattern, so that we can select and train models that have the necessary flexibility. The question arises, wow can we determine the complexity of the underlying pattern?

In machine learning there are many families of models available to us to solve any given problem. Each family of models can produce solutions of different shapes and degrees of complexity and ideally, we should be selecting the right one for each problem at hand. Unfortunately, in machine learning we usually lack any previous insight that would guide us in choosing the right family of models. For instance, consider a simple polynomial regression problem. Changing the value of the degree of the polynomial $D$ leads to subfamilies of models of different complexity. A linear model is a polynomial model where $D=1$ and is very rigid, quadratic models are polynomial models where $D=2$ and are more flexible, and so on. The question would be, how do we identify the right value of $D$, in other words, what is the complexity of our problem? Validation tasks allow us to identify the  complexity of the problem and select the right family of models. By producing an estimation of the deployment quality of several family of models, validation tasks allow us to choose the family that we will train, test and hopefully deploy. As usual, to conduct validation we need a dataset extracted from our population (see {numref}`PopulationSamplingValidation`).

% There are situations where we would like to be able to explore different families of models, so that we can select and train the right one. Selecting a family of models that has the right complexity for a specific machine learning problem is one of such cases.

```{figure} images/PopulationSamplingValidation.svg
---
name: PopulationSamplingValidation
---
Validation tasks use datasets extracted from the population to assess the potential of a family of models to solve a problem.
```

Validation involves training each of the families of models that we are considering. It is important to emphasise that **models that are trainined during a validation task are not meant to be deployed**. Training a family of models is a prerequisite for us to be able to **estimate their deployment quality**. Once we have estimated the deployment quality of each family of models, we can select the best one and train it ready for test and deployment. There are three main validation methods: validation set, leave-one-out cross-validation (LOOCV), and $k$-fold cross-validation.


The **validation set** approach is the simplest one and consists of one single round of training followed by deployment quality estimation. Before validation, the available dataset is split into a training dataset and a validation (or hold-out) dataset ({numref}`ValidationSet`).

```{figure} images/ValidationSet_1.svg
---
name: ValidationSet
---
In the validation set approach, the available dataset is split into two sets, one of them is used for training each family of models, the other to estimate their deployment quality.
```

To split the available dataset, we need to specify what fraction of samples will be assigned to each split and the assignment should be done randomly. Unfortunately, there is no general rule to decide which fraction of samples should be assigned to each split. Hence, we might ask ourselves, how good is our estimation of the future deployment quality of each family of models? If the validation set is small, the estimation itself will be poor. If it is large, the estimation will be better, however the number of samples used for training will be low and therefore, we will have a poorly trained model. Consequently, we will have a good estimation of the deployment quality of a model that has been poorly trained. In both extreme cases, the estimation of the deployment quality of each family of models will not be very reliable. In other words, it might not reflect the potential of each family of models to solve the machine learning problem.


In **LOOCV** we conduct multiple rounds of training and deployment quality estimation. During each round, we use one sample for deployment quality estimation and the remaining samples for training - hence the name *leave-one-out*. During the first round we leave the first sample out, during the second round the second sample and so on until the last round, where we leave the last sample out ({numref}`LOOCV`). Therefore, there are as many training and deployment quality estimation rounds as there are samples.


```{figure} images/LOOCV.svg
---
name: LOOCV
---
In leave-one-out cross-validation $N$ rounds of training and deployment quality estimation are conducted, where $N$ is the size of the available dataset. In each round, only one sample is used to assess the deployment quality and at the end, an average is computed.
```

After completing all the rounds, we collect the individual estimations of the deployment quality produced by each round and compute an average. The resulting figure is the final estimation of the deployment quality of the family of models under investigation. Compared to the validation set approach, LOOCV is computationally intensive, as it requires one round of training and performance estimation per sample in the dataset, whereas the validation set approach consisted of one single round. On the other hand, models are trained using most of the available samples and hence, are better trained.

The third validation approach is **$k$-fold** cross validation. This validation approach randomly splits the available dataset into $k$ subsets, also known as *folds*. Then it runs $k$ training and deployment quality estimation rounds. During each round, one of the folds is used for deployment quality estimation and the remaining folds for training ({numref}`kfoldCV`).

```{figure} images/kfoldCV.svg
---
name: kfoldCV
---
$k$-fold cross validation condicts $k$ rounds of training and deployment quality estimation, by splitting the dataset into $k$ subsets or folds.
```

We have already seen that although simple and computationally inexpensive, the validation set approach produces poor deployment quality estimations. Since the validation set approach uses a fraction of the available samples, the trained models will be worse than models trained using all the available samples. Therefore, the deployment quality estimation tends to be too pessimistic. LOOCV uses most of the samples for training, and therefore suffers less from overpessimistic estimations, at the expense of increased computational cost. $k$-fold cross validation offers a trade-off that reduces the computational cost by reducing the number of rounds.

%Furthermore, even though $k$-fold cross validation produces estimations that are more pessimistic than LOOCV, LOOCV produces estimations that have higher variance.


## Summary and discussion

In machine learning we seek to build solutions to problems that involve a target population, and our main challenge is to build such solutions without having access to a perfect description of the population. Instead of a perfect description of the population, we assume that have access to datasets extracted from it. Understanding the relationship between populations and datasets is fundamental in machine learning. Since we use datasets as surrogates of our target populations, our datasets have to be **representative**. To achieve this, our datasets need to be **IID** and have a **sufficiently large** number of samples.

%Samples from a population follow a pattern, which we want to capture, and suffer from deviations from this pattern due to external random factors. Our datasets reflect both patterns and deviations, and our job will be to tell them appart.

To create machine learning models and assess how well they work, we have a methodology which we need to follow rigorously. Perhaps the most important task in this methodology is the **test task**. A test task allows us to evaluate the future deployment quality of an already built model, using a test dataset and a notion of quality. **Test datasets need to be independent from training datasets**, otherwise we risk falling into a data trap and fool ourselves. In addition, to correctly interpret a test quality, it is important to remember that the test quality is random, due to the random nature of the test dataset. **Training tasks** use optimisation approaches to identify the best model according to a notion of quality on a training dataset, which we call empirical ***cost*, *loss* or *error***. Gradient descent is an example of an optimisation method, which navigates the empirical error surface in search of the optimal model. We need to be mindful, however, that the the optimal model defined by the empirical error surface might be different from the one defined by the true error surface, which is the one that we would like to find. Finally, **validation tasks** can be used to compare different families of models and asses which one might be more suitable to be trained to solve a particular problem.

Understanding the difference between **true and empirical quantities** is essential in machine learning, as we only have direct access to the latter, but would like to know the former. A test quality and a training error are both empirical quantities defined on datasets, and can be seen as estimations of the true ones defined on the population. Furthermore, the notions of quality during training and deployment are not always the same. This is why we sometimes distinguish between deployment **quality** and training **cost**. Not everyone makes this distinction explicit, hence when reading reports on machine learning projects, we might need to find it out by ourselves.


We started this chapter remembering Ventris' decisive check and its role in the decipherment of the Linear B script. Indeed, Ventris' decisive check illustrates one of the most important red lines in the machine learning methodology: never test a machine learning model using samples that you have used for training. There are many other details of the story of the dechiperment of Linear B that resonate with machine learning. We suggested that dechipering Linear B should be an impossible proposition, as we are looking at fragments of text of uknown contents, written in an unknown script, encoding an unknown language. Other ancient scripts were only deciphered thanks to the discovery of multilingual documents, which allowed researchers to access the contents of fragments of text and speculate about the sounds represented by its symbols. The Rosetta stone's role in deciphering Egyptian hieroglyphs is without doubt the best example of this. Ventris came up with his solution after **hypothesising** that the language encoded in Linear B tablets was an archaic variant of classical Greek. This hypothesis went against the general consensus at the time and constituted a leap in the dark, but turned out to be correct. The important point here is that, as a hypothesis, the Greek nature of the language encoded in Linear B was not a conclusion from analysing his data, it was Ventris' starting point. In machine learning we sometimes make similar choices and adopt angles that are not suggested by our data. These choices are known as hypotheses.

An excellent understanding of our problems and the domain they belong to is fundamental to come up with sensible hypothesis. Remember our first top tip: **know your domain!** Ventris' success can be ascribed to the amount of efforts he put into deciphering Linear B, but also to his excellent knowledge of the ancient world, ancient languages and linguistics. Without this background, he would not have been able to decipher Linear B. Interestingly, Ventris also acknowledges that to decipher an unknown script, it is essential to have sufficient material. This also resonates with our discussion about datasets: we need a sufficiently rich collection of samples, in order for our datasets to be representative.

If you are dissapointed that you have not had the opportunity to decipher the Egyptian hieroglyphs or the Linear B script, do not worry, there are still a few writing systems that await decipherment. Examples include the instance the Proto-Elamite, the Rongorongo or the Voynichese scripts. Why not giving them a try?



% Educated guesses

% Change your mind


% Cost function vs quality

% Splitting datasets

% Overfitting can indeed be seen as a situation where the empirical and true error surfaces are so different, that the optimal empirical model is very far from the true optimal model. In addition, we could imagine different datasets producing empirical error surfaces which are very different from one another. Etc.

% Bias vs variance?
