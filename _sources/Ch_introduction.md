% What should we do with hyperlinks and references? For instance "in this study..." should we make it print-friendly?
% Same for hyperlinks to books Goodfellow... and James...
% Heart rate of a rabbit: question box
% Correct figure so that it contains the error with the Syrian bear
% Glossary: item, sample, point, attribute, attribute space, predictor, label, feature
% Probability question in unsupervised learning
% flatworm question


(Intro)=
# Introduction

In this chapter we first define machine learning and discuss the dataset-first and deployment-first views of machine learning. Then, a taxonomy that organises machine learning problems and techniques into different families is introduced. Understanding this taxonomy will allow us to cast real-world problems into machine learning problems and select the right approaches to solve them. Next, the scope of machine learning and its relation to other disciplines, including statistics and computer science, is discussed. Finally, we conclude this chapter discussing the organisation of this book and sharing with you our first top tip.


(Intro1)=
## From mouse to whale, through rabbit

Behind a façade of simplicity hides a complex and truly fascinating biological system: the heart. It is astonishing to learn that the billions of cells that form the heart not only contract uninterruptedly and in perfect synchrony during the entire lifetime of the organism: they do so at the required rate. Our heart rate is on average around 80 beats per minute (bpm), that is if we are referring to adults, as the heart of an infant beats faster. When we relax our heart rate goes down, it goes up when we exercise. Stop for a second and think of billions of individual cells contracting at unison to meet the instantaneous demand of your body. Truly amazing.

Humans are, of course, not the only living organisms equipped with a heart, other animals do too and they have their own heart rates as well. You might be familiar with the basic observation that smaller animals have a faster heart than bigger ones, in the same way as children (small humans) have faster hearts than adults (big humans). Using this qualitative observation we would be able to conclude, for instance, that a rabbit's heart should surely be faster than our heart. This is indeed the case, but how fast is it? In principle the only way for us to find out would be to get hold of a rabbit, feel her pulse, and count the number of beats that she produces within a minute. Can you think of a way for us to find the heart rate of a rabbit that does not involve *measuring* it?

Understanding the differences in heart rate across different animal species is an interesting scientific question that many researchers have explored in the past. In a study by [Noujaim et al](https://www.ahajournals.org/doi/10.1161/01.CIR.0000146785.15995.67) you can find the average heart rate of several animal species together with their body mass, from the tiny wild mouse to the massive humpback whale. We have plotted these values in {numref}`HrvsBM`. Go ahead and have a look at them.

```{figure} images/HRvsBM_earth.svg
---
name: HrvsBM
---
Heart rate and body mass of several animal species, from the tiny wild mouse (top left) to the massive humback whale (bottom right). Note that the body mass axis uses a logarithmic scale, where g stands for *gram*, kg for *kilogram* and t for *ton*. Data from [Noujaim et al](https://www.ahajournals.org/doi/10.1161/01.CIR.0000146785.15995.67).
```

{numref}`HrvsBM` supports our earlier understanding that smaller animals have a faster heart rate. At one end of the graph we have the wild mice, weighting barely more than 20 g and beating at nearly 500 bpm. At the other we have the humpback whale, which weights 30 t and has a heart that beats at 30 bpm. Between the wild mouse and the humpback whale, the heart rate decreases as the body mass increases. Let us get back to the heart of a rabbit.

<!-- Can you use {numref}`HrvsBM` to provide a quantitative *guess* for a rabbit's heart rate? Would you say it is lower than 100 bpm? Between 100 bpm and 300 bpm? Higher than 300 bpm? -->

```{admonition} Question for you
:class: question1

Can you use {numref}`HrvsBM` to provide a quantitative *guess* for a rabbit's heart rate? Would you say it is lower than 100 bpm? Between 100 bpm and 300 bpm? Higher than 300 bpm?

Submit your response here: <a href="https://forms.office.com/e/xtd6zLArg7" target = "_blank">Your Response</a>

```

If your guess was somewhere between 100 bpm and 300 bpm, you guessed right. In fact, it is around 250 bpm. The point is not, however, what the right answer is, but *how we arrived at it*. Note that the heart rate of a rabbit is a quantity that can be measured. However, instead of measuring it, you just did something else and arrived to a quantitatively correct answer. So what did you do? Give it a few moments to reflect about it.

This is what you migth have done. First you have observed there is an association between body mass and heart rate. Then you must have thought that the body mass of a rabbit is somewhere between, say 1 kg and 10 kg. Looking at {numref}`HrvsBM` you have noted that animals that weight 1 kg have on average a heart rate of 300 bpm, whereas animals that weight 10 kg have on average a heart rate of 100 bpm. Conclusion: the heart rate of a rabbit must be somewhere between both values, i.e. between 100 bpm and 300 bpm.

Is this, or something along these lines, what you did? If this is the case congratulations: **you have just solved a machine learning problem**. Simple as it may look, this problem contains all the ingredients that you will find in any machine learning problem. Note that we could have asked ourselves about the heart rate of any animal species and following the same procedure, we would have been able to obtain a sensible guess. Thus, the procedure that we have just created allows us to map the body mass of an animal species to its heart rate. Crucially, we have solved this problem using previous observations consisting of the body mass and heart rate of *other* animal species. In other words, we have used **data**.


(Intro2)=
## What is machine learning?

In the previous section we used a collection of observations consisting of the body mass and heart rate of different animal species, to build a mechanism that can map the body mass of *any* animal to its heart rate. Why did we declare that we had solved a machine learning problem?

Let us first read the words of some of the most influential machine learning experts. According to [Goodfellow, Bengio and Courville](https://www.deeplearningbook.org/), machine learning is the *ability to acquire **knowledge**, by extracting patterns from raw **data***. [James, Witten, Hastie and Tibshirani](https://www.statlearning.com/) define machine learning as *a set of tools for **modeling** and **understanding** complex **datasets***. Note that both definitions include two main components:
1. Data / datasets.
2. Knowledge / modeling / understanding.

Hence, if we want to understand what machine learning is, we need to first explore what we mean by data and knowledge.


### What is data?

Data can be defined as the **materialisation** of an action, an observation or a measurement. A [clay tablet containing symbols written in an ancient script](https://en.wikipedia.org/wiki/Linear_B) is data, and so is the collection of bits stored on a hard drive that encode a digital picture.  

In machine learning we use data formatted as **datasets**. Datasets are collections of **items** described by the same set of pre-defined **attributes**. Each attribute has a type that can be simple or complex:
- Simple attributes incude **continuous** values (e.g. a temperature, a stock price) and **discrete**, also known as **categorical**, values (e.g. a symbol in a writing system).
- Complex attributes can be seen as collections of simple values (e.g. a digital image consisting of RGB pixels).

Datasets can be represented as tables, where each row corresponds to an item and each column to one of the attributes. For instance, {numref}`HRcsBMTable` represents a dataset consisting of the body mass and heart rate of 3 animal species, namely the wild mouse, the rabbit and the humpback whale.

```{list-table} Body mass and heart rate of three animal species
:header-rows: 1
:name: HRcsBMTable

* - Species (ID)
  - Body mass [g]
  - Heart rate [bpm]
* - Wild Mouse
  - $22$
  - 480
* - Rabbit
  - $2.5\times 10^3$
  - 250
* - Humpback whale
  - $30\times 10^6$
  - 30
```
Note that the first column in {numref}`HRcsBMTable` corresponds to the item's identifier and should not be seen as an attribute. The attributes are the body mass and the heart rate.

Datasets can also be represented as points in a space known as the **attribute space**, where each axis corresponds to one attribute. {numref}`HrvsBM` is an example of an attribute-space representation of a dataset, in this case the dataset that consists of the body mass and heart rate of several animal species. Each animal species (item) is represented as a point in a 2D space, where the coordinates of the point are the body mass attribute and the heart rate attribute. We will sometimes use the terms **sample** and **point** as synonyms of the term item, and the term **feature** as synonym of the term attribute.


### What is knowledge?

Knowledge can be an evasive concept, yet according to the definitions that we have presented, machine learning *extracts knowledge* from datasets. Therefore, in machine learning we must be able to represent knowledge somehow. Here are different ways to **represent knowledge**:

1. **Propoposition**, i.e. as a statement that can be true or false. An example of a proposition is *smaller animals have faster heart rates*.
2. **Narrative**, description or story. For instance, *the size and heart rate of an animal are associated and in general we observe that larger animals tend to have a slower heart rate than smaller animals, for instance, the wild rabbit [...].*
3. **Model**, i.e. a quantitative relationship between attributes. For instance, using the dataset shown in {numref}`HrvsBM`, [Noujaim et al](https://www.ahajournals.org/doi/10.1161/01.CIR.0000146785.15995.67) suggested that the body mass $m$ in kg and heart rate $r$ in bpm of an animal species are related by the mathematical expression $r = 235 \times m^{-1/4}$.

In machine learning we use **models** to represent the knowledge that we extract from a dataset. Models can be expressed using mathematical notation (e.g. $r = 235 \times m^{-1/4}$) or equivalently can be implemented as a computer program. For instance, the relationship between heart rate and body mass proposed by [Noujaim et al](https://www.ahajournals.org/doi/10.1161/01.CIR.0000146785.15995.67) can be expressed in the Python programming language as

```python
r = 235 * m**(-1/4)
```

The question arises, why would we want to extract knowledge from a dataset?

% Note that we sometimes distinguish between **mathematical models** and **computer models**, however for us they are equivalent representations.


### The deployment-first view

The dataset-first perspective of machine learning states that machine learning is a *set of tools for extracting knowledge from datasets*. In other words, the starting point in machine learning is a **dataset** and its output is a **model**. We find that dataset-first views of machine learning can make it harder for us, specially beginners, to understand why we want to build a model, how to buid a model correctly or even whether it makes sense to use machine learning at all.

Before formally presenting the deployment-first perspective of machine learning, let us return to our toy problem of guessing the heart rate of an animal species given its body mass. What are the steps that we, as machine learning experts, would have taken in the context of this problem? First of all, the need to know the heart rate of an animal species without having to measure it would have prompted us to formulate our problem. Right afterwards, we would have acknowledged that we do not know the exact relationship between heart rate and body mass, nor we are  aware of any laws in biology that put together would allow us to derive such relationship. In the absence of previous knowledge, we would have measured the average body mass and heart rate of several animal species. In other words, we would have obtained a suitable dataset. Using the dataset we would have built a model, for instance the one according to which the heart rate $r$ of an animal is calculated as $r = 235 \times m^{-1/4}$, where $m$ is the body mass. Once the model is built, we would have discarded the dataset, as all our learning is done. Every time we ask ourselves about the heart rate $r$ of an animal species, we would simply plug the value of its body mass $m$ into the model $r = 235 \times m^{-1/4}$ and compute $r$. Putting our model to work is what we call **deployment**.

In summary, the steps we would have taken are:
1. Formulate a problem (*guess the heart rate of an animal species given its body mass*).
2. Secure a dataset (the samples shown in {numref}`HrvsBM`).
3. Build a solution ($r = 235 \times m^{-1/4}$).
4. Deploy the solution (plug a value for $m$ in the model to obtain $r$).

With this example in mind, we are in a good position to describe what we mean by the **deployment-first** perspective of machine learning. Using a deployment-first perspective, machine learning can be described as a *set of **tools** together with a **methodology** for solving scientific, engineering and business **problems** using **data***. Our starting point is a problem, not a dataset.

A deployment-first view of machine learning can help us understand why we want to use machine learning, when we should consider using (or discarding) machine learning and how machine learning works:

1. **Why**? We use machine learning to solve problems. The solutions are models that when deployed, deliver value.
2. **When**? We use datasets to build models because we do not know how the attributes relate to each other. If we knew it, we would not need machine learning. For instance, to build a model that predicts the distance driven given our speed and journey duration, we do not need machine learning.
3. **How**? Machine learning models are meant to be deployed. This means that the datasets that we secure need to be representative of the deployment scenario. Furthermore, we need to be able to evaluate or *test* the performance of our models before deployment. This test should be done in deployment conditions.

This brings us to the concept of **model lifecycle**. Machine learning models go through two basic stages ({numref}`MLLife`):
1. **Learning** stage: The model is built and tested. Data and domain knowledge about the problem are used in this stage.
2. **Deployment** stage: The model is used, for instance to make a prediction, decide on an action or gain insight.


To design a suitable testing strategy we need to understand how the model will be deployed, as we need to test a model in deployment conditions, i.e. *as if it had been deployed*. One of the limitations of dataset-first views is that the main focus is usually placed on the learning stage, whereas the deployment stage is, if discussed at all, presented as a secondary, applied aspect of machine learning. Without using the notion of deployment, it can be difficult to understand how to test a model correctly. By contrast using the deployment-first perspective, deploying a model is the ultimate goal of machine learning. Everything we do during the learning stage, including building and testing models, is designed with an eye on the future deployment of the model.


<!-- ```{figure} images/MLLifecycle.jpg
---
name: MLLife
---
The machine learning model lifecycle.
``` -->

```{figure} images/MLLifecycle.svg
---
name: MLLife
---
The machine learning model lifecycle.
```


(Intro3)=
## The machine learning taxonomy

We have defined machine learning as a set of tools together with a methodology for solving problems using data. In machine learning we first formulate a problem, then secure a dataset, subsequently we build and test a model, and finaly we deploy it. In this section we discuss the types of problems that can be formulated in machine learning.

Machine learning problems can be organised in a taxonomy ({numref}`MLTax`). Understanding this taxonomy is important, as there are different machine learning teachniques for each family of problems. Let us look at each family of machine learning problems.

<!-- ```{figure} images/MLTaxonomy.jpg
---
name: MLTax
---
The machine learning taxonomy.
``` -->

```{figure} images/MLTaxonomy.svg
---
name: MLTax
---
The machine learning taxonomy.
```




### Supervised learning

We have already discussed a supervised learning problem, namely that of obtaining the heart rate of an animal species whose body mass we know. Using machine learning lingo, the problem that we want to solve can be formulated as follows: given an **item** (e.g. a rabbit) such that the value of **one its attributes is unknown** to us (the heart rate), **estimate** (guess) the missing value using its **known attributes** (body mass).

The unknown attribute that we want to predict is called the **label**, whereas the known attributes are called **predictors**. In supervised learning we build models that estimate the label of an item based on its predictors. To build such models, we use datasets consisting of items of which we know predictors and label. A dataset that is used in a supervised learning problem is sometimes called **labelled dataset**, as the label attribute of all the items is known. The term **supervised** is metaphorical, and it suggests there is a supervisor showing our model examples consisting of sets of predictors together with a target label, so that the model learns the correct mapping of predictors to label.

There are two families of supervised learning problems, namely regression and classification. In a **regression** problem, the label that we want to predict is a **continuous** value. Examples of regression problems include the problem of predicting the heart rate of an animal species, the energy consumption of a household, the price of a company's stock and tomorrow's average temperature. By contrast, in a **classification** problem the label is a **discrete** value. Examples of classification problems include determining whether an email is spam or not, identifying the sentiment of a fragment of text or recognising a letter from an alphabet in a picture.


### Unsupervised learning

The term *unsupervised* might not be the most appropriate for the second family of machine learning problems. This term can be understood as learning without supervision or as learning that is not of the supervised type. Either way, it does not give away the essence of this second family of machine learning problems. Let us simply use the term *unsupervised* as a name for the second family of machine learning problems and avoid trying to justify why we have chosen it. It really does not matter.

We will use the heart rate vs body mass dataset to illustrate the essence of unsupervised learning. For convenience, it is shown again in {numref}`HrvsBMbis`. Have a close look at the point cloud representing the body mass and heart rate of several animal species. Can you see anything odd about it? You migth have spotted a point that lies in an unusual location. That point corresponds to an animal species that weights 250g and has a heart rate of 70 bpm. This animal species is the Syrian bear. Its body mass is however not 250 **g**, but 250 **kg**. It turns out that we made a mistake when we typed the body mass of the Syrian bear into our dataset and this mistake went unnoticed until one of our students pointed out that there seemed to be something unusual about that animal. We decided to leave this mistake, as it is one of the best illustrations of unsupervised learning.


```{figure} images/HRvsBM_earth.svg
---
name: HrvsBMbis
---
The heart rate vs body mass dataset (again). Can you spot a misbehaving animal?
```

If we were told that there is an animal species with a body mass of 250 g and a heart rate of 70 bpm, we would probably shrug our shoulders. However, in relation to the body mass and heart rate of other animals, as shown in {numref}`HrvsBMbis`, it feels really odd. Why? Because it does not seem to *belong* with the other animal species. This is what we call an **outlier**. At this point it is worth stopping and reflecting on our thought process. To determine what we mean by a normal relationship between heart rate and body mass, we have identified in our dataset the region of the attribute space where our samples are mostly distributed. Anything outside this region looks odd. Understanding how samples are distributed in the attribute space is the main goal in unsupervised learning.

**Unsupervised learning** builds models that describe how our samples are **distributed in the attribute space**. Note that the notions of predictors and label do not exist in unsupervised learning: all the attributes are treated equally. In unsupervised learning we can distinguish between two different approaches for describing how our samples are distributed. The first one is finding the underlying structure of our dataset. We will call this approach **structure analysis**. Grouping the samples of our dataset into clusters of similar samples is one popular approach to describe the underlying structure. This method is known as **clustering**. We can also identify directions in the attribute space along wich samples are aligned. This method is known as **basis discovery**.

The second family of unsupervised learning problems is **density estimation**. In density estimation we build models that we can use to obtain the probability of finding a future sample within a region of the attribute space. Equivalently we can obtain the fraction of future samples that will lie within that region of interest. For instance, looking at {numref}`HrvsBMbis`, what would you say is the probability of finding animal species whose body mass is between 10 kg and 100 kg and heart rate between 400 bpm and 500 bpm? Would you say it is a probability close to zero? Why? Questions like this one can be answered if we have a probability model. In density estimation, we use datasets to build probability models.


There are many applications of unsupervised learning. For instance, customer segmentation is an application of unsupervised learning where we create groups of customers that have similar preferences. Community detection in social networks is another application, where groups of connected individuals are identified. Evolutionary analysis allows us to investigate how animal species have evolved by analysing similarities in their DNA. Interestingly, we will sometimes need to solve an unsupervised learning problem to create models that will then be embeddeded within a supervised learning model.



(Intro4)=
## The scope of machine learning

The term *machine learning* can be both inspiring and misleading. If you are reading about machine learning for the first time, you will be wondering when we will start talking about the actual *machine that learns*. Or you might be asking yourself why we celebrated having solved a machine learning problem, back when we offered a guess for the rabbit's heart rate. After all it was *us* who did it, not a *machine*. As it turns out [some authors](https://www.statlearning.com/) prefer to use the term *statistical learning* instead of *machine learning*, to emphasise that this is in fact a branch of statistics, which is a discipline concerned with the analysis of data. In this section, we present machine learning in the context of data science. Then, we discuss its connection with statistics, computer science, digital processing, artificial intelligence and big data.


### Data and science

Whether we use a dataset-first view or a deployment-first view of machine learning, it should be clear that without datasets, there is no machine learning. Specifically, the deployment-first view emphasises that we use datasets to solve problems. The question arises, are there alternative ways for solving those problems that do not require data? If so, when should we use machine learning? And how can we use data correctly?

%Are there other approaches to solve problems using data?

To answer these questions, we need to briefly introduce the notion of **population** or **data source**. The goal of machine learning is to solve problems that are defined on an entity that we call the target population. For example when we are asking ourselves about the heart rate of an animal whose weight we know, we are considering as a target population the collection of all the animals on earth, past, present and future. We say that we have a perfect description of a population if we know accurately the relationship between the attributes that define the population. Using a perfect description of our population, we can answer any question about it. In most cases, however, we do not have such a perfect description. In these scenarios machine learning plays a crucial role. In machine learning we use datasets of items extracted from the population as *surrogates* for the perfect description of the population. In other words, we use datasets and machine learning *because* we lack a perfect description of our target population. If we had such description, there would be no need to use machine learning.


% Datasets are subsets of items that have been extracted from a population. For instance, every time we measure the body mass and heart rate of one animal, we are extracting one sample from this population, and the collection of all our measurements forms a dataset.

% In machine learning we use datasets precisely *because* we lack such description.




%Machine learning uses datasets as *surrogate* for a perfect description for our population.

%or precisely to discover such description.

%A population is the entity from which the items that form our dataset are extracted. Populations are described by the same attributes that their items have.


A limitation of the dataset-first view of machine learning is that it seems to suggest that as long as we have datasets, we are good to go. By contrast, in the deployment-first view of machine learning we acknowledge that datasets are used to solve problems, and for each problem we need to acquire the *right dataset*. Specifically, datasets need to be **representative** of our population. Putting our emphasis on the datasets might also lead us to think that datasets and only datasets is what we need to solve our problem. We frequently read that *data is objective* or that *data demonstrates* some, usually controversial, statement. This is however not the case. Data is fundamentally dumb and certainly not immune to subjectivity. We, after all, are the ones who decide how to create our datasets and hence our datasets will carry our own personal biases. Therefore, the quality of our solutions relies on our ability to secure the right data. As we will see throughout this book machine learning needs data, but is much more than data.

%In general, when we build machine learning models we will always incorporate **assumptions** based on what we already know (or think we know) about the target population. Machine learning needs data, but is much more than data.

% Ignoring this fact is one of the most obvious opportunities to fool ourselves.


% Data extraction is one of the processes where we will use some prior knowledge, as illustrated in ({numref}`MLLife`).

%[The Mismeasure of Man](https://en.wikipedia.org/wiki/The_Mismeasure_of_Man), by Stephen J. Gould, provides a fascinating account of historical episodes where data has been used to support biological worth, i.e. the idea that some groups of individuals are superior to others. Interestingly those who have sought using data to support theories of biological worth always claim that the data they analysed was objective and their conclusions were uncontaminated. Do you find this suspicious? This statement amounts to confessing a crime.

% At the end of the day

Machine learning can also be seen as a scientific endeavour. The essence of science lies in our ability to evaluate our knowledge. In machine learning, our ability to check that our models actually work is as important as being able to build them. In fact, there might be situations where we need to deploy a model created by others. In such situations we do not really care about how sophisticated the machine learning model is, or even whether the model has been built using machine learning at all, we just want to know if the model will actually work during deployment. In other words, we need to be able to **test our models**. In an [experience](https://www.youtube.com/watch?v=cqoYrSd94kA) designed to expose dowsing, i.e. the idea that some humans have a supernatural sensitivity that allows them to detect underground water, the magician and professional skeptic James Randi summarised their goal using the following words: *my concern is not **how** they do it, but **if** they do it*. Many machine learning solutions are said to be too complex for those who have not designed them to understand. This should not deter us, as we can still use machine learning to test them and check if they actually do what they are intended to do or not.

Finally, as we will see in this book machine learning operates by exploiting associations between the attributes of the samples in a dataset. It is sometimes tempting to interpret such associations using a causal lens. For instance, our ability to predict the heart rate of an animal species using its body mass could lead us to think that *the body mass of an animal causes the heart rate*. This is an illusion that can be uncovered as soon as we ask ourselves if we can predict the body mass of an animal species from its heart rate. We can, indeed, but would we now say that *the heart rate of an animal also causes its body mass*? The only logical conclusion is that the associations between attributes that machine learning creates, should never be interpreted as causal relationships. Causality is simply out of machine learning's reach. To investigate causality, [other approaches](http://bayes.cs.ucla.edu/WHY/) exist that also use data.




### Related fields

%In fact, using the language of probability we can describe supervised learning as a problem where we build a conditional probability, namely the probability of a label given a set of predictors, and unsupervised learning as a problem where we build joint probabilities.

Machine learning is, first and foremost, a branch of **statistics**. This is why many researchers prefer using the term *statistical learning*, which they feel describes more accurately what this discipline is about. Why do we then talk about *machine* learning? The term machine refers to computers and in machine learning computers play a crucial role, for even though computers are not essential, models are nothing but computations and models are built performing computations on datasets. From a practical point of view, even the simplest machine learning problem can benefit from using a modest amount of computational power and in most cases we will not be able to realistically build and deploy models if we lack computational power. This is the reason why machine learning can also be seen as a branch of **computer science**.

**Digital signal and image processing** is a discipline that deals with temporal and spatial data, such as audio recordings or photographs, and can play a central role in many machine learning projects. First, when dealing with temporal or spacial data, we can create datasets consisting of attributes that are obtained by digitally processing such data. For instance, in an audio scenario, we can crete a dataset where one of the attributes is the pitch of a sound. Therefore, digital processing can be used as a preprocessing stage prior to machine learning modeling. Second, digital signal and image processing define operations on data. These operations can be hand crafted, but also learnt using machine learning approaches.

%Machine learning can be extended to create complex models, which include digital processing stages that are also learnt using data.

Some of you might be surprised that we have not mentioned **artificial intelligence** yet. Machine learning is frequently introduced as a subset of artificial intelligence and, after all, the name seems to suggest that we are building *machines that learn*. The ultimate goal of artificial intelligence is to create machines that act or think like humans, and indeed, machine learning models can be incorporated *into* such intelligent machines. However artificial intelligence is one of the many application areas of machine learning, by no means the only one, and artificial intelligence can use approaches other than machine learning. The relationship between machine learning and artificial intelligence is the same as the relationship between an engine and a car. A subset of cars would be all the cars of a given colour, or of a given brand. Engines are not subsets of cars, but rather a component of a car. There are different types of engines that we can use in the car, and engines can be used in other systems that are not cars, for instance to extract water from a well. In the same vein, we do not see machine learning as a subset of artificial intelligence, but as a component that can be used in an artificial intelligence system.

%We prefer to use the term *machine learning* as the name for a discipline and not as its definition, in as much the same way as we do not think about *states* when we define *statistics*.

Finally, **big data technologies** can play a significant role in machine learning projects. Contrary to what it seems to connote, big data is not a term that refers to *the existence of very large datasets*, neither it should be used as implying that *the more data we have, the better models we will create*. Big data is a field in computer science concerned with creating data engineering systems that operate seamlessly where conventional computer systems, e.g. our laptop or PC, fall short due to insufficient computing resources. Examples of this include video streaming platforms, serving hundreds of million of hours of video every day. Big data is the collectiong of technologies that make possible this type of platforms. In machine learning, big data technologies can play a crucial role in those cases where the computing resources that we need for, say, building a model or deploying a model, exceed what our modest laptops can do. In other words, big data can be used to solve computational obstacles, but does not suggest we should use large datasets to solve machine learning problems.





(Intro5)=
## Structure of this book

This book is organised around three main topics: supervised learning, unsupervised learning and the machine learning methodology. The first part of the book is devoted to supervised learning and the second to unsupervised learning. You will find three methodology chapters intercalated between supervised and unsupervised chapters. Each chapter is followed by a Python Jupyter notebook, which you can use to experiment and consolidate your understanding. There is also an appendix at the end of the book, covering background topics such as linear algebra, basic probability concepts and how to set up a data science computing environment.

In the first part of the book, we will focus on supervised learning. We will learn to formulate regression and classification problems and will study some of the most popular supervised learning models. Our focus is not on exhaustively covering as many models as possible. We will focus on the principles so that in the future you can independently and confidently learn and successfully apply new machine learning models. In the second part of the book, we will focus on unsupervised learning. First, we will cover structure analysis, specifically clustering and basis discovery. Then, we will discuss density estimation problems and will present several applications, including building class densities for classification problems and outlier detection.


The machine learning methodology is a horizontal topic, relevant to both supervised and unsupervised problems. As our definition of machine learning implies, the machine learning methodology is a first-class citizen and therefore we will discuss it in separate chapters. The first methodology chapter follows the chapter on regression and will present the machine learning tasks of model testing, model training and model validation.  The second methodology chapter will introduce the notion of machine learning pipeline, which will allow us to extend the notion of model discussed up until then, to include pre-processing stages and ensemble approaches. Understanding pipelines will pave the ground to understanding more complex models, including deep neural networks. In the third methodology chapter we will look at machine learning from a professional perspective. We will discuss how end-to-end machine learning projects are managed using the notion of machine learning workflows and how to work professionally and within a solid ethical framework.




(Intro6)=
## Summary and first top tip

This chapter provided an introduction to machine learning. We have seen that machine learning lies in the intersection between statistics and computer science and can be found in many fields, from science and high-tech to applied areas such as retail and finance. In a nutsheel, machine learning provides a set of tools together with a methodology for solving scientific, engineering and business problems using data. These problems involve a target population, whose description is unknown to us and the machine learning approach consists of using datasets as surrogates for the perfect description of the population.

We have also seen that the type of problems that machine learning can solve can be arranged into a taxonomy and belong to one of two main types: supervised learning, where we build models that predict the value of one of the attributes of a sample, and unsupervised learning, where we set out to describe relationships between the attributes of our samples. In a typical machine learning workflow we start by formulating a problem that involves a population. Then, we secure a dataset. Third, we build and test a model and finally, we deploy it.

We conclude this chapter looking again at {numref}`HrvsBM`. By now, we should be confident that we can provide an educated guess for the heart rate of any animal species, provided that we know their body mass.
<!-- Let us consider the flatworm. The flatworm is a very small animal with a body mass of 10 g. Our dataset includes animal species that have a body mass between 22 g (wild mouse) to 30 t (humpback whale). Do you think we can produce an estimate for the heart rate of the flatworm?  -->

```{admonition} Question for you
:class: question1

Let us consider the flatworm. The flatworm is a very small animal with a body mass of 10 g. Our dataset includes animal species that have a body mass between 22 g (wild mouse) to 30 t (humpback whale).

Using {numref}`HrvsBM`, do you think we can produce an estimate for the heart rate of the flatworm?

Submit your response here: <a href="https://forms.office.com/e/yKxE4LUpZF" target = "_blank">Your Response</a>

```

Some of you might be concerned that the body mass of the flatworm lies outside the observed body mass range and hence might conclude that we cannot say anything about it. However, {numref}`HrvsBM` shows a clear upward trend as the body mass decreases, and there is no reason we could not extend it beyond the observed range. Doing so, we would be able to say that the heart rate of a flatworm is greater than 500 bpm. In fact, we could simply plug the value $m = 0.01$ kg in $r = 235 \times m^{-1/4}$, and we would obtain an estimated value of around 700bpm. Was this a good guess?

Machine learning models give us answers even in scenarios where our intuition hesitates. There is however, one catch in our previous guess for the heart rate of the flatworm. It turns out that flatworms do not have a heart like the one we have, and because of it they simply have no heart beat. Therefore, asking about the heart rate of a flatworm does not make any sense whatsoever. Machine learning is unaware of this: we could have given our machine learning model the body mass of a flower or a brick, and we would have got an answer. Machine learning abstracts away domain details, but that does not mean that domain details are irrelevant.

<!-- Our first top tip is

**Know Your Domain**!

```{tip}
**Know Your Domain**!
``` -->

```{admonition} Our first top tip is
:class: tip
<!-- <p style="text-align: center;"><b>Know Your Domain!<b></p> -->
<h3 style="text-align: center;"><b>Know Thy Domain!</b></h3>
```

If you do not, you will risk ending up with an astonishing solution for a meaningless problem.
