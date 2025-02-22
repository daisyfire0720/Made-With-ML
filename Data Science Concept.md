# Fundamental Concepts in Machine Learning!
### Machine Learning: Main Ideas
1. Classify things - Classification
2. Make quantitative predictions - Regression

### Terminology Alert
1. Discrete Data - **countable** and only takes specific values
    * For example, we count the number of people that love the color green or love the color blue
    * Because we are counting individual people, and the totals can only be whole numbers, the data are **Discrete**
    * American shoe sizes, Rankings and other orderings are also Discrete
2. Continuous Data - **measurable** and can take any numeric value within a range
    * For example, Height measurements are **Continuous** data
    * Height measurements can be any number between 0 and the height of the tallest person on the planet
    * If with a more precise ruler, then the measurement get more precise
    * So, the precision of Continuous measurements is only limited by the tools we use
    
# Cross Validation
Rather than worry too much about which specific points are best for **Training** and best for **Testing**, **Cross Validation** uses *all* points for both in an *iterative* way, meaning that we use them in steps

### Terminology Alert
* **Date Leakage** - reusing the same data for training and testing, usually results in you believing the machine learning method will perform better than it does because it is **Overfit**

### Leave-One-Out
1. **Leave-One-Out** uses all but one point for Training
2. and uses the one remaining point for Testing
3. and then iterates until every single point has been used for Testing
* Some experts say that when the dataset is large, use **10-Fold Cross Validation**, and when the dataset is very small, use **Leave-One-Out**

# Fundamental Concepts in Statistics
### Histograms
1. **Histograms** are one of the most basic, but surprisingly useful, statistical tools we can use to gain insights into data - easy to see trends in the data
    * we divided the range of values into bins and stack the measurements that fail in the same bin
    * Not too wide or too narrow, need to try a bunch of different bin widths to get a clear picture

## Probability Distributions
Because we have **Discrete** and **Continuous** data there are:
1. Discrete Probability Distributions 
2. Continuous Probability Distributions 

### Discrete Probability Distributions
1. Use *mathematical equations* to calculate probabilities, the **Binomial Distribution**
    * **Binomial Distribution** gives us probabilities for sequences of **binary outcomes** (wins and losses, yeses and noes, etc.), like 2 out of 3 people preferring pumpkin pie
2. **Poission Distribution**
    * **events** that happen in discrete units of time or space, like reading 10 pages an hour, we can use **Possion Distribution**

### Continuous Probability Distributions
Similarly, we can use *mathematical formulas* just like we did with Discrete Distributions, we can use a **Normal Distribution**, which creates a *bell-shaped curve* instead of a histogram (bins too wide or too narrow)

1. a **Normal** distribution is symmetrical about the mean, or average, value.
2. The **Normal Distribution**'s maximum likelihood value occurs at its mean
3. **Likelihood** - y-axis (represents most likely or less likely)
4. Lots of things can be approximated with **Normal Distribution**: height, birth weight, blood pressure, job satisfaction, and many more!
5. The width of a Normal Distribution is defined by the **Standard Deviation**
    * **95%** = +/- 2 **Standard Deviations** around the **Mean**
    * mean adult measruement is *177 cm* for height, and 2 x the standard deviation = 2 * 10.2 = 20.4
    * about **95%** of the adult measurements fall between *156.6* and *197.4 cm*
6. Conclude - to draw a Normal Distribution you need to know
    * The **Mean** or average measurement - this tells you where the center of the curve goes
    * The **Standard Deviation** of the measurements - this tells you how tall and skinny, or short and fat, the curve should be
7. 68.27-95.45-99.73% Rule

### Standard Deviation vs Standard Error
* Standard Deviation quantifies the variation within a set of measurements
    * It tells you how much individual data points vary from the mean
* Standard Error quantifies the variation in the means from multiple sets of measurements
    * Can be estimated from a single set of measurements, even though it describes the means from multiple sets
    * It tells you how accurate your sample mean is as an estimate of the true population mean

#### Calculating Probabilities with Continuous Probability Distributions
1. For Continuous Probability Distributions, **probabilities** are the **area under the curve** between two points
    * For example, given this Normal Distribution with mean = 155.7 and standard deviation = 6.6, the probability of getting a measurement between 142.5 and 155.7 is equal to the area under the curve, which is 0.48
2. Regardless of how tall and skinny or short and fat a distribution is, the total area under its curve is **1** - meaning the probability of measuring anything in the range of possible values is **1**
3. Probability for a specific measurement is 0, because probabilities are areas, and the area of something with no width is **0**
    * Another way is to realize that a continuous distribution has infinite precision, thus, we are really asking the probability of measuring someone who is exactly 155.70000000000000... tail
    * **Likelihoods** are the y-axis coordinates for a specific point on the curve - uses to fit curves
        * L (distribution | data)
        * Example: L (mean = 32 and standard deviation = 2.5 given weighs 34 grams), assume weighed more than one
    * whereas **Probabilities** are the area under the curve between two points
        * pr( data | distribution)
     
### Maximum Likelihood for the Normal Distribution
* Equation has two paramters
    * μ (mu): determines the location of the normal distribution's **mean**
        * A small value for μ (mu) moves the mean of the distribution to the left
        * A larger value for μ (mu) moves the mean of the distribution to the right
    * σ (sigma): is the **standard deviation** and determines the normal distribution's width
        * A smaller value for σ (sigma) makes the nromal curve taller and narrower
        * A larger value for σ (sigma) makes the normal curve shorter and wider
* If we had more data...
    * Then we could plot the likelihoods for different values of σ (sigma)
    * And the maximum likelihood estimate for σ (sigma) would be at the peak, where the slope of the curve = **0**
* To calculate the likelihood of a normal distribution when we have more than one measurement (we just multiply together the individual likelihoods)
* To find the maximum likelihood estimate for μ (mu), we need to solve for where the deviate with respect to **μ (mu) = 0**, because the slope is **0** at the peak of the curve
* Likewise, to find the maximum likelihood estimate for σ (sigma), we need to solve for where the derivative with respect to **σ (sigma) = 0**, because the slope is **0** at the peak of the curve
* Summary
    * The mean of the data is the maximum likelihood estimate for where the center of the normal distribution should god
    * The standard deviation of the data is the maximum likelihood estimate of how wide the normal curve should be
* Summary
    * The mean of the data is the maximum likelihood estiamte for where the center of the normal distribution should god
    * The standard deviation of the data is the maximum likelihood estimate of how wide the normal curve should be

#### Other Continuous Distributions
1. **Exponential Distributions** - commonly used when we are interested in how much time passed between events.
    * For example, we could measure how many minutes pass between page turns in this book.
2. **Uniform Distributions** - commonly used to generate random numbers that are equally likely to occur
    * For example, if I want to select random numbers between 0 and 1, then I would use a Uniform Distribution, that goes from 0 to 1, which is called a Uniform 0,1 Distribution because it ensures that every value between 0 and 1 is equally likely to occur
    * In contrast, we want to generate random numbers between 0 and 5, then we use a Uniform Distribution that goes from 0 to 5, which is called a Uniform 0,5 Distribution
    * We can even span any 2 numbers, so we could have a Uniform 1, 3, 5 Distribution if we wanted one

#### Summary
Just like **Discrete Distributions**, **Continuous Distributions** 
1. spare us from having to gather tons of data for a histogram, 
2. and additionally, Continuous Distributions also spare us from having to decide how to bin the data
* Continuous Distribution use equations that represent smooth curves and can provide likelihoods and probabilities for all possible measurements
* Like **Discrete Distributions**, there are **Continuous Distritbuions** for all kinds of data, like values we get from measuring people's height or timing how long it takes you to read a book page

In the context of machine learning, both types of distributions allow us to create **Models** that can predict what will happen next

## The Center Limit Theorem
**Means are normally distributed** disregard uniform distribution or exponential distribution
* Doesn't matter what distribution you start with, if you collect samples from those distributions, then the means will be normally distributed
* When we do an experiment, we don't always know what distribution our data comes from
    * To this, **The Central Limit Theorem** says, **"Who Cares???"**
* The sample means will be normally distributed and because we know that the sample means are normally distributed
    * We don't need to worry too much about the distribution that the samples came from
    * We can use the mean's normal distribution to make **confidence intervals**
    * Do **t-tests**, where we ask if there is a difference between the means from two samples
    * And **ANOVA**, where we ask if there is a difference among the means from three or more samples
    * And pretty much any statistical test that uses the sample mean
* **Note:** Rule of thumb, the sample size must be at least **30**, but it is just a rule of thumb

## Effective Sample Size (ESS)
1. Effective Sample Size = The number of samples / 1 + (the number of samples - 1) * the correlation
* High correlation 0.7 for two people, 2 / 1 + (2 - 1) * 0.7 = 2 / 1 + (1 * 0.7) = 2 / 1 + 0.7 = 1.18
    * When the correlation is high between two people, instead of being counted as 2, they are counted as 1.18 people
* Low correlation 0.1 for two people, 2 / 1 + (2 - 1) * 0.1 = 2 / 1 + (1 * 0.1) = 2 / 1 + 0.1 = 1.82
    * Almost count as much as two people
* Samples that are highly correlated don't count as individual samples
Summary
* If ESS ≈ N, the data is mostly independent
* If ESS < N, the data is highly correlated, and fewer independent samples exist 

## Models
1. Models approximate reality to let us explore relationships and make predictions
2. In machine learning, we build models by training machine learning algorithms with **Training Data**
3. Statistics can be used to determine if a model is useful or believable
    * how statistics can quantify the quality of a model - **Sum of the Squared Residuals**

### The Sum of the Squared Residuals
Quantify the quality of a model and its predictions - to calculate the **Sum of the Squared Residuals**
1. calculating **Residuals**, the differences between the **Observed** values and the values **Predicted** by the model
    * Residual = Observed - Predicted
2. the smaller the **Residuals**, the better the model fits the data, but in a line top data point (Observed - Predicted +12 difference) and bottom data point (Observed - Predicted -12 difference), the value of residuals will cancel out (0 difference residuals = top +12 bottom -12)
    * instead of calculating the sum of **Residuals**, we square the **Residuals** first and calculate the **Sum of the Squared Residuals (SSR)**
    * **Note**: **Squaring**, as opposed to taking the **absolute value**, makes it easy to take the derivative
3. We can calculate **SSR** for all kinds of models, not only a simple straight line, a curve line, a sinusoidal model

### Mean Squared Error (MSE)
* **SSR** is not super easy to interpret because it depends, in part, on how much data you have
* the increase of SSR *does not* suggest that the model is worse, it only tells us that one model with more data has more **Residuals**

One way to compare the two models that may be fit to different-sized datasets is to calculate the **Mean Squared Error(MSE)**, which is simply the average of the **SSR**
* **Mean Squared Error(MSE)** = **The Sum of Squared Residuals(SSR)** / **Number of Observations (n)**
* **MSEs**  are still difficult to interpret on their own because the maximum values depend on the scale of the data
    * if y-axis is in *millimeters* and the **Residual**s are 1, -3, and 2, then **MSE** = 4.7
    * however, if we change the y-axis to *meters*, then the **Residuals** for the exact same data shrink to 0.001, -0.003, and 0.002, and the **MSE** is now 0.0000047, it is tiny!
    * use **SSR** and **MSE** to calculate **R^2**, which independent of both the size of the dataset and the scale

### Confidence Intervals
A 95% confidence interval is just an interval that covers 95% of the means
* Because the interval covers 95% of the means, we know that anything outside of it occurs less than 5% of the time
* That is to say, p-value of anything outside of the confidence interval is < 0.05 (and thus, significantly different)

### R^2 (R squared)
**R^2** is calculated by comparing the **SSR** or **MSE** around the **mean** y-xis value
* For example, calculate the **SSR** or **MSE** around the **average** Height and compare it to the **SSR** or **MSE** around the model we are interested in
    * That means, we calculate the **SSR** or **MSE** around the model (line) that uses Weight to predict Height
    * **R^2** then gives us a percentage of how much the predictions improved by using the model we are interested in instead of just the **mean**
    * In this example, **R^2** would tell us how much better our predictions are when we use the model, the line (Weight to predict Height), instead of predicting that everyone has the **average** Height
* **R^2** values go from 0 to 1 and are interpreted as percentages, and the closer the value is to 1, **the better the model fits the data relative to the mean y-axis value** 
* **R^2** = (**SSR(mean)** - **SSR(fitted line)**) / **SSR(mean)** 
    * 1.6 - 0.5 / 1.6 = 0.7 -> tells us that there was a **70%** reduction in the size of the **Residuals** between the **mean** and the **fitted line**
* **R^2** values tell us the percentage of the **Residuals** around the mean **shrank** when we used the fitted line
    * When **SSR(mean)** = **SSR(fitted line)** -> both models' predictions are equally good (or equally bad), **R^2** = 0
        * SSR(mean) - SSR(fitted line) / SSR(mean) = 0 / SSR(mean) = 0 
    * When **SSR(fitted line)** = 0, meaning that the fitted line fits the data perfectly, then **R^2** = 1
        * SSR(mean) - 0 / SSR(mean) = SSR(mean) / SSR(mean) = 1
    * **Note**: any 2 random data points have **R^2 = 1**
        * because regardless of the Residuals around the mean, the Residuals around a fitted line will be 0
    * because a small amount of random data can have high (close to 1) R^2, any time we see a trend in a small dataset, it is difficult to have confidence that a high R^2 value is not due to random chance
    * if we had a lot of data organized randomly using a random *ID Number*, we would expect to have a relatively small (close to 0) R^2 because the Residuals would be similar
    * In contrast, when we see a trend in a large amount of data, we can, intuitively (in statistics, use **p-values** to quantify how much confidence we should have in **R^2** values and pretty much any other method that summarizes data, have more confidence that a large R^2 is not due to random chance

#### R^2: FAQ
1. **Does R^2 always compare to the mean to a straight fitted line?**
* The most common way to calculate **R^2** is to compare the **mean** to a **fitted line**
* However, we can calculate it for anything we can calculate the **Sum of the Squared Residuals** for
    * For example, for rainfall data, we use **R^2** to compare a **square wave** to a **sine wave**
    * In this case, we calculate **R^2** based on the  **Sum of the Squared Residuals** around the **square** and **sine**waves
    * **R^2** = **SSR(square)** - **SSR(sine)** / **SSR (sqaure)**
2. **Can R^2 be negative?**
* When we are only comparing the **mean** to a **fitted line**, **R^2** is positive, but when we compare other types of models, anything can happen
    * For example, if we use **R^2** to compare a **straight line** to a **parabola**
    * RSS(straight line) = 5, RSS(parabola) = 12
    * R^2 = 5(SSR(line)) - 12(SSR(parabola) / 5(SSR(line) = -1.4
    * We get a negative R^2 value, **-1.4**, and it tells us the Residuals increased by **140%**

### p-values
**p-values** （calculated using **Fisher's Exact Test**) give us a measure of confidence in the results from a statistical analysis
* **p-values** are numbers between 0 and 1, the closer a **p-value** is to 0, the more confidence
* the threshold - **0.05**
* For example, it means that if there's no difference between Drug A and Drug B, and if we did this exact same experiment a bunch of times, then only **5%** of those experiments would result in the wrong decisions
     * 0.05 threshold for **p-values** means that **5%** of the experiments, where the only differences come from weird, random things, will generate a **p-value** smaller than **0.05**
     * In other words, if there's no difference between Drug A and Drug B, in 5% of the times we do the experiment, we will get a **p-value** less than **0.05**, and that would be a **False Positive**
* **False Positive** - getting a small **p-value** when there is no difference
* Using a threshold of **0.001** would get a **False Positive** only once in every **1,000** experiments (Drug A vs Drug B)
* Using a threshold of **0.2** means we are willing to get a **False Positive** 2 times out of 10 (ice-cream truck arrives on time)
* That said, the most common threshold is **0.05** because trying to reduce the number of **False Positives** below **5%** often costs more than it's worth
* p-value regardless of the **size of the difference (data pool size)** between Drug A and Drug B

**Hypothesis Testing** - determine if these drugs are the same or not
* the **Null Hypothesis** is that the drugs are the same
* and the **p-value** helps us decide if we should *reject* the **Null Hypothesis**
    * Back to Drug A vs Drug B, if **p-value < 0.05**, Drug A is different from Drug B
    * if **p-value = 0.24**, we are not confident that Drug A is different from Drug B

## Linear Regression
**Linear Regression** fits a **line** to the data that *minimize* the **Sum of the Squared Residuals (SSR)**
* once we fit the line to the data, we can easily calculate **R^2** (how accurate our predictions will be)
* and also provides us with a **p-value** (how confident we should be that the predictions made with the **fitted line** are better than predictions made with the **mean** of y-axis coordinates for the data) for the **R^2** value
* Linear Regression selects the **line**, the y-axis intercept and slope, that results in the minimum **SSR** 
* One way to find the lowest point of the (SSR) **curve** is to calculate the **derivates** of the **curve**, and solve for where the **derivative** is equal to 0, at the bottom of the **curve**
    * Solving this equation results in an **Analytic Solution**, meaning, we end up with a formula that we can plug our data into, and the output is the optimal value
    * Analytics solutions are awesome but they are rare and only work in very specific situations
* Another way to find a optimal slope and y-axis intercept is to use an **Iterative Method**, called **Gradient Descent**
    * an **Iterative Method** starts with a guess for the value and then goes into a loop that improves the guess one small step at a time
    * Although **Gradient Descent** takes longer than an analytics solution, it's one of the most important tools in machine learning because it can be used in a wide variety of situations where there are no analytics solutions, including **Logistic Regression**, **Neural Networks**, and many more

#### p-values for Linear Regression and R^2
**p-value** tells us the probability that random data could result in a similar **R^2** value or a better one
* In other words, the **p=value** will tell us the probability that random data could result in an **R^2 >= 0.66**
* **Multiple Linear Regression** - Use **2** or more variables

**Linear Models** allow us to use **discrete** data, like whether or not someone loves the movie Troll 2, to predict something **continuous**, like how many grams of Popcorn they eat each day
* in this case, **p-value** of **0.04** is relatively small, which suggests that it would be unlikely for random data to give us the same result or something more extreme. 
* In other words, we can be confident that knowing whether or not someone loves Troll 2 will improve our prediction of how much Popcorn they will eat

**Linear Models** also easily combine **discrete data**, like whether or not someone loves Troll 2, with **continuous** data, like how much Soda Pop they drink, to predict something **continuous**, like how much Popcorn they will eat
* In this case, adding how much Soda Pop someone drinks to the model dramatically increased the **R^2** value, which means the predictions will be **more accurate**, and reduced the **p-value**, suggesting we can have **more confidence** in the predictions

# Gradient Descent
Use **Gradient Descent** when there is no analytical solution 
* **Gradient Descent** is an **iterative solution** that incrementally steps toward an optimal solution and is used in a very wide variety of situations
1. **Gradient Descent** starts with an initial guess
2. then improves the guess, one step at a time
3. until it finds an optimal solution or reaches a maximum number of steps
* **Gradient Descent**, minimize the **Loss** or **Cost Function** (minimize the **SSR**) by taking steps away from the initial guess toward the optimal value
    * In this case, we *increase* the **intercept**, the x-axis of the central graph, we *decrease* the **SSR**, the y-axis 
    * Instead of randomly trying a bunch of values, we plot the **SSR** as a function of the y-axis **intercept** (using **SSR** as a step)
* Use derivative of the curve, which tells us the slope of any **tangent line** that touches it
    * a relatively large value for the derivative, which corresponds to a relatively steep slope for the **tangent line**, suggests we are relatively far from the bottom of the curve, so we should take a relatively large step
        * and a *negative* derivative, or slope, tells us that we need to take a step to the right to get closer to the lowest **SSR**
    * a relatively small value for the derivative suggests we are relatively close to the bottom of the curve, so we should take a relatively small step
        * and a *positive* derivative tells us that we need to take a step to the left to get closer to the lowest **SSR**

**Loss Function** and **Cost Functions** refer to anything we want to optimize when we fit a model to data
* For example, we want to optimize the **SSR** or the **Mean Squared Error(MSE)** when we fit a straight line with **Regression** or a squiggly line (in **Neural Networks**)
* That said, some people use the term **Loss Function** to specifically refer to a function (like the **SSR**) applied to *only one data point*,
* and use the term **Cost Function** to specifically refer to a function (like the **SSR**) applied to *all* of the data

**Parameters** - things we want to optimize

**Learning Rate** - prevents us from taking steps that are too big and skipping past the lowest point in the (SSR) curve
* Typically, for **Gradient Descent**, the **Learning Rate** is determined automatically
    * it starts relatively large and gets smaller with every step taken
    * However, you can also use **Cross Validation** to determine a good value for the **Learning Rate** 

**Step Size** = **Derivative** * **Learning Rate**
* Keep updating the **intercept** after each iteration until the **Step Size** is close to **0**, or we take the maximum number of steps, which is often set to **1,000** iterations
* **New intercept** = **Current intercept** - **Step Size** - to get closer to the optimal value with this equation

#### Stochastic Gradient Descent
1. What if we had 1,000,000 data points, which compute 1,000,000 terms per derivative
2. What if we had a complicated model with 10,000 parameters, then we have 10,000 derivatives to compute
3. Taking 10,000 derivatives, each with 1,000,000 terms to compute, is a lot of work, and all of that work only gets us one step into a process that can take 1,000s of steps

**Stochastic Gradient Descent** - can drastically reduce the amount of computation required to optimize parameters
* **Stochastic Gradient Descent** does is randomly select one data point per step. So, regardless of how large your dataset is, only one term is computed per derivative for each iteration
* It is more common to randomly select a small subset of the observations, called **Mini-Batch Stochastic Gradient Descent**
    * Using a small subset rather than a single point, usually converges on the optimal values in fewer steps and takes much less time than using all of the data

#### Gradient Descent: FAQ
Will Gradient Descent always find the best parameter values?
* Unfortunately, **Gradient Descent** does not always find the best parameters values
* it may possibly get stuck at the bottom of **local minimum** SSR, instead of finding the bottom **global minimum**

We can try:
1. Try again using different random numbers to initialize the parameters that we want to optimize. Starting with different values may avoid a local minimum
2. Fiddle around with the **Step Size**. Making it a little larger may help avoid getting stuck in a local minimum
3. Use **Stochastic Gradient Descent**, because the extra randomness helps avoid getting trapped in a local minimum

How do you choose the size of a Mini-Batch for Stochastic Gradient Descent
* really depends on the computer hardware (how much high-speed memory we have access to), to more high-speed memory we have, the large the **Mini-Batch** can be

# Logistic Regression
**Logistic Regression**, which probably should have been named Logistic Classification, fits a **squiggle** to data that tells us the predicted probablities (between 0 and 1) for **discrete** variables, like whether or not someone loves Troll 2
* Also has have the metrics like **Linear Regression**, similar to **R^2** and **p-values**
* we can mix and match **discrete** and **continuous** features to make **discrete** classifications
1. **Logistic Regression** represents probability and goes from 0 to 1
2. Usually the threshold for classification is 0.5, Loves Troll 2 or Does Not Love Troll 2
    * loving Troll 2 **> 0.5** will be classified as someone who Loves Troll 2
    * and anyone with a probability **<= 0.5** will be classified as someone who Does Not Love Troll 2

**Logistic Regression** swaps out the **Residuals** for **Likelihoods** (y-axis coordinates) and fits a **squiggle** that represents the **Maximum Likelihood**
* However, because we have two classes of people (Love Troll 2 vs. Does Not Love Troll2), there are two ways to calculate **Likelihoods**, one for each class
    * To calculate the **Likelihood** for a person who Loves Troll 2 -> use the **squiggle** to find the y-axis (the y-axis is both probability and likelihood) coordinate that corresponds to the amount of Popcorn they ate
    * To calculate the **Likelihood** for a person who Does Not Love Troll 2 -> 1 minus the probability that a person Love Troll 2
    * p(Does Not Love Troll 2) = 1 - p(Love Troll 2) =  L(Does Not Love Troll 2) = 1 - L(Love Troll 2) 
* **Linear Regression** fits a **line** to the data by minimizing the **Sum of the Squared Residuals (SSR)**
* We can calculate the **Likelihood** for the entire **squiggle** by multiplying the individual **Likelihoods** together - 0.02
    * We can calculate the **Likelihood** for a different **squiggle** - 0.004 
    * 0.02 vs 0.004 - the goal is to find the **squiggle** with the **Maximum Likelihood** - we can usually find the optimal **squiggle** using **Gradient Descent**

#### Fitting A Squiggle
* When the **Training Dataset** was much larger, we might run into a computational problem, called **Underflow**, that happens when you try to multiply a lot of small numbers between 0 and 1
    * **Underflow** happens when a mathematical operation, like multiplication, results in a number that's smaller than the computer is capable of strong - result in errors or result in weird, unpredictable results, which are worse
* A very common way to avoid **Underflow** errors is to just take the **log** (usually the **natural log**, or **log base e**), which turns the multiplication into addition
    * Turns a number that was relatively close to 0, like 0.02 into a number relatively far from 0, -4.0

#### Logistic Regression: Weaknesses
* When we use **Logistic Regression**, we assume an **s-shaped squiggle** or an s-shaped surface (more than one independent variable)
* Use **Decision Tree** or **Neural Network** if we cannot fit an **s-shaped squiggle** to the data = not a valid assumption

# Assessing Model Performance
1. Use **Confusion Matrices**, simple grids that tell us what each model did right and what each one did wrong
2. **Receiver Operator Curves (ROCs)**, which give us an easy way to evaluate how each model performs with different classification thresholds

## Confusion Matrix
Keep track of four things
1. True Positives (both Yes from actual and predicted)
2. False Negatives (actual value is Yes, but predicted value is NO) - **Type II error**
3. False Positives (actual value is NO, but predicted value is Yes) - **Type I error**
4. True Negatives (both NO from actual and predicted)
* In general, the size of the matrix corresponds to the number of classifications we want to predict (2 classes, 3 classes)
* No standard for how a **Confusion Matrix** is oriented. In many cases,
    * row reflect the actual, or known, classifications
    * columns represent the predictions

### Sensitivity and Specificity
**Sensitivity** - we want to quantify how well an algorithm (like **Naive Bayes**) correctly classifies the *actual* **Positives**, in this case, the known people *with* Heart Disease, we calculated **Sensitivity**, which is the percentage of the actual **Positives** that were *correctly* classified
* **Sensitivity** = **True Positives / True Positives + False Negatives**
    * 142 / 142 + 29 = 0.83 -> which means that **83%** of the people *with* Heart Disease were correctly classified
    
**Specificty** - we want to quantify how well an algorithm (like **Logistic Regression**) correctly classifies the *actual* **Negativies**, in this case, the known people *without* Heart Disease, we calculate **Specificty**, which is the percentage of the actual **Negatives** that were *correctly* classified
* **Specificty** = **True Negatives / True Negatives + False Positivies**
    * 115 / 115 + 20 = 0.85 -> which means that **85%** of the people *without* Heart Diaseas were correctly classified

### Precision and Recall
**Precision** - tells us the percentage of the *predicted* **Positive** results (so, both **True** and **False Positives**) that were *correctly* classified
* **Precision** = **True Positivies / True Positives + False Positivies**
    * 142 / 142 + 22 = 0.87 -> which means that of the **164** people that we predicted to have Heart Disease, **87%** actually have it
    * In other words, **Precision** gives us a sense of the quality of the positive results
    * When we have high **Precision**, we have high-quality positive results

**Recall** - another name for **Sensitvity**, which is the percentage of the *actual* **Positives** that were *correctly* classified
* **Recall** = **Sensitivity** = **True Positives / True Positives + False Negativies**

### True Positive Rate and False Positive Rate
**True Positive Rate** is the same thing as **Recall**, which is the same thing as **Sensitivty**
* **True Positive Rate** = **Recall** = **Sensitivity** = **True Positives / True Positives + False Negativies**

**False Positive Rate** - tells you the percentage of *actual* **Negativies** that were *incorrectly* classified
* In this case, it is the known people *without* Heart Disease who were *incorrectly* classified
* **False Positive Rate** = **False Positives / False Positives + True Negatives**
    
**Note** - Remember, **Specificty** is the propotion of actual **Negativies** that were *correctly* classified, thus...
* **False Positive Rate** = 1 - **Specificty** and
* **Specificty** = 1 - **False Positive Rate**

### ROC
* Loves Troll 2 (>50%) vs Does Not Love Troll 2 (<= 50%)
* If it was super important to correctly classify every single person who Loves Troll 2, we could set the threshold to 0.01
    * Classify people with Ebola virus -> catch every single person with the virus to minimize the risk of an outbreak -> even if it results in more **False Positives** (Not Love but predicted as Love)  
* On the other hand, if it was super important to correctly classify everyone who Does Not Love Troll 2, we could set the classification threshold to 0.95
    * we would have **0 False Positives** because all of the people that we know Do Not Love Troll 2 would be correctly classified -> more **False Negatives** (Love but predicted as Not Love)
* We could set the classification threshold to 0, and classify everyone as someone who Loves Troll 2
* Or, we could set the classification threshold to 1 and classify everyone as someone who Does Not Love Troll 2
    * Ultimately, we can try any classification threshold from **0** to **1** - ending up many **Confusion Matrices**
 
 **ROC (Receiver Operating Characteristic)** graphs are super helpful when we need to identify a good classification threshold because they summarize how well each threshold performed in terms of the **True Positive Rate** and the **False Positive Rate**
 * y-axis - the higher the percentage of actual **Positives** were *correctly* classified
 * x-axis - the further to the left along the x-axis, the lower the percentage of actual **Negatives** that were *incorrectly* classified
 * If we want to avoid all **False Positives**, but want to maximize the number of actual **Positives** correctly classified, we pick the middle-top-left point
 * but if we can tolerate a few **False Positives**, we would pick top-top-but not too left point, because it *correctly* classifies all of the actual **Positives**

### AUC (Area Under Curve)
* Compare models - use **AUC** (the **Area Under** each **Curve**)
* **Logistic Regression** AUC = 0.9 vs **Naive Bayes** AUC = 0.8
* because **Logistic Regression** has a larger **AUC**, we can tell that overall, **Logistic Regression** performs better than **Naive Bayes** with these data

### Precision Recall Graphs
* An **ROC** graph uses the **False Positive Rate** on the x-axis, only fine when the data are *balanced*
* if the data are *imbalanced* -> **ROC** graph is hard to interpret becasue the **False Positive Rate** barely budges above **0** before we have a **100%** **True Positive Rate**
* In other words, the **ROC** graph makes any model that simply predicts **No 100%** of the time looks really good
* Use **Precision Recall** graphs to deal with this problem

**Precision Recall** graph simply replaces the **False Positive Rate** on the x-axis with **Precision** and renames the y-axis **Recall**, since **Recall** is the same thing as the **True Positive Rate**
* with **Precision** on the x-axis, good classification thresholds are closer to the right side, and now we can see a bend where the classification thresholds start giving us a lot **False Positives**
* The reason **Precision** works better than the **False Positive Rate** when the data are highly imbalanced is that **Precision** does not include the number of **True Negatives**

# Decision Trees
Two types of **Trees** in machine learning:
1. Classification - trees classify people or things into two or more *discrete* categories
2. Regression -trees try to predict a *continuous value*
* **Root Node** - very top of the tree
* **Internal Node**
* **Branches** (Arrows) - labeled with **Yes** or **No**
    * usually that if statement in a **Node** is **True**, go to the *Left*, and if it is **False**, go to the *Right*
* **Leaf Nodes** (Leaves)

## Classification Tree
* We can't use **Logistic Regression** when we plot Age vs Loves Troll 2, unable fit an **s-shaped squiggle** to the data, both young and old people *do not* love Troll 2, with the people who love Troll 2 in the middle
    * In this example, an **s-shaped squiggle** will incorrectly classify *all*  of the older people

**Classification Tree**, which can handle all types of data, all types of relationships among the independent variables (the data we are using to make predictions, like Loves Soda and Age), and all kinds of relationships with the dependent variable (the thing we want to predict, which in this case is Loves Troll 2)
1. First thing - Decide whether Loves Popcorn, Loves Soda, or Age should be the very top of the tree
    * to make the decision, start by looking at how well each feature predicts whether or not someone Loves Troll 2
* **Leaves** that contain mixtures of classifications are called **Impure**
* Several ways to quantify the **Impurity** of **Leaves** and **Trees**
    * most popular method is, **Gini Impurity**
    * there are also fancy-sounding methods like **Entropy** and **Information Gain**
    * In theory, all of the methods give similar results
2. Start calculating the **Gini Impurity** to quantify the **Impurity** in the **Leaves** - here for Loves Popcorn
    * to calculate the **Gini Impurity** for Loves Popcorn, first we calculate the **Gini Impurity** for each individual **Leaf**
    * start by plugging the numbers from the *left* **Leaf** into the equation for **Gini Impurity**
    * **Gini Impurity** for a **Leaf** = 1 - (the probability of "yes")^2 - (the probability of "no")^2
        * For the **Leaft** on the *left* (Loves Popcorn - Yes), Loves Troll 2 - Yes = 1, No = 3, and Total = 1 + 3
        * 1 - (1 / 1+3)^2 - (3 / 1+3)^2 = 0.375 **Gini Impurity**
        * For the **Leaft** on the *right* (Loves Popcorn - No), Loves Troll 2 - Yes = 2, No = 1, and Total = 2 + 1
        * 1 - (2 / 2+1)^2 - (1 / 2+1)^2 = 0.444 **Gini Impurity**
     * *left* **Leaf** has 4 people and *right* **Leaf** - to compensate for the differences in the number of people in each **Leaf**, the total **Gini Impurity** for Loves Popcorn is the **Weighted Average of the two *Leaf* Impurities** 
     * **Total Gini Impurity** = weighted average of **Gini Impurities** for the **Leaves**
        * the weight for the *leaf* **Leaf** is 4, and *right* **Leaf** is 3 - total 7
        * Total **Gini Impurity** = (4 / 4 + 3) * 0.375 + (3 / 4 + 3) 0.444 = 0.405 (**Gini Impurity** for Loves Popcorn)
     * **Gini Impurity** for Loves Popcorn = 0.405 vs **Gini Impurity** for Loves Soda = 0.214
     * The lower **Gini Impurity** for Loves Soda, 0.214, does a better job classifying people who love and do not love Troll 2
3. Calculate the **Gini Impurity** for Age (numeric data), not just **Yes/No** values, calculating the **Gini Impurity** is a little more involved
     * First thing we do is sort the rows by Age, from lowest to highest
     * Second thing we do is calculate the average Age for all adjacent rows
     * Third, we calculate the **Gini Impurity** values for each average Age
         * For example, the first average Age is 9.5, so we use 9.5 as the threshold for splitting the rows into 2 leaves - Total Gini Impurity = 0.429
         * average **Age 15** - Gini Impurity = **0.343**
         * average Age 26.5 - Gini Impurity = 0.476
         * average Age 36.5 - Gini Impurity = 0.476
         * average **Age 44** - Gini Impurity = **0.343**
         * average Age 66.5 - Gini Impurity = 0.429
         * Identify the thresholds with the lowest **Impurities**, and because the candidate thresholds **15** and **44** are tied for the lowest **Impurity**, **0.343**, we can pick either one for the **Root**
         * In this case, we will pick **15**
4. Remember, our first goal was to determine whether we should ask about Loves Popcorn, Loves Soda, or Age at the very top of the tree
    * because Loves Soda has the lowest **Gini Impurity**, we will put it at the top of the tree
        * **Gini Impurity** for Loves Popcorn = 0.405
        * **Gini Impurity** for Loves Soda = 0.214
        * **Gini Impurity** for Age < 15 = 0.343
5. After we have the Top root, we will continue to split the tree if the **Gini Impurity** at one of the node's **Gini Impurity** is not 0
    * We pick Age < 12.5 among the remaining 4 data points from the node, because the **Gini Impurity** is **0.0** vs Loves Popcorn **Gini Impurity 0.25**
    * Once we add the branch Age < 12.5 and return **Gini Impurity** = 0.0, now these new nodes are Leaves because neither is **Impure**
    * Lastly, assign output values for each **Leaf**, Loves Troll 2 or Does No Love Troll 2
6. We are not confident that the tree will do a great job making predictions with future data if so few people in the **Training Data** made it to a **Leaf**
   * Apply **Pruning**
   * Limits on how trees grow, for example, by requiring **3** or more people per **Leaf**
        * **Leaf** would be **Impure** but we would also have a better sense of the accuracy of predictions
        * Use **Cross Validation** and pick the number that works best
        * **Note**: even though the **Leaf** is **Impure**, it still needs an output value and because most of the people in this **Leaf** love Troll 2, that will be the output value

## Regression Tree
Just like a **Classification Tree**, a **Regression Tree** can handle all types of data and all types of relationships among variables to make decisions, but now the output is a *continuous* value, which, in this case, is Drug Effectiveness
1. Just like for **Classification Tree**, the first thing we do for a **Regression Tree** decides what goes in the **Root**
* First build a very simple tree that splits the measurements into 2 groups
    * Does < 3 (Does < 3 Yes - Average = 0, Does >= 3 - No Average = 38.8)
    * For a specific point, which has Does > 3 and 100% Effectiveness, the tree predicts that the Effectiveness will be 38.8, which is a pretty bad prediction
* We can *visualize* how good or bad the **Regression Tree** is making predictions by drawing the **Residuals**, the differences between the Observed and Predicted values
* We can also *quantify* how good or bad the predictions are by calculating the **SSR**
    * and when the threshold for the tree is Does < 3, then the **SSR** = 27,468.5
    * Lastly, we can compare the **SSR** for different thresholds by plotting them on the graph, which has Does on the x-axis and **SSR** on the y-axis
    * Shift to Does < 5 (average of the second and third measurements) (Does < 5 Yes - Average = 0, Does >= 5 - No Average = 41.1), **SSR** less than Does < 3
    * Shift to Does < 7 (average of the third and fourth measurements) (Does < 7 Yes - Average = 0, Does >= 7 - No Average = 43.7)
    * Keep shifting and compare **SSR** -> Does < 14.5 had the smallest **SSR** = **Does < 14.5**  will be the **Root** of the tree
    * Which corresponds to splitting the measurements into two groups based on whether or not the Does < 14.5
    * There are 6 data points on the **Does < 14.5 Yes side**, *in theory*, we could subdivide these 6 data points
    * However, keep splitting, and ending one single measurement (data point) in the **Leaf** = **Overfit**
    * To prevent, often keep **minimum 20 data points** to avoid overfitting, in our example, we set the number for 7 data points, which will be a **Leaf**
    * Continue to split for the remaining data points Does >= 29, and only 4 measurements satisfy -> make it as a **Leaf**
    * Continue to split, because there are more than 7 measurements with Does between 14.5 and 29, and thus, with more than 7 measurements in the **Node**, we can split the measurements into two groups further by finding the Does threshold that results in the lowest **SSR**
2. Just like **Classification Tree**, Does < 14.5 is a *candidate* for the Root if there are other variables like Age and Sex
* First find the threshold for Does that gives us the smallest **SSR**, then Age (second *candidate*) for the **Root**
* Third, Sex, even though Sex only has one threshold for splitting the data, we still calculate the **SSR**, just like before, and Sex becomes the third *candidate* for the Root
* Compare the **SSRs** for each *candidate* for the **Root** -> pick the one with the lowest value, and because Age > 50 had the lowest **SSR**, it becomes the **Root** of the **Regression Tree**
    * We continue to split, but this time, Does, Age, and Sex will continue to be our candidates, and we select whichever gives us the lowest **SSR** until we can no longer subdivide the data any further. At that point, we are done building the **Regression Tree** 

# Random Forest
* **Decision Trees** are easy to build, easy to use, and easy to interpret, but in practice, they are not that awesome
    * Trees have one aspect that prevents them from being the ideal tool for predictive learning, namely **inaccuracy**
    * In other words, they work great with data used to create them, but **they are not flexible when it comes to classifying new samples**

**Random Forests** combine the simplicity of decision trees with flexibility resulting in a vast improvement in accuracy
1. Step 1 - Create a "bootstrapped" dataset
* To create a bootstrapped dataset that is the same size as the original, we just randomly select samples from the original dataset
* The important detail is that we are **allowed to pick the same sample more than once**
2. Step 2 - Create a **Decision Tree** using the bootstrapped dataset
* But, only use a random subset of variables (columns) at each step
* Instead of considering all 4 variables (in this case) to figure out how to split the root node, we randomly select 2 as candidates for the root node
* After the select a variable as the root node, we randomly select 2 variables as candidates again (the same feature (or variable) can be selected **multiple times** in a tree)
* Every time we select a subset of features to choose from, we choose from the full list of features, even if we have already used some of those features
    * Thus, a single feature can appear multiple times in a tree
* And we just build the tree as usual, but only considering **a random subset of variables** at each step
3. Now go back to Step 1 and repeat
* Make a new bootstrapped dataset and build a tree considering a subset of variables at each step (ideally, we do this 100's of times)
* Using a bootstrapped sample and considering only a subset of the variables at each step results in a wide variety of trees
* The variety is what makes random forests more effective than individual decision trees

### How do we use Random Forest?
* We have a new observation (a new patient, with all measurements), and now we want to know if they have heart disease or not
* Then, we take the data and run it down the first tree that we made - Tree 1 says "Yes"
* Then, we run the data down the second tree that we made - Tree 2 says "Yes"
* Repeat for all the trees that we made
* After running the data down all of the trees in the random forest, we see which option received more votes (in this case Total 6 trees, **5 Yes**, 1 No) 
    * We conclude that this patient has heart disease 

### Terminology Alert
**Bagging** - **B**oostrapping the data plus using the **agg**regate to make a decision
* Typically, about 1/3 of the original data does not end up in the bootstrapped dataset

**Out-Of-Bag Dataset** - the data that didn't make it into the bootstrapped dataset
* the 1/3 that does not end up in the bootstrapped dataset

### How do we know if the Random Forest is any good?
* We can run this (the) **Out-Of-Bag** sample (dataset) through all of the other trees that were built without it
1. Ultimately, we can measure how accurate our **Random Forest** is by the proportion of **Out-Of-Bag** samples that were correctly classified by the **Random Forest**
2. The proportion of **Out-Of-Bag** samples that were *incorrectly* classified is the **Out-Of-Bag Error**
* Now, we can compare the **Out-Of-Bag Error** for a random forest built using only **2 variables** per step to **Random Forest** built using **3 variables** per step
* And we test a bunch of different settings and choose the most accurate **Random Forest**
* Typically, we start by using the **square root** of **the number of variables** and then try a few settings above and below that value

# XGBoost

## XGBoost Regression
**Note: XGBoost** was designed to be used with large, complicated data sets.
1. Step 1 - Fitting **XGBoost** to **Training Data** is to make an initial prediction
* This prediction can be anything, but by default it is **0.5**, regardless of whether you are using **XGBoost** for **Regression** or **Classification**
* Unlike **Gradient Boost**, which uses regular off-the-shelf, **Regression Trees**, **XGBoost** uses a unique **Regression Tree** that StatQuest called an **XGBoost Tree**
2. Step 2 - Each tree starts out as a single leaf and of the **Residuals** go to the leaf
* calculate a **Quality Score**, or a **Similarity Score** for the **Residuals**
* **Similarity Score** = Sum of Residuals, Squared / Number of Residuals + λ (Lambda: **Regularization** parameter, for now it is **0**)
    * Example: **4 Residuals**, (-10.5, 6.5, 7.5, -7.5)^2 / 4 + 0
    * **Note:** We do not square the **Residuals** *before* we add them together in the numerator, **7.5** and **-7.5** cancel each other out
    * Likewise, **6.5** cancels out most of **-10.5** = (-4)^2 / 4, Thus **Similarity Score** for the **Residuals** in the root = **4**
    * Question is, whether or not we can do a better job clustering similar **Residuals** if we split them into two groups
    * Example: Dosage < 15, Left -10.5, Right 6.5, 7.5, -7.5
    * Left Similarity Score = (-10.5)^2 / (1 + 0) = 110.25, Right Similarity Score = (6.5 + 7.5 + -7.5)^2 / (3 + 0) = (6.5)^2 / 3 = 14.08
    * If **Residuals** in a node are very different, they cancel each outer and the Similarity Score is relatively **small**
    * In contrast, when the **Residuals** are similar, or there is just one of them, they do not cancel out and the Similarity Score is relatively **large**
3. Step 3 - Quantify how much better the leaves cluster similar Residuals than the root
* Calculating the **Gain** of splitting the **Residuals** into two groups
* Gain = Left Similarity Score + Right Similarity Score - Root Similarity Score
    * 110.25 + 14.08 - 4 = 120.33
    * Gain for the threshold Dosage < 15, compare it to the Gain calculated for other thresholds
* Dosage < 22.5 (New Threshold)
    * Left Leaf = -10.5, 6.5, Similarity Score = 8
    * Right Leaf =  7.5, -7.5, Similarity Score = 0
    * Gain = 8 + 0 - 4 = 4
* Since the Gain for Dosage < 22.5) is less than Gain for Dosage < 15)
    * Gain = 4 vs Gain 120.33, Dosage <15 is better at splitting the Residuals into clusters of similar values
* Dosage < 30 (New Threshold)
    * Left Leaf = -10.5, 6.5, 7.5, Similarity Score = 4.08
    * Right Leaf = -7.5, Similarity Score = 56.25
    * Gain = 4.08 + 56.25 - 4 = 56.33
    * Gain = 56.33 vs Gain 120.33, Dosage <15 is better
* Use the threshold that gave us the **largest Gain**, the Dosage < 15
4. Step 4 - Continue Split using the threshold (Example: Continue to split the right Leaf)
* Dosage < 22.5
    * Left Leaf, 6.5, Similarity Score = 42.25
    * Right Leaf 7.5, -7.5, Similarity Score = 0
    * Gain = 42.25 + 0 - 14.08 ((6.5 + 7.5 + -7.5)^2 / (3 + 0) = (6.5)^2 / 3 = 14.08 Root Node) = 28.17
* Dosage < 30
    * Left Leaf, 6.5, 7.5, Similarity Score = 98
    * Right Leaf -7.5, Similarity Score = 56.25
    * Gain = 98 + 56.25 - 14.08 = 140.17
    * Gain 140.17 > Gain 28.17 = Dosage < 30 is better
    * We can continue to split the Left Leaf, but we keep it for **Level 2**, default (max_depth) is **Level 6**
    * Output Value = Sum of Residuals / Number of Residuals + λ
    * Similarity Score = Sum of Residuals, Squared / Number of Residuals + λ
    * **Note:** The Output Value equation is like the Similarity Score except we do not square the sum of the residuals
5. Step 5 - Prune Tree based on Gain values
* Start by picking a number, for example, 130, γ (gamma)
* Calculate the difference between the Gain associated with the lowest branch in the tree and the value for γ (gamma)
* Gain - γ
    * If difference between Gain and γ is **negative** = remove the branch
    * If **positive** = will not remove the branch
    * 140.17 - 130 = 10.17 = will not remove = done pruning
* **Note:** The Gain for the root, 120.3 < 130 (γ), so difference will be **negative**
    * However, because we did not remove the first branch, we will not remove the root
    * In contrast, if we set γ = 150, then we would remove this branch, 140.17 - 150 = a negative number
    * Root, 120.33 - 150 = a negative number = Removed
6. Step 6 - Recalcualte Similarity Scores with λ (Lambda) = 1 (intended to reduce the prediction's sensitivity to individual observations)
* **4 Residuals**, (-10.5, 6.5, 7.5, -7.5)^2 / 4 + 1 = 3.2 = 8/10 when λ = 0 = 20%
* Left Leaf = (-10.5)^2 / (1 + 1) = 55.12 = half of when λ = 0 = largrest decrease, 50%
* Right Leaf = (6.5)^2 / (3 + 1) = 10.56 = 3/4s of when λ = 0
* Gain = 55.12 + 10.56 - 3.2 = 62.48, a lot less than 120.33, the value when λ = 0
* When **λ greater than 0**, Similarity Scores are **smaller**
    * the amount of decrease is inversely proportional to the number of Residuals in the node
* Gain < 130, Prune the Whole Tree Away = Remove All = When λ > 0, easier to prune leaves because the values for Gain are smaller
* **Note:** If γ (gamma) = 0, Gain = -16.07, -16.07 - 0 = -16.07, still prune this branch (the tree), γ (gamma) = 0 does not turn off pruning
7. Step 7 - Making New Prediction
* Starting with the initial Prediction, 0.5 + Learing Rate (eta, default value = 0.3)
* 0.5 + (0.3 * - 10.5 (Output Value)) = -2.65, -2.65 vs - 10.5 (new Residual is smaller than before = small step in the right direction)
8. Step 8 - Build Another Tree based on the New Residuals
* Continue to make new prediction that give us even smaller residuals...and then build another tree based on the newest residuals...
* Until the Residuasl are super small, or we have reached the maximum number

### XGBoost Trees for Regression Summary
* Calcualte Similarity Scores and Calcualte Gain to Determine how to split the data
* Prune the tree by calculating the differences between Gain values and a user defined Tree Complexity Paramter, γ (gamma)
    * If positive, do not prune
    * If negative, then prune, work our way up the tree
* Then calcuate the Output Value for the remaining leaves
* Lastly,  when  λ (Lambda) > 0, it results in more pruning, by shrinking the Similarity Scores, and it results in smaller Output Values for the leaves

## XGBoost Classification
* Initial Prediction = 50% chance the drug is effective
* Classification has a new formula for the Similarity Scores
* Similarity Scores = Sum of Residuals, Squared / Sum of Previous Probability * (1 - Previous Probability) + λ (Lambda)
1. Step 1 - Build Tree
* Start a single leaf and all residuals go to the leaf
* Calculate Similarity Scores = (-0.5 + 0.5 + 0.5 + -0.5)^2 = 0
* **Note:** We do not square the Residuals before we add them together, they will cancel each other out
* Now decide if we can do a better job clustering similar Residuals if we split them into two groups
* Dosage < 15 (we chose 15, because it is the average value between last two observations)
    * Left Leaf = -0.5, 0.5, 0.5, Similarity Scores = (-0.5 + 0.5 + 0.5)^2 / (0.5 (Initial Prediction) * (1 - 0.5)) + (0.5 * (1 - 0.5)) + (0.5 * (1 - 0.5)) + 0 = 0.33
    * Right Leaf = -0.5, Similarity Scores = (-0.5)^2 / (0.5 * (1 - 0.5)) = 1
    * Gain = 0.33 + 1 - 0 = 1.33 = First Branch in the Tree (assume no other threshold gives us a larger Gain Value)
2. Step 2 - Continue Split using the threshold
* Dosage < 10 (Dosage < 5 has a higher Gain than Dosage 10)
    * Leaf Leaf = -0.5, 0.5, Similarity Scores = 0
    * Right Leaf = 0.5, Similarity Scores = 1
    * Gain = 0 + 1 - 0.33 = 0.66
* Dosage < 5 (2.66 > 0.66) = New Branch Threshold)
    * Leaf Leaf = -0.5, Similarity Scores = 1
    * Right Leaf = 0.5, 0.5, Similarity Scores = 2
    * Gain = 1 + 2 - 0.33 = 2.66
**Note:** XGBoost also has a threshold for the minimum number of Residuals in each leaf
* The minimum number of Residuals in each leaf is determined by calculating something called **Cover**
* **Cover** = Sum of Previous Probability * (1 - Previous Probability) = the Demominator of the Similarity Score
* **Cover** is equal to Number of Residuals in a leaf, by default the minimum value for **Cover** is **1**
* Thus, by default, when we use XGBoost for Regression, we can have as few as 1 Residual per leaf
    * In other words, when we use XGBoost for Regression and use the default minimum value for Cover, Cover has no effect on how we grow the tree
* Classification more complicated, because Cover depends on the previously predicted probability of each Residual in a leaf
* Cover for Left Leaf (-0.5) = 0.5 * (1 - 0.5) = 0.25
    * And since the default value for the minimum Cover is 1, XGBoost would not allow this leaf
    * Right Leaf = 0.5, 0.5, Cover = 0.5, also would not allow this leaf either
    * Both are removed
* Go back to Left Leaf = -0.5, 0.5, 0.5, Cover = 0.75, woud not allow this leaf either
    * Ultimately, if we used the default minimum value for Cover, 1, then we would be left with the Root
    * And XGBoost requires trees to be larger than just the Root
* Set the minimum value for Cover = 0, min_child_weight = 0 (default value is 1)
3. Step 3 - Prune Tree based on Gain values
* 2.66 - γ (set γ = 2) = would not prune because the difference is a **positive** number
* In contrast, if we set γ = 3, would prune because the difference is a **negative** number
* Also prune the Root, 1.33 - 3 = a **negative** number, left with original prediction 0.5
Step 4 - Recalcualte Similarity Scores with λ (Lambda)
* λ (Lambda) **reduces** the Similarity Scores, and that **lower Similarity Scores** results in **lower values for Gain**
* λ (Lambda) = lower values for Gain
* That means a lower value for γ (gamma) will result in a negative difference and cause us to prune branches
* In other words, values for λ (Lambda) greater than 0 reduce the sensitivity of the tree to individual observations by pruning and combining them with other observations
5. Step 5 - Determine the Output Values for the leaves
* Output Value = Sum of Residuals / Sum of Previous Probability * (1 - Previous Probability) + λ (Lambda)
* Left Leaf = -0.5 / 0.5 * (1 - 0.5) + 0 (No Regularization) = -2
* Left Leaf = -0.5 / 0.5 * (1 - 0.5) + 1 (With Regularization) = -0.4 = closer to zero
* In other words, when λ > 0, then it reduces the amount that this single observation adds to the new prediction
    * λ reduces the prediction's sensitivity to isolated observations
* λ = 0, Left Leaf Output Value = -2, Right Leaf Output Value = 2
* λ = 1, Left Leaf Output Value = -0.4, Right Leaf Output Value = 0.67 (the effect of λ is smaller this time because there are two observations in this leaf)
* λ = 0, upper level Right Leaf Output Value = -2
6. Step 5 - Making New Prediction
* Starting with the initial Prediction， 0.5, convert this probability to a **log(odds)** value
* p / 1 - p = odds
* log (p / 1 - p) = log(odds) = log (0.5 / 0.5) = 0 = log(odds)
* Output = log(odds) = 0 (initial prediction) + 0.3 (default value eta, Learning Rate = 0.3)
* log(odds) Preidction = 0 + (0.3 * -2) = -0.6
* Convert log(odds) into a probability into Logistic Function
    * Probability = e^log(odds) / 1 + e^log(odds) = e^-0.6 / 1 + e^-0.6 = 0.35
* Original Prediction was 0.5, Original Residual was also 0.5, New Predicted Probability is 0.35 = Smaller Residual = a small step in the right direction
* Initial Prediction can be changed at any time other than 0.5
    * For example, if 75% of the observations in the Training Data said that the drug was effective, we might set the initial prediction to 0.75 and now initial log(odds) = 1.1
    * log(odds) Preidction = 1.1 + (0.3 * -2) = 0.5
* Make a new prediction, with Dosage = 8 
    * log(odds) Preidction = 0 (use 0 for remaining example) + (0.3 * 2) = 0.6 = e^0.6 / 1 + e^0.6 = 0.65 = Smaller Residual
7. Step 7 - Build Another Tree based on the New Residuals
* **Note:** When we build the second tree, calculating the Similarity Scores is a little more interesting because the Previous Probabilities are no longer the same for all of the observations
* For example, since all of the Residuals start in the root of the tree
    * We would plug in the previously predicted probabilities for each observation into the denominator, and this time they are not all the same
    * Sum of Residual, Squared / (0.35 * (1 - 0.35)) + (0.65 * (1 - 0.65)) + (0.65 * (1 - 0.65)) + (0.35 * (1 - 0.35)) + λ
* Similarly, Output Value for the Root will be
    * Sum of Residual / (0.35 * (1 - 0.35)) + (0.65 * (1 - 0.65)) + (0.65 * (1 - 0.65)) + (0.35 * (1 - 0.35)) + λ
* Make new predictions that give us even smaller Residuals
* Continue build another tree based on the new Residuals
* Until the Residuasl are super small, or we have reached the maximum number of trees

### XGBoost Trees for Classification Summary
* Calculate Similarity Scores and Calculate Gain to Determine how to split the data
* Prune the tree by calculating the differences between Gain values and a user defined Tree Complexity Parameter, γ (gamma)
    * If positive, do not prune
    * If negative, then prune (subtract γ (gamma) from the next Gain value (etc. etc. etc.), work our way up the tree
* Then calculate the Output Value for the remaining leaves
* Lastly, when λ (Lambda) > 0, it results in more pruning, by shrinking the Similarity Scores, and it results in smaller Output Values for the leaves
* **Note:** when using XGBoost for Classification, we have to be aware that the minimum number of Residuals in a leaf is related to a metric called **Cover**, which is the denominator of the Similarity Score, minus λ (Lambda) (without λ (Lambda))

## XGBoost Optimizations
#### Approximate Greedy Algorithm
* Instead of testing every single threshold, we could divide the data into **Quantiles**
* and only use the quantiles as candidate thresholds to split the observations
* more quantiles we have, the more thresholds we will have to test, and that means it will take longer to build the tree
* By default, the Approximate Greedy Algorithm uses about 33 quantiles
* Parallel Learning = **Quantile Sketch Algorithm**, combines the values from each computer to make an *approximate histogram*
* **XGBoost** uses a **Weighted Quantile Sketch**
    * Weigh for each observation is the 2nd derivative of the **Loss Function**, what we are referring to as the **Hessian**
    * **Regression, the **Weights** are all equal to **1**
    * **Classification**, the **Weights** = Previous Probability * (1 - Previous Probability)
* Predicted probability is close to **0.5** = not much confidence, weights are relatively large
* In contrast, when the previously predicted probability is very close to **0** or **1** = a lot of confidence in the classification, weights are relatively small
* Observations will end up in the same leaf together in the tree (same quantiles)
    *  Positive residual will cancel out the negative residual
    *  Divide the observations into quantiles where **the sums of the weights are similar**
    *  Split low confidence predictions into separate bins
* Summary
    * XGBoost uses an **Approximate Greedy Algorithm**
    * Uses **Parallel Learning** to split up the dataset so that multiple computers can work on it at the same time
    * a **Weighted Quantile Sketch** merges the data into an *approximate* histogram
    * the histogram is divided into **weighted quantiles** that put observations with low confidence predictions into quantiles with fewer observations
    * When the **Training Datasets** are small, XGboost just uses a normal, every day **Greedy Algorithm**
    * **XGBoost** can also speed things up by allowing you to build each tree with only a random subset of the data
    * And **XGBoost** can speed up building trees by only looking at a random subset of features when deciding how to split the data
 
#### Sparsity-Aware Split Finding
* Missing values, calculate the Residual (using initial prediction 0.5)
* Put all Residuals into a single leaf
    * Split Residuals into two leaves will do a better job that cluster them
* Split tables with one will no missing value (sort from low to high like other continuous variables), and one with missing values
* Split the between two Observations or Split the Quantile for New Threshold
    * Put the missing value that fit into the New Threshold
    * For example, put into Left Leaf or Right Leaf to calculate Gain Value
    * Choose the threshold that gave us the largest value for Gain
    * **Note:** Once is chosen, the default path for all future observations that contains missing values
  
## Hyper-Parameter
**Random Forest**
* n_estimates: the number of trees to be used in the forest
    * More trees improve stability but slow down training
    * Too many = Overfitting
* max_depth = controls tree size
    * Deeper trees = More complex models (higher variance, possible overfitting
* min_samples_leaf = Ensures each leaf has at least this many samples. Higher values prevent overfitting
* min_samples_split = Minimum samples required to split a node. Higher values reduce overfitting
* criterion: gini, max_dept, min_sample_split, max_feature, boostrap

**XGBoost**
* n_estimates: number of trees
    * More trees improve performance but require early stopping to prevent overfitting
* max_depth: depth of tree
* min_child_weight: control depth of tree
    * Stop trying to split once your sample size in a node goes below a given threshold
* gamma
* learning_rate = 
* subsample
* early_stopping_rounds
* alpha, lambda

## Random Forests: Missing Data and Sample Clustering
**Random Forests** consider 2 types of missing data
1. Missing data in the original dataset used to create the 
* The general idea for dealing with missing data is to make an initial guess that could be bad, then gradually refine the guess until it is (hopefully) a good guess
    * Categorical - pick the most common value as the initial guess (depends on the Yes/No label - the original dataset)
    * Numeric - pick the median value (depends on the Yes/No label - the original dataset) 
* Now to refine these guesses, we do this by first determining which samples are similar to the one with missing data

How to determine similarity?
1. Step 1: Build a **Random Forest**
2. Step 2: Run all of the data down all of the trees
* First, we will start by running all the data down the first tree
    * Both Row 3 (normal entry) and Row 4 (missing data entry) end up at the same **Leaf** Node, meaning that they are similar 
* We keep track of similar samples using a **Proximity Matrix** (count the appearance of pair samples)
    * **1** in the **Proximity Matrix** means the samples are as close as close can be
* Then run all the data down the second tree
    * Row 2, 3, and 4 all ended up in the same **Leaf** Node, continue to keep track of that
* Then run all the data down the third tree
    * Row 3 and 4 ended up in the same **Leaf** Node - 3 times appear for Row 3 and 4 / Row 4 and 3 in the **Proximity Matrix**
* Ultimately, we run the data down all the trees and the **Proximity Matrix** fills 
* Then we divided each **Proximity Value** by the total number of tree
    * In this example, we have 10 trees and 8 times appear for Row 3 and 4 / Row 4 and 3 = 0.8
* Now we use the proximity values for row 4 to make better guesses about the missing data

For categorical, we calculate the weighted frequency of "Yes" and "No" using **Proximity Value** as the weights
* The frequency (column A)- Yes = 1/3, No = 2/3
* The weighted frequency - Yes = 1/3 * The weight for "Yes" (0.1) = **0.03**
* The weight for "Yes" = Proximity of "Yes" / All Proximities
    * In this example, the **Proximity Value** is row 2 (the only one with "Yes)
    * 0.1 / 0.1 + 0.1 + 0.8 = 0.1 
* The weighted frequency - No = 2/3 * The weight for "No" (0.9) = **0.6**
* The weight for "No" = Proximity of "No" / All Proximities
    * In this example, the **Proximity Value** are row 1 and row 3 (both have "No)
    * 0.1 + 0.8 / 0.1 + 0.1 + 0.8 = 0.9
* No has a way higher weighted frequency so we will assign No to row 4 (the one with missing data)

For numeric, we use the proximities to calculate a weighted average
* Weighted average = row 1 (column B value 125) * row 1's weighted average column value
    * 0.1 / 0.1 + 0.1 + 0.8 (the sum of the proximities) 
    * (125 * 0.1)
* Weighted average = row 1 (column B value 180) * row 2's weighted average column value (180 * 0.1)
* Weighted average = (125 * 0.1) + (180 * 0.1) + (210 * 0.8 (the pair count, which 8 times appear)) = **198.5**

After we have revised our guesses a little bit, we do the whole thing over again
* Build a ****
* Run the data through the trees
* Recalculate the **Proximities** and recalculate the missing values
* Do this 6 or 7 times until the missing value converge (i.e. no longer change each time we recalculate)
* With the **Proximity Value: 1**, 1 - the **Proximity Value** = **Distance** 
    * Close as can be = No distance between: 0
    * Not close = Far Away: 0.9
    * We can draw a heatmap or an MDS plot to show how the samples are related to each other 

1. Missing data in a new sample that you want to categorize
* Create two copies of the data
    * one has heart disease (Yes)
    * one doesn't have heart disease (No)
* Then we use the iterative method to make a good guess about the missing values
* Then we run the two samples down the trees in the forest and we see which of the two is correctly labeled by the  the most times

# Support Vector Classifiers and Machines (SVMs)
**Support Vector Machine** can add a new axis to the data and move the points in a way that makes it relatively easy to draw a **straight line** that correctly classifies people

The 3 main steps for creating **Support Vector Machines**
1. Step 1: Start with low-dimensional data
* In this case, we start with **1-Dimensional** data on a number line (Popcorn Consumed (g))
2. Step 2: Then use the existing data to create higher dimensions
* In this example, we created a **2-Dimensional** data by squaring the original Popcorn measurements (Popcorn^2)
3. Step 3: Then find a **Support Vector Classifier** that separates the higher dimensional data into two groups

### How do we decide how to transform the data?
* To make the mathematics efficient, **Support Vector Machines** use something called **Kernel Functions** that we can use to *systematically* find **Support Vector Classifier** in higher dimensions
* Two of the most popular **Kernel Functions**
1. **Polynomial Kernel**: (a * b + r)^d
    * where **a** and **b** refer to two different observations in the data
    * **r** determines the coefficient of the polynomial
    * and **d** determines the degree of the polynomial
    * In the example, we set **r = 1/2** and **d = 2** after a lot of math, we ended up with the **Dot Product**
2. **Radial Basis Function**: e^γ(gamma)^(a - b)^2

### Terminology Alert
* The name **Support Vector Classifier** comes from the fact that the points that define the threshold, and any points close to the threshold are called **Support Vectors**
* The distance between the points that define the threshold and the threshold itself is called the **Margin**
* And when we allow misclassifications, the distance is called the **Soft Margin**

# Preventing Overfitting with Regularization
Overfitting - *low* **Bias** because it fits the **Training Data** well, but *high* **Variance** because it does a bad job with **New Data**

**Regularization** reduces how sensitive the model is to the **Training Data**
* **Regularization** *increases* **Bias** a little bit, but in return, get a big *decrease* in **Variance**
* **Note**: **Ridge** and **Lasso** can be used with almost any machine learning algorithm to improve performance

## Ridge/Squared/L2 Regularization
* normally, when we fit a line to **Training Data**, we want to find the **y-axis intercept** and the **slope** that minimize the **SSR**
* In contrast, when we use **Ridge Regularization** to optimize parameters, we simultaneously minimize the **SSR** and a penalty that's proportional to the square of the slope
* **SSR** + λ * slope^2 
* ***λ***, is a positive number that determins how strong an effect **Ridge Regularization** has on the **new line** (new model)
* Line fit the **Training Data** perfectly, **SSR** = 0
    * 0 + 1 (lets make lambda as 1 for now) * 1.3^2 (slope) = 1.69 (**Ridge Score**)
* Line that doesn't fit the **Training Data** as well, **SSR** = 0.4, 0.4 + 1 * 0.6^2 (slope) = 0.76 (**Ridge Score**)
* We pick the line that minimizes the **Ridge Score**, which is the **0.76**

### Lambda λ
* When **λ** = 0, **Ridge Score** = 0, that means we only minimize the **SSR**
* As we continue to increase **λ**, the slope gets closer and closer to 0 and the y-axis intercept becomes the average Height in the Training Dataset (1.8).
* In other words, Weight no longer plays a significant role in making predictions, instead, we just use the mean Height

How do we pick a good value for **λ**?
* Unfortunately, there's no good way to know in advance what the best value for **λ** will be
* so we just pick a bunch of potential values, including 0, and see how well each one performs using **Cross Validation**

**A More Complicated Model**
* If we had a more complicated model that used Weight, Shoe Size, and Age to predict Height
    * Height = **intercept** + **slope w** * Weight + **slope s** * Shoe Size + **slope a** * Age
* The **Ridge Penalty** would include the sum of the squares of the 3 slopes associated with those variables
    * **SSR** + **λ** * (slope w^2 + slope s^2 + slope a^2)
* The **Ridge Penalty** never includes the **intercept** because the **intercept** doesn't directly affect how any of the variables (weight, SHoe Size, or Age) predict Height
* When we apply **Ridge Regularization** to models with multiple parameters, it will shrink the parameters, **slope w**, **slope s**, and **slope a**, but not equally 
    * Model -> Height = **intercept** + **slope w** * Weight + **slope s** * Shoe Size + **slope a** * Airspeed of a Swallow
* For example, if Weight and Shoes Size are both useful for predicting Height, but the Airspeed of a Swallow is not, then their slopes, **slope w**  and **slope s** , will shrink a little bit, compared to the **slope a** (the slope associated with Airspeed of a Swallow), which will shrink a lot
    * So, what causes this difference? When a variable, like Airspeed of a Swallow, is useless for making predictions, shrinking its parameter, **slope a** a lot, will shrink the **Rridge Penalty** a lot (without increasing the **SSR**)
    * In contrast, if we shrink the slopes associated with Weight and Shoe Size, which are both useful for making predictions, then the **Ridge Penalty** would shrink, but the **SSR** would increase a lot

### Ridge/Squared/L2 Regularization: FAQ
1. All of the examples showed how increasing **λ**, and thus decreasing the slope, made things better, but what if we need to increase the slope? Can **Ridge Regularization** ever make things worse?
* As long as you try setting **lambda** to 0, when you are searching for the best value for **λ**, in theory, **Ridge Regularization** can never perform worse than simply finding the line than minimizes the **SSR**
2. How do we find the optimal parameters using **Ridge Regularization**
* With one parameter (one slope), we can use **Gradient Descent**
* More complicated models, beyond the scope of current knowledge

## Lasso/Absolute Value/L1 Regularization
**Lasso Regularization**, also called **Absolute Value** or **L1 Regularization** - replaces the square that we use in **Ridge Penalty** with the absolute value
* **Ridge Penalty** - **SSR** + **λ** * slope^2
* **Lasso Penalty** - **SSR** + **λ** * |slope|
* Line fit the **Training Data** perfectly, **SSR** = 0
    * 0 + 1 (lets make lambda as 1 for now) * |1.3| (slope) = 1.3 (**Lasso Score**)
* Line that doesn't fit the **Training Data** as well, **SSR** = 0.4, 0.4 + 1 * |0.6| (slope) = 1.0 (**Lasso Score**)
* We pick the line that minimizes the **Lasso Score**, which is the **1.0**

The big difference between **Ridge** and **Lasso Regularization** is that
1. **Ridge Regularization** can only shrink the parameters to be asymptotically **close to 0**
2. In contrast, **Lasso Regularization** can shrink parameters **all the way to 0**
* Height = **intercept** + **slope w** * Weight + **slope s** * Shoe Size + **slope a** * Airspeed of a Swallow
* regardless of how useless the variable Airspeed of a Swallow is for making predictions, **Ridge Regularization** will never get **slope a = 0**
* In contrast, if Airspeed of a Swallow was totally useless, then **Lasso Regularization** can make **slope a = 0**, resulting in a simpler model that no longer includes Airspeed of a Swallow
* Height = **intercept** + **slope w** * Weight + **slope s** * Shoe Size + **slope a** * Airspeed of a Swallownger
* Height = **intercept** + **slope w** * Weight + **slope s** * Shoe Size ~~+ **slope a** * Airspeed of a Swallow~~
1. Thus, **Lasso Regularization** can exclude useless variables from the model and, in general, tends to perform well when we need to remove a lot of useless variables from a model
2. In contrast, **Ridge Regularization** tends to perform better when most of the variables are useful
#### Note: Ridge and Lasso Regularization are frequently combined to get the best of both worlds

### Ridge vs. Lasso Regularization
**Ridge**
1. When we calculate **SSR** + **λ** * **slope^2**, we get a nice curve for different values for the slope
2. When we increase **λ**, the lowest point in the curve corresponds to a slope value closer to 0, but not quite 0
**Lasso**
1. **Lasso** has a slight kink when the slope is 0, and as **λ** increases, we get a shape with the kink where the slope is 0 that is now more prominent
2. When **λ** is 40, a kink where the slope is 0, and now that kink is also the lowest point in the shape, and that means, when **λ** = **40**, the slope of the optimal line is 0
**Summary**
1. When we increase the **Ridge**, **Squared** the optimal value for the slope shifts toward 0, but we retain a nice **parabola** shape, and the optimal slope is never 0 itself
    * **Note**: Even if we increase **λ** all the way to **400**, the **Ridge Penalty** gives us a smooth curve and the lowest point corresponds to a slope value slightly greater than 0, but never 0 
2. In contrast, when we increase the **Lasso**, **Absolute Value**, or **L1 Penalty**, the optimal value for the slope shifts toward 0, and since we have a kind at 0, 0 ends up being the optimal slope

# K-Nearest Neighbors Algorithm
* A super simple way to classify data
1. Step 1: Start with a dataset with known categories
2. Step 2: Add a new observation that we don't know this observation's category yet
3. Step 3: Classify the new observation by looking at the **Nearest Neighbors**
* If the **K** in **K-Nearest Neighbors** is equal 1, then we only use the nearest neighbor to define the category
* If **K = 11**, we would use the **11 Nearest Neighbors**
* If **K = 11** and the new observation is between two (or more) categories, we simply pick the category that "gets the most votes"
    * For example, 7 nearest neighbors are RED, 3 are ORANGE, and 1 is GREEN
    * Since RED got the most votes, the final assignment is RED

#### Summary
* Low values for K (like K=1, or K=2) -> can result in **Overfitting**
* While too large a number may make the model overly generalized -> out voted by other categories
* There is no physical way to determine the best value for **K** -> A standard approach is to test multiple models with varying levels of K using **K-Fold Cross Validation**
* Hyperparameter: **the number of neighbors** (*n_neighbors*)

# K-Means Clustering: Unsupervised Learning
1. Step 1: Select the number of clusters you want to identify in your data - the **K** in **K-Means Clustering**
    * In this example, we want to identify 3 cluster -> select K = 3
2. Step 2: Randomly select 3 distinct data points (from the initial clusters)
3. Step 3: Measure the distance between the 1st point and the three initial clusters
    * Distance from the 1st point to BLUE, GREEN, and ORANGE cluster
4. Step 4: Assign the 1st point to the nearest cluster
    * In this case, the nearest cluster is the BLUE cluster
    * Repeat for the 2nd data point, measure the distances, and assign the point to the nearest cluster -> 3rd, 4th, 5th ... points -> until all of the points are in clusters
5. Step 5: Calculate the mean of each cluster
    * Repeat what we just did (measure and cluster) using the mean values
    * Assess the quality of the clustering by adding up the variation within each cluster
    * Since **K-means clustering** can't "see" the best clustering, its only option to keep track of these clusters, and their total variance, and do the whole thing over again with different starting points -> back at the beginning (Step 1)
6. Back to Step 1
    * picks 3 initial clusters and clusters all the remaining points
    * calculates the mean of each cluster
    * and then reclusters based on the new means - repeats until the clusters no longer change
    * sum the total variation within the clusters

#### How do we figure out what value to use for "K"?
* Try different K start with 1 (K = 1) -> K = 1 is the worst case scenario, we can quantify its "badness" with the total variation
* Then compare the total variation within the different numbers of K
* Each time we add a new cluster, the total variation within each cluster is smaller than before
    * And when there is only one point per cluster, the variation = **0**
* **Elbow Plot** - Plot the reduction in variance per value for K **Within Group Sum of Squares (WSS)** -> pick **K** by finding the **Elbow** in the plot
    * There will be a huge reduction in variation with K = 3, but after that, the variation doesn't go down as quickly 

#### How is K-means clustering different from hierarchical clustering
* **K-Means Clustering** specifically tries to put the data into the number of clusters you tell it to
* **Hierarchical Clustering** just tells you, pairwise, what two things are most similar

### What if data isn't plotted on a number line?
* Just like before, we pick 3 random points
* And we use the **Euclidean Distance** -> calculate the distances between things
    * In 2 dimensions, the **Euclidean Distance** is the same thing as the **Pythagorean Theorem**
    * If we have 2 samples (columns), or 2 axes the **Euclidean Distance** is square root x^2 + y^2
    * If we have 3 samples, or 3 axes the **Euclidean Distance** is square root x^2 + y^2 + z^2
    * If we have 4 samples, or 4 axes the **Euclidean Distance** is square root x^2 + y^2 + z^2 + a^2 -> and etc.
* Then, just like before, we assign the point to the nearest cluster
* And, just like before, we then calculate the center of each cluster and recluster

# Principal Component Analysis (PCA) 
* Starting by plotting the data (variable 1 on the x-axis, variable 2 on the y-axis)
    * then calculate the average measurement for variable 1 (mean x) and variable 2 (mean y)
    * with the average values we can **calculate** the **center of the data**
    * then we shift the data so that the center (the center point of mean x and mean y) is on top of the origin (0,0) in the graph
    * **Note**: shifting the data did not change how the data points are positioned relative to each other
    * Then start with a random line and rotate the line until it fits the data as well as it can, given that it has to go through the origin
    * To quantify how good the random line fits the data, **PCA** projects the data onto it
        * and then it can either measure the distances **from the data to the line** and try to find the line that **minimizes** those distances
        * **minimizes** = distances between line and point shrink = **b^2**
        * or it can try to find the **line** that **maximizes** distances **from the projected points** to the origin
        * **maximizes** = the distances of the line get large when the line fits better = **c^2** -> **easier to calculate **c^2** the distance from the projected point to the origin**
    * In Mathematical Way, **PCA** uses **Pythagorean Theorem**  - a^2 = b^2 + c^2
        * **a^2** (the distance of origin (center point) to a fixed data point) -> we get a right angle between a^2 and the random assigned line
        * if **b^2** gets bigger, then **c^2** must get smaller
        * likewise, if **c^2** gets bigger, then **b^2** must get smaller
* **PCA** finds the best fitting line by **maximizing the sum of the squared distances from the projected points to the origin**
    * measures the distances of all data points between the data point itself and the origin
        * d1, d2, d3, d4, d5, d6
     * Then next thing we do is square all of them
        * d1^2, d2^2, d3^2, d4^2, d5^2, d6^2 -> negative values don't cancel out positive values
     * Then sum up all these squared distances
        * d1^2 + d2^2 + d3^2 + d4^2 + d5^2 + d6^2 = **Sum of Squared Distances** = **SS(Distances)**
     * We repeat until we end up with the line with the largest **Sum of Squared Distances** between the projected points and the origin by keeping rotating the line
     * Ultimately, we end up with the largest **SS(Distances)** line = **Principal Component 1 (PC1)**
     * **PC1** has a slope of 0.25
     * In other words, for every 4 units that we go out along the variable 1 axis, we go up 1 unit along the variable 2 axis
     * That means that the data are mostly spread out along with variable 1 axis
     * and only a little bit spread out along with variable 2 axis
     * one way to think about PC1 is in terms of a cocktail recipe
         * to make PC1, Mix 4 parts variable 1 with 1 part variable 2 
     * the ratio of variable 1 to variable 2 tells you that variable 1 is more important when it comes to describing how the data are spread out
* When we do **PCA** with **SVD**, the recipe for PC1 is scaled to **length = 1**
    * For example, if a^2 = 4.12, b^2 = 1, c^2 = 4
    * then, a^2 = 4.12 / 4.12, b^2 = 1 / 4.12 , c^2 = 4 / 4.12
    * 4.12 / 4.12 = square root (4 / 4.12)^2 + (1 / 4.12)^2
    * 1 = 0.242 + 0.97
        * to make PC1, Mix 0.97 parts variable 1 with 0.242 part variable 2 -> same ratio
* For PC2, because this is only a 2-D graph, PC2 is simply the line through the origin that is perpendicular to PC1, without any further optimization that has to be done
    * that means, that the PC2 recipe is -1 parts variable 1, +4 parts variable 2 
    * if we scale,  -0.242 parts variable 1, +0.97 parts variable 2, which is the **Loading Scores** for PC2
    * this is the **Singular Vector** for PC2 or the **Eigenvector** for PC2
    * that tells us that, in terms of how the values are projected onto variable 2, variable 2 is 4 times as important as variable 1
* We can convert the **Eigenvalues** into variation around the origin (0, 0) by dividing by the sample size minus 1 (i.e. n - 1)
    * SS(Distances for PC1) / n - 1 = Variation for PC1 = in this example, PC1 = 15
    * SS(Distances for PC2) / n - 1 = Variation for PC2 = PC2 = 3
    * That means, the total variation = 15 + 3 = 18
    * That maens, PC1 accounts for 15 / 18 = 0.83 = 83% of the total variation around the PCs
    * That maens, PC2 accounts for 3 / 18 = 0.17 = 17% of the total variation around the PCs
    * **Scree Plot** - show the percentage of total variation
* For 3-D graphs (3 or more variables), for 4-D, we can't draw a graph but that doesn't stop us from doing the **PCA** math
   * Start with an awkward 3-D graph
   * Calculate the **Principal Components** for each variable (PC1, PC2, PC3)
       * PC3 will be a perpendicular line, as well as PC4, PC5, PC6, and so on 
   * Then, with the **Eigenvalues** for PC1, and PC2, we can determine that a 2-D graph would still be very informative by looking at the total variation 
   * Lastly, use PC1 and PC2 to draw a 2-D graph
   * **Note**: if the **Scree Plot** where PC3 and PC4 account for a substantial amount of variation, then just using the first 2 PCs would not create a very accurate representation of the data
   * However, even a noisy PCA plot can be used to identify clusters of data

#### Terminology Alert
* **Linear Combination** of variable 1 and variable 2 = cocktail recipe
* **Singular Vector** or the **Eigenvector** = 0.97 for variable 1 and 0.242 for variable 2
* **Loading Scores**: the proportion of each variable
* **Eigenvalue**: thes best fit **Eigenvalue** line for PC1
    * **SS(Distances)** = **Eigenvalue** for PC1
* **Singular Value**: the square root of **Eigenvalue** for PC1 = **Singular Value** for PC1
* **Scree Plot** is a graphical representation of the percentages of variation that each PC accounts for

## PCA: Practical Tip
1. Make sure the variables are on the same scale, and if not scale them, otherwise, we will be biased toward one of the variable
* The standard practice is to divide each variable by its standard deviation
     * If a variable has a wide range, it will have a large standard deviation, and dividing by it will scale the values a lot
     * If a variable has a narrow range, it will have a small standard deviation and scaling will be minimal
2. Make sure the data is centered
* Double check that the PCA program you are using centers the data or center it yourself
3. How many principal components can you expect to find?
* No, the Eigenvalue for PC3 is always 0 as it rotates to match PC1 or PC2, either one will be 0

# Neural Networks
All **Neural Networks** do is fit **fancy squiggles** or **ben shapes** to data
* And like **Decision Trees** and **SVMs**, **Neural Networks** do ine with any relationship among the variables

## Terminology Aler! Anatomy of a Neural Network
* **Neural Networks** consist of **Nodes**, the square boxes
* and connections between **Nodes**, the arrows
* the bent or curved lines inside some of the **Nodes** are called **Activation Functions**, and they make **Neural Networks** flexible and able to fit just about any data
* The numbers along the connections represent parameter values that were estimated when the **Neural Network** was fit to data using a process called **Backpropagation**

## Terminology Aler! Layers
* **Neural Networks** are organized in **Layers**
    * Usually, a **Neural Network** has multiple **Input Nodes** that form an **Input Layers**
    * and usually there are multiple **Output Nodes** that form an **Output Layer**
* **Hidden Layers**: **Layers** of **Nodes** between the **Input** and **Output Layers** are called
    * Part of the *art* of **Neural Networks** is deciding how many **Hidden Layers** to use and how many **Nodes** should be in each one
    * Generally speaking, the more **Layers** and **Nodes**, the more complicated the shape that can be fit to the data
* In the example, we have a single **Input Node** (Does - Input) -> a single **Hidden Layer** with 2 **Nodes** in it -> and a single **Output Node** (Effectiveness - Output)

## Terminology Aler! Activation Functions
* **Activation Functions** are the basic building blocks for fitting squiggles or ben shapes to data
* There are a lots of different **Activation Functions**. Here are three that are commonly used:
1. **ReLU**: **Rectified Linear Unit**
* probably the most commonly used **Activation Function** with large **Neural Networks**
* it is a **Bent Line**, and the bend is at ***x* = 0**
2. **SoftPlus**: modified form of **ReLU Activation Function**
* The big difference is that instead of the line being bent at **0**, we get a nice **Curve**
3. **Sigmoid**: is an **s-shaped squiggle** that is frequently used when people teach  **Neural Networks** but is rarely used in practice

**Activation Functions** just like the mathematical functions you learned when you were a teenager: you plug in an x-axis coordinate, do the math, and the output is a y-axis coordinate
* For example, the **SoftPlus** function is: **SoftPlut**(*x*) = **log**(1 + e^x)
* where **the log()** function is the natural log, or **log base e**, and **e** is **Euler's Number**, which is roughly **2.72**
* So, if we plug in an x-axis value ***x*** = **2.14**
* then the **SoftPlus will tell us the y-axis coordinate is **2.25** because **log**(1 + e^2.14) = **2.25**

## Terminology Aler! Weights and Biases
* In **Neural Networks**, the parameters that we multiple are called **Weights**
* and the parameters we add are called **Biases**
* **Backpropagation**: to optaimize the **Weights** and **Biases**
    * Just like we did for **R^2**, **Linear Regression**, and **Regression Trees**, we can quantify how well the **green squiggle** fits all of the **Training Data** by calculating the **Sum of the Squared Residuals (SSR)**
    * Use **Graident Descent** to quick find the lowest point between **Final Bias**(x-axis) and **SSR** (y-axis)

# Entropy/Information Entropy
**Entropy** quantifies the amount of uncertainty (or surprise) involved in the value of a random variable or the outcome of a random process
* self-information = single event (how much information can be provided within a single event)
    * To quantify, an event that is less likely to happen to provide more information than an event that is more likely to happen 
    * Meaning, probability low, log high
* **Information Entropy** = all events within the system
    * The distribution of how events happen within a system
    * Example: 1/4，1/4，1/4，1/4 of 4 random colors ball within a box
    * entropy = the uncertainty of a random variable or the whole system
    * the higher the entropy = the uncertainty of a random variable or the system
* Each system has a true distribution and our objective is to eliminate uncertainty with the optimal strategy and least effort
    * The cost of doing this will be the entropy

# Cross-Entropy
* **Cross-Entropy** uses another distribution strategy to apply to the true distribution to eliminate uncertainty of a system
* The lowest = the best, which means Q (another) distribution = P (the true, real) distribution of a system

# Relative Entropy
* **Relative Entropy** the difference between the two distributions

# Term Frequency-Inverse Document Frequency (TF-IDF)
Intuitively, it makes sense that a term that appears in only a few documents should have more weight than one that appears in many documents
1. Step 1 - Count term frequency = a word that appears in a documentation
* TF = words appear / total words in a documentation
2. Step 2 - IDF = log (corpus *total number of documentations within a corpus*) / documentation that contains that word + 1)
* If a word is more popular, then IDF close to 0, add 1 to avoid no documentation containing that word 
3. Step 3 - TF * IDF = TF-IDF
* TF-IDF has a positive relationship with word that appears in a word documentation
* But has a negative relationship with the words that appear in a corpus

## TF-IDF Advantage 
* Fast and easy to understand.

## TF-IDF Disadvantage
* Sometimes important words didn't appear much, and the TF-IDF calculation can not identify the exact location of where the word appears.
* Cannot understand the structure of a sentence or a paragraph.

**TF-IDF** pseudo-code
* count the total number of documents in the corpus 
* create a vocabulary
* create a matrix of zeroes of size *num_docs* * *num_words_in_vocab*
* for each word in the vocabulary:
    * tally number of documents in which the word appears
    * compute inverse document frequency
    * store this value
* for each document in the corpus:
    * for each word in the document's bag of words:
        * tally term frequency
    * multiply term frequency by corresponding inverse document frequency
    * store this value at the appropriate location in the matrix
* return the filled-in matrix

## Neural Network
* A **Neural Network** consists of **Nodes** and **connections** between the nodes
    * Input Node
    * Hidden Layers
        * When you build a Neural Network, one of the first things you do is decide how many HIdden Layers you want
        * And how many Nodes go into each Hidden Layer 
    * Output Node 
* **Note:** The numbers along each connection represent parameter values that were estimated when this Neural Network was fit to the data
* These paratmer estimate are analogous to the **slope** and **intercept** values that we solve for when we fit a **straight line** to data
    * Neural Network starts out with unknown parameter values that are estimated when we fit the Neural Network to a dataset using a method called **Backpropagation**
    * Parameters that we multiply are called **weights**
    * Parameters that we added are called **biases**
* The **bent** or **curved lines** are the buidling blocks for fitting a **squiggle** to data
    * Can be reshaped by the parameter values
    * And then added together to get a final squiggle that fits the data

### Activation Functions
* When building a Neural Network, you have to decide which Activation Function or Functions you want to use
* Remember, the Neural Networks stats with identical Activation Functions, for example, CNNs, but not RNNs
1. Sigmoid
    * Takes any number and squashes it between 0 and 1
    * Looks like an S-shaped curve
    * Works well when we need probabilities, like in binary classification (spam vs. not spam)
    * Pros
        * Outputs are **between 0 and 1** (good for probabilities)
        * Smooth and easy to compute
    * Cons
        * Vanishing Gradient Problem – When x is too big or too small, the gradient (derivative) is super tiny, which slows down learning
        * Not zero-centered – Everything is positive, which means updates in gradient descent are not balanced
    * Used In
        * Output layer for **Binary Classification**
2. ReLU (Rectified Linear Unit)
    * If x is positive, keep it!
    * If x is negative, set it to 0
    * Super simple and FAST!
    * Pros
        * Solves the vanishing gradient problem
        * Because gradients don’t shrink to zero for positive values
        * Super efficient and used in almost all deep learning networks
    * Cons:
        * Dying ReLU Problem – If too many values become 0, those neurons just stop learning
    * Used In
        * Most deep learning models (CNNs, Transformers, etc.)
3. SoftPlus
    * Similar to ReLU, but smooths out the transition from negative to positive values
    * Looks like ReLU but curvier
    * Pros
        * Fixes dying ReLU problem – Because it never actually hits zero, meaning neurons don’t stop learning
        * Smooth gradients – Helps with optimization dying ReLU problem – Because it never actually hits zero, meaning neurons don’t stop learning
    * Cons:
        * Computationally slower than ReLU – Because of the logarithm
        * Still suffers from vanishing gradients for very negative values
    * Used In
        * Some deep learning architectures where smooth gradients are preferred
4. Leaky ReLU (Fixing Dying ReLU)
    * Just like ReLU, but instead of zeroing out negatives, it lets a small leak (0.01 x) pass through
    * Keeps neurons from dying
    * Pros
        * Fixes dying ReLU problem!
        * Still super fast and efficient
    * Cons:
        * The leak coefficient (0.01) is manually set, which means it’s not learned
    * Used In
        * Deep networks that had issues with dead neurons
5. ELU (Exponential Linear Unit) - The Smarter ReLU!
    * Instead of just leaking like Leaky ReLU, it uses an exponential function for negative values
    * This gives it stronger negative gradients, which helps training
    * Pros
        * Better than Leaky ReLU – It learns faster and reduces bias
        * Solves vanishing gradient problem better than ReLU
    * Cons:
        * More computation than ReLU (exponential function)
    * Used In
        * CNNs and deep networks where better learning is needed
7. Tanh (Hyperbolic Tangent)
    * Like Sigmoid, but better!
    * Outputs between -1 and 1, meaning it’s zero-centered
    * Pros
        * Zero-centered – Helps with gradient updates
        * Works better than Sigmoid in deep networks
    * Cons:
        * Still suffers from vanishing gradients for large positive or negative inputs
    * Used In
        * RNNs (Recurrent Neural Networks) and time-series models
8. Softmax (For Multi-Class Classification)
    * Converts numbers into probabilities that sum to 1
    * Great for multi-class classification
    * Pros
        * Converts outputs into clear probabilities
        * Good for picking one correct class
    * Cons:
        * Computationally expensive (exponentials)
    * Used In
        * The last layer of classification networks (ImageNet, NLP models)
 
### Backpropagation
* **Note:** *Conceptually*, Backpropagation starts with the last parameter and works its way backwards to estimate all of the other parameters
* Use the **Chain Rule** to find out how much each weight contributes to the **Sum of the Squared Residuals** with respect to biases
* Use **Gradient Descent** to update the weights in the right direction
    * For example, x1 (x-axis value) = Input * weight #1 * bias #1
    * Then plug x1 into the first Activation Function (for example, SoftPlus)
    * Give us a y1 (y-axis value)
    * Then y1 * weight #3, and add the y2 * weight #4, and the bias #3 to get the final line, and the **Predicted** values
    * Lastly, use the **Predicted** values to caluate the **Sum of the Squared Residuals**
* Initialize the **Weights** and **Biases**
* **Step Size** = **Derivative** * **Learning Rate**
* **New Value** = **Old Value** - **Step Size**

## Convolutional Neural Networks (CNNs)
* Do 3 things to make image classification practical
1. Reduce the number of input nodes
2. Tolerate small shifts in where the pixels are in the image
3. Take advantage of the correlations that we observe in complex images
* The first thing a CNN does is apply a **Filter (aka Kernel)** to the **Input Image**
    * A filter is just a smaller square that is commonly 3 pixels by 3 pixels
    * And the intensity of each pixel in the filter is determined by **Backpropagation**
    * In other words, before training a CNN, we start with random pixel values
    * And after training with Backpropagation, we end up with something more useful
    * To apply the Filter to the input image
        * We overlay the Filter onto the image
        * And then we multiply together each overlapping pixel
        * And then add each product together
        * To get a final value (In the example, is 3) = **Dot Product**
    * Add a **Bias** term to the output of the **Filter**
    * Put the final value into something called a **Feature Map** (Filled up the Feature Map)
    * Then apply another filter to the new Feature Map
        * Unlike before, we simply select the **maximum value** = Max Pooling
        * And this filter usually moves in such a way that it does not overlap itself
    * **Max Pooling** = select the maximum value in each region
        * Selected the spot where the **Filter did the best job matching the input image
    * Alternatively, we could calculate the average value for each region and that would be called **Average** or **Mean Pooling** 
* By computing the **Dot Product** between the input and the **Filter**, we can say that the **Filter** is **Convolved** with the input, and that is waht gives **Convolutional Neural Networks their name
* Because each cell in the **Feature Map** correspodngs to a group of neighboring pixels
* The Feature Map helps take advantage of any correlations there might be in the image
    * Typically, we run the Feature Map through a **ReLU** Activation Function
        * That means that all of the negative values are set to **0**, and the positive values are the same as before
* Summary- No matter how fancy the Convolutional Neural Network is, it is still based on...
1. Filters, aka **Convolution**
2. Applying an **Activation Function** to the filter output
3. Pooling the output of the **Activation Function**

## Recurrent Neural Networks (RNNs)
* Just like other neural networks that we have seen before, Recurrent Neural Networks have **weights**, **biases**, **layers**, and **activation functions**
* The big difference is that RNNs also have **feedback loops**
    * And, although this neural network may look like it only takes a single input value, the **feedback loop** makes it possible to use *sequential* input values, like stock market prices collected over time, to make predictions
    * Instead of having to remember which value is in the loop, and which value is in the input
        * We can **unroll** the feedback loop by making a copy of the neural network for each input value
* **Note:** Regardless of how many times we **unroll** a recurrent neural network, the **weights** and **biases** are shared across every input
    * In other words, even thoug hthis **unrolled** network for example has 3 inputs
    * The **weight**, weight #1, is the same for all 3 inputs
    * And the **bias**, bias #1, is also the same for all 3 inputs
    * Likewise, all of the other **weights** and **biases** are shared
    * So, no matter how many times we **unroll** a recurrent neural network, we never increase the number of **weights** and **biases** that we have to train
* One big problem is that the more we unroll a recurrent neural network, the harder it is to train
* This problem is called **The Vanishing/Exploding Graident Problem**
    * When **Gradient** contains **A Huge Number**, we will end up taking relatively large steps
    * And instead of finding the optimal parameter, we will just bounce around a lot
    * Input * 2^50 (2 = weight) = Input * A Huge Number
    * One way to prevent **The Explording Gradient Problem** would be limited the weight to values **< 1**
        * However, this results in **The Vanishing Gradient Problem**
* Vanishing Graident Problem = **super close to 0**
    *  We end up taking steps that are too small
    *  Input * 0.5^50 (0.5 = weight) = Input * a number super close to **0**

## Long Short-Term Memory (LSTM)
* Use two separate paths to make predictions about future value
    * One path is for **Long-Term Memories** = **Cell State**
        * No **Weights** and **Biases** that can modify it directly
        * Without causing the graident to **explode** or **vanish**
    * And one is for **Short-Term Memories** = **Hidden State** 
        * Connected to **Weights** that can modify them
* LSTM uses **Sigmoid** (turn x-axis to y-axis between 0 and 1) Activation Functions and **Tanh** (turn x-axis to y-axis between -1 and 1) Activation Functions
1. The first in a LSTM unit determines what percentage of the Long-Term Memory is remembered == **Forget Gate**
2. The second stage = **Input Gate**
*  Right Block = Short-Term Memory and the Input combines, and to create a **Potential Long-Term Memory**
*  Left Block = determines what percentage of that Potential Memory to add to Long-Term Memory
3. The final stage updates the Short-Term Memory = **Output Gate**
* Start with the New Long-Term Memory and use it as input to the Tanh Activation Function to create **Potential Short-Term Memory**
* Decide how much of Potential Short-Term Memory to pass on
* LSTM resues the exact same **Weights** and **Biases** is so it can handle data sequences of different lengths

## Word Embedding and World2Vec
* First, rather than just assign random number to words, we can train a relatively simple Neural Network to assign numbers for us
* The advantage of using a Neural Network is that it can use the contexts of words in the training dataset to optimize **Weights** that can be used for embedding
* And this can result in similar words ending up with similar embeddings
* Lastly, having similar words with similar embeddings means training a Neural Network to process language is easier
* Because learning how one word is used helps learn how similar words are used
* Reuse the same **Word Embedding** network for each input word or symbol
    * This mean that regardless of how long the input sentence is, we just copy and use the exact same Word Embedding network for each word or symbol = same Weights
    * Each Weight starts out as a random number
    * But when we **Train** the **Transformer** with English phrases and known Spanish translations
    * **Backpropagation** optimizes these values one-step-at-a time and results in these infal **Weights**
    * The process of optimizing the **Weights** = **Training**

#### Word2Vec
* A popular method for creating Word Embeddings (capture meaning and relationships between words), uses to include more context
* Two Strategies in Word2Vec
1. **Continuous Bag of Words** = increases the context by using the surrounding words to predict what occurs in the middle
* For example, Troll 2 and great! to predict word that occurs between them is
2. **Skip Gram** = increases the context by using the word in the middle to predict the surrounding words
* For example, the Skip Gram method could use the word is to predict the surrounding words, Troll 2, great! and Gymkata
* Lastly, people often use **100** ore more activation functions to create a lot of **Embeddings per word
    * For example, 3,000,000 words and phrases times at least 100, the number of **Weights** each word has going to the activation functions
    * Times 2, for the Weights that get us from the activation functions to the outputs
    * For a total of 600,000,000 Weights = Training can be slow
3. Word2Vec speeds things up is use something called **Negative Sampling**
* Negative Smapling works by randomly selecting a subset of words *we don't* want to *predict* for optimization
* For example, say like we wanted the word aardvark to predict the word A
    * That means that only the word aardvark has a 1 in it, and all the other words have 0s
    * And that means we can ignore the  **Weights** coming from every word but aardvark, because the other words multiply their Weights by 0
    * That alone removes close to 300,000,000 Weights from this optimization step
* **Note:** In practice, Word2Vec would select between 2 and 20 words that *we don't* want to *predict*
* For example, only uses the output values for A and abandon
* Tha means for this round of Backpropagation, we can ignore the Weights that lead to the all of the other possible outputs
* So, in the end, out of the 600,000,000 total Weights in this Neural Network, we only optimize 300 per step

## Seq2Seq and Encoder-Decoder
* Use an **Embedding Layer** to convert the words into numbers
* Because the vocabulary contains a mix of words and symbols, we refer to the individual elements in a vocabulary as **Tokens**
* In essence, the **Encoder** encodes the input sentence, "Let's go", into a collection of long and short tem memories (cell and hidden states)
* The last long and short term memories (the cell and hidden states) from both layers of the LSTM cells in the **Encoder** are called the **Context Vector**
* To decode the Context Vector, the first thing we do is connect the long and short term memories (the cell and hidden states) that form the Context Vector to a new set of LSTMs
* The ultimate goal of the **Decoder** is to decode the **Context Vector** into the output sentence
* Just like in the **Encoder**, the first layer comes from an **Embedding Layer**
    * However, now the **Embedding Layer** creates embedding values for the Spanish words, ir, vamos, and y, and the <EOS> (End of Sentence) symbol
* To summarize, the **Decoder** stage
    * The Context Vector, created by Encoder's unrolled LSTM cells, are used to initialize the LSTMs in the Decoder
    * And the input to the LSTMs comes from the output Word Embedding layer that starts with <EOS>
    * but subsequently uses whatever word was predicted by the Output Layer
    * In practice, the Decoder will keep predicting words until it predicts the <EOS> token, or it hits some maximum output length
* **Encoder-Decoder** models have two special things that happen during training
    * When Training an Encoder-Decoder, instead of using the predicted token as input to the Decoder LSTMs, we use the known, correct token
    * In other words, if the first predicted token was the Spanish word y, which translates to and in English, and thus, is the wrong word
    * Then, during Training, we will still uses vamos, the correct Spanish word, as input to the unrolled LSTMs
    * Also, during Training, instead of just predicting tokens until the Decoder predcits the <EOS> token
    * Each output phrase ends where the known phase ends
    * In other words, even if the second predicted token was the Spanish word ir, instead of the correct token <EOS>
    * Then, during Training, we will still stop predicting additional tokens
* Plugging in the known words and stopping at the known phrase length, rather than using the predicted tokens for everything, is called ***Teacher Forcing**

## Attention
* Main idea of **Attention** is to add a bunch of new paths from the **Encoder** to the **Decoder**, one per input value, so that each step of the **Decoder** can directly access input values
* An **Encoder-Decoder Model** can be as simple as an **Embedding Layer** atached to a single **Long Short-Term Memory** unit
* But if we want a slightly more fancy **Encoder**, we can add additional LSTM cells
1. So, in the example, the first thing that Attention does is determine how similar the outputs from the Encoder LSTMS are at each step to the outputs from the Decoder LSTMs
    * In other words, we want a similarity score between the LSTM outputs (the short term memories or hidden states) from the *first step* in the Encoder
    * And the LSTM outputs from the first step in the Decoder
   * And we also want to calculate a similarity score between the LSTM outputs from the *second step* in the Encoder
   * And the LSTM outputs from the first step in the Decoder
* There are a lot of ways to calculate the similarity of words, or, more precisely, sequences of numbers that represent words
* And different Attention algorithms use different ways to compare these sequences
* One simple way to determine the similarity of two sequences of numbers that represent words is with the **Cosine Similarity**
    * The numerator is waht calculates the similarity between two sequences of numbers
    * And the denominator scales that value to be between -1 and 1 (vs Dot Product not being scaled, Magnitude + Direction)
* We can just calculating the numerator, which is also called the **Dot Product**
    * Calculating the Dot Product is more common than the Cosine Similarity for Attention because
    * Super easy to calculate and roughly speaking
        * Large positive numbers mean things are more similar than small positive numbers
        * And large negative numbers mean things are more completely backwards than small negative numbers
* Summary
    * When we add **Attention** to a basic **Encoder-Decoder Model**
    * The **Encoder** pretty much stays the same
    * But now, each step of decoding has access to the individual encodings for each input word
    * And we use similartiy scores and the **SoftMax** function to determine what percentage of each encoded input word should be used to help predict the next output word
    * **Note:** Now that we have **Attention** added to the model, you might wonder if we still need the **LSTMs**
    * Well, it turns out that we don't need them = **Transformers**

### Cosine Similarity
* Cosine Similarity is to cacluate metric that can tell us how similar or different things are
* Cosine Similarity is determined entirely by the **angle** between lines, and **not** by the **lengths** of the lines
* When both phrases are exactly the same, the angle between them will be **0 degree = 1 = Cosine Similarity**
* In contrast, if the phrases don't have any words in common, then the angle between the two phrases will be **90 degree = 0 = Cosine Similarity**
* Summary
    * When two phrases have absolutely nothing in common, the Cosine Similarity is **0**
    * And when phrases are the exact same, the Cosine Similarity is **1**
    * And when there is some overlap between the two phrases, but they are not exactly the same, the Cosine Similarity is between 0 and 1
* Way we have been computing the Cosine Similarity
1. Step 1 - Make a table of the word counts
2. Plot the points
3. Figure out the angle
4. Calculate the **Cosine** of the angle

## Transformer
* **Positional Encoding**, which is a technique that **Transformer** use to keep track of word order
1. Convert the words, "Squatch eats pizza" into numbers using **Word Embedding**
* In this example, we are creating 4 Word Embedding values, per word
    * However, in practice, people often create hundreds or even thousands of embedding values per word
2. We add a set of numbers that correspond to word order to the embedding values for each word
* In this case, the numbers that represent the word order come from a sequence of alternating **Sine** and **Cosine** squiggles
* Each squiggle gives us specific position values for each word's embeddings
* **Note:** Because the **Sine** and **Cosine** squiggles are repetitive, it is possible that two words might get the same position, or y-axis, values
    * However, because the squiggles get wider for larger mebedding positions
    * And the more embedding values we have, then the wider the squiggles get
    * Then, even with a repeat value here and there, we end up with a unique sequence of position values for each word
    * Thus, each input word ends up with a unique sequence of position values
3. Now all we have to do is add the position values to the embedding values
* And we end up with word embeddings plus **Positional Encoding** for the whole sentence
* **Note:** If we reverse the order of the input words to be "Pizza eats Squatch"
    * Then the embeddings for the first and third words get swapped
    * But the positional values for 1st, 2nnd, and 3rd word **stay the same**
    * And when we add the positional values to the embedding, we end up with new **Positional Encoding** for the 1st and 3rd words
* Thus, **Position Encoding** allows a **Transformer** to keep track of word order

##### How a Transformer keeps track of the relationship among words
* For example, "The pizza came out of the oven and it tasted good!
    * It, could refer to pizza
    * Or, potentially, it could refer to the word oven
    * **Self-Attention**, which is a mechanism to correctly associate the word it with the word pizza

## Self-Attention
* In general, terms, **Self-Attention** works by seeing how similar each word is to all of the words in the sentence, including itself
* For example, **Self-Attention** calculates the similarity between the first word, The, and all of the words in the sentence
* And **Self-Attention** calculates these similarities for every word in the sentence
* Once the similarities are calculated, they are used to determine how the **Transformer** encodes each word
    * For example, if you looked at a lot of sentences about pizza and the word it was more commonly associated with pizza than oven
    * Then the similarity score for pizza will cause it to have a larger impact on how the word it is encoded by the **Transformer**
4. Later, we multiply the position encoded values with **Weights**
* We have 2 new values to represent the word, for example, Lets, and in Transformer terminology, we called them **Query** values
* And now that we have **Query** values for the word, Let's, we use them to calculate the similarity between itself and the word, go
5. Continues to create two values for itself and the word, go
* Both sets of new values are called **Key** values
    * And we use them to calculate similarities with the **Query** for Let's
    * One way to calcualte similarities between the **Query** and the **Keys** is to pcalculate something called a **Dot Product**
    * Since Let's is much more similar to itself than it is to the word go
    * Then we want Let's to have more influence on its encoding than the word go
* And we do this by first running the similarity scores through something called a **SoftMax** function
    * The main idea of a **SoftMax** function is that it preserves the order of the input values, from low to high, and translates them into numbers between 0 and 1 that add up to 1
    * So we can think of the output of the **SoftMax** function as a way to determine what percentage of each input word we should use to encode the word Let's
        * In this case, because Let's is so much more similar to itself than the word go, we will use **100%** of the word Let's to encode Let's
        * And **0%** of the word go to encode the word Let's
6. We create 2 more values, that we will called **Values**, to represent the word Let's
* Scale the **Values** that represent Let's by 1.0
* And continues to create 2 **Values** to represent the word go, and scale those **Values** to 0.0
7. Lastly, we add the scaled values together
* And these sums, which combine separate encodings for both input words, Let's and go, relative to their similarity to Let's, are the Self-Attention values for Let's
8. Next, to calculate Self-Attention values for the word go
* We do not need to re-calcualte the **Keys** and **Values**
* Instead, all we need to do is create the **Query** that represents the word go
9. To encode the input, we take the **Position Encoded** values and add them to the **Self-Attention** values**
* These bypasses are called **Residual Connections, and they make it easier to train complex neural networks
* By allowing the **Self-Attention layer to establish relationship among the input words without having to also preserve the **Word Embedding** and **Position Encoding** information

**Transformer Encoder Summary**
1. Word Embedding (values) - Encode words into numbers
2. Poistional Encoding (values) - Encode the positions of the words
3. Self-Attention (values) - Encode the relationships among the words
4. Residual Connections (values) - Relatively easily and quickly train in parallel

**Note** 
1. First, the **Weights** that we use to calculate the **Self-Attention Queries** are exact same for both Let's and go
* In other words, this example uses one set of **Weights** for calculating **Self-Attention Queries, regardless of how many words are in the input
* Likewise, we reuse the sets of **Weights** for calculating **Self-Attention Keys** and **Values** for each input word
* This means that no matter how many words are input into the **Transformer**, we just reuse the same sets of **Weights** for **Self-Attention Queries**, **Keys**, and **Values**
2. We can calculate **Queries**, **Keys**, and **Values** for each word at the same time

**Transformer Decoder**
1. Start with **Word Embedding**, output vocabulary, which consists of the Spanish words
2. Then **Position Encoding** with exact same **Sine** and **Cosine** squiggles that we used when we encoded the input
3. Calculate the **Query**, **Key**, and **Value** (Weight are different from the sets we used in the **Encoder**) and calculate its **Self-Attention** values
4. Add **Residual Connections**

**Encoder-Decoder Attention**
* To keep track of things between the input and output phrases to make sure important words in the input are not lost in the translation
    * **Query** in Decoder
    * **Keys** in Encoder
    * Calcualte **Values** for each input word
    * Scale those **Values** by the **SoftMax** percentages
    * Move to a **Fully Connected Layer** returns the prediction (with **SoftMax**)
 
## Decoder-Only Transformers (ChatGPT)
* **Masked Self-Attention** (values) works by seeing how similar each word is to *itself* and all of the *preceding* words in the sentence
    * Applied equally to the input prompt
    * And to the output that is generated
* **Masked Self-Attention** allows a **Decoder-Only Transformer** to determine how words in the prompt are related
* And make sure that it keeps track of important input words when generating the output
* **Note:** When are **Training** a normal **Transformer**, the **Masked Self-Attention** only includes the output tokens
* In contrast, a **Decoder-Only Transformer** uses **Masked Self-Attention** all of the time, not just during training, and it includes the input and the output

**Summary:** The 3 big difference between a normal **Transformer** and a **Decoder-Only Transformer** are:
1. A normal **Transformer** uses one unit to encode the input, called the **Encoder**, and a separate unit to generate the output, called the **Decoder**
2. And a normal **Transformer** uses two types of **Attention** during inference
* **Self-Attention**
* **Encoder-Decoder Attention**
3. Lastly, during **Training**, a normal **Transformer** uses **Masked Self-Attention**, but only on the output

**In contrast**
1. A **Decoder-Only Transformer** has a single unit for both encoding the input and generating the output
2. And a **Decoder-Only Transformer** uses a single type of attention, **Masked Self-Attention**
3. And a **Decoder-Only Transformer** uses **Masked Self-Attention** all the time on everything, the input and the output

## Encoder-Only Transformers (like BERT) for RAG
* Combining **Word Embedding**, **Positional Encoding**, **Self-Attention** into new type of embeding is sometiems called **Context Aware Embedding** or **Contextualized Embedding**
* **Encoder-Only Transformers**, like (BERT), that only use **Self-Attention**
* **Context Aware Embedding** can help cluster similar sentences, or similar documents
    * Just like plain old **World Embedding** can help cluster similar words
* **Note:** The ability to cluster similar sentences and documents is the foundation for something called **Retrieval-Augmented Generation**, or **RAG**
* **RAG** works by breaking a document into blocks of text and then using an **Encoder-Only Transformer** to create **Context Aware Embeddings** for each one
    * For example, when someone gives an AI a prompt, like What is Pizza?
    * **RAG generates embeddings for What is Pizza? And finds the chunks of text that are the most similar
