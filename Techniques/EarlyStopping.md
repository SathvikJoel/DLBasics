# Early Stopping 

[7.8 in Deeplearning book](https://www.deeplearningbook.org/contents/regularization.html)

[Early Stopping implementation in Pytorch](https://github.com/Bjarten/early-stopping-pytorch)

## When to use

When training large models with suﬃcient representational capacity to overﬁt the task

## Advantages

* Simple to implement
* No major change need to be done to the already existing training loop
* "trainng time" Hyper parameter, differnt from conventional Hyper paramters. So it is advantageous to use early stopping

## Disadvantages

* Maynot always work
* Doesnt use all the training data unless you extend the algorithm for a second round of training, which introduces subtilities


## Tricks

Refer to the trick to extend Early stopping to use all the training data, the book suggest two stratagies for doing that. ( Not useful when the dataset is large )

## Mathematical Explanation

The book gives mathematical intuition as to how early stopping produces a similar effect to L2 regularization. Hence, early stopping is a regularization technique.





