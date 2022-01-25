"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    [The agent only cares about the best reward, thus always tries to move east]
    """

    answerDiscount = 0
    answerNoise = 0.2

    return answerDiscount, answerNoise

def question3a():
    """
    [Changed living reward to -2 so it forces the agent to choose the closer exit]
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = -2

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    [lowered discount factor so the agent cares about the future state less
    Thus going for the longer path. Changed living reward to negative so it
    chooses the closer 1 over the 10]
    """

    answerDiscount = 0.5
    answerNoise = 0.2
    answerLivingReward = -1

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    [Changed the living reward to -1, so it chooses the riskier path, but
    low enough so it chooses the 10 over 1]
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = -1

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    [No need to change the default values]
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    [High enough living rewards so it doesn't want to exit]
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 2

    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    [Not possible since if the episolon is high, it will explore to new areas
    but won't be able to reach an optimal path within 50 iterations. If episilon
    is low, the agent won't explore enough and will be even slower than higher epsilon]
    """

    return NOT_POSSIBLE

if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))
