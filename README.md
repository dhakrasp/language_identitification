Dear candidate, 

Congratulations for having passed to the challenge stage in our hiring process!

This is a short 48-hour challenge to understand how you work, your thought process and metodology. 
You do not need to be familiar with the described problem to be able to do this challenge. 
In fact, first and foremost, we are going to focus on your reasoning and the steps you took.

Don't forget to plan your time accordingly, and good luck! :-)

-- The Unbabel Team


## Description

At Unbabel, we deal with many types of multilingual content at all time. Thus, the first thing we 
need to be able to do is to identify these languages properly.

We need you to build a service that provided with a text, identifies the language in which it is 
written, and provides that answer.
We provide you with the initial repo to implement this, and some requirements that the service 
should satisfy. 

Whether you choose to implement an existing approach or compile one, make sure you document it
and explain your reasoning.

### Tasks

1 - Implement a Language Identification service that returns the language code of the language in which the text is written. The provided data and test will
target Spanish (ES), Portuguese (PT-PT) and English (EN)

2 - Train the system to distinguish between language variants. In this case we wish to distinguish between European Portuguese (PT-PT) and Brazilian Portuguese (PT-BR)

3 - Implement a deep learning model (recommended: a BILSTM tagger) to detect code switching (language mixture) and return both a list of tokens and a list with one language label per token.
To simplify we are going to focus on English and Spanish, so you only need to return for each token either 'en', 'es' or 'other'

*See more information about tasks 1 and 2 in langid folder, and about task3 in code_switching folder*

## Evaluation

We will evaluate the system on our own test sets, by querying your system for each test case. We will use different datasets for each task.

## Guidelines
* ***Fork this _git repo_*** and add relevant files and explanations for what you feel necessary (starting with a clear description of what is implemented in the README)
* You can program in your prefered language, but we suggest you code in **Python**, since it is our primary language at unbabel
* Make sure you ***document everything*** and assume nothing.
* Don't forget to keep your code clean and commit regularly.
* Please avoid using existing language identification libraries for the final solution.
* ***Send us a link to your fork*** as soon as you start working on it, and then let us know when you're done. Be sure to set our role as `reporter` and not `guest`.
* If you can't finish the challenge due to unforeseen personal reasons let us know ASAP so we can adapt your deadline and/or challenge.
* Any challenge related questions, technical or otherwise, feel free to contact us: `ai-challenge@unbabel.com`.

