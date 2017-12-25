# Structured

Task Description

SemEval-2013: Sentiment Analysis in Twitter

- Please register: http://www.cs.york.ac.uk/semeval-2013/index.php?id=registration

- Join the Google group: semevaltweet-2013@googlegroups.com

- Full training, development and test datasets, and a scorer/format checker, have been released.

- Having problems downloading the data? Try the latest version of the Python script.

 

Abstract:

 
In the past decade, new forms of communication, such as microblogging and text messaging have emerged and become ubiquitous. While there is no limit to the range of information conveyed by tweets and texts, often these short messages are used to share opinions and sentiments that people have about what is going on in the world around them.  We propose this task and the development of a twitter sentiment corpus to promote research that will lead to a better understanding of how sentiment is conveyed in tweets and texts. There will be two sub-tasks: an expression-level task and a message-level task; participants may choose to participate in either or both tasks.
 
- Task A: Contextual Polarity Disambiguation
Given a message containing a marked instance of a word or phrase, determine whether that instance is positive, negative or neutral in that context.
 
- Task B: Message Polarity Classification
Given a message, classify whether the message is of positive, negative, or neutral sentiment. For messages conveying both a positive and negative sentiment, whichever is the stronger sentiment should be chosen.


I. Introduction and Motivation

In the past decade, new forms of communication, such as microblogging and text messaging have emerged and become ubiquitous.  While there is no limit to the range of information conveyed by tweets and texts, often these short messages are used to share opinions and sentiments that people have about what is going on in the world around them.

Working with these informal text genres presents challenges for natural language processing beyond those typically encountered when working with more traditional text genres, such as newswire data.  Tweets and texts are short: a sentence or a headline rather than a document.  The language used is very informal, with creative spelling and punctuation, misspellings, slang, new words, URLs, and genre-specific terminology and abbreviations, such as, RT for “re-tweet” and #hashtags, which are a type of tagging for Twitter messages.  How to handle such challenges so as to automatically mine and understand the opinions and sentiments that people are communicating has only very recently been the subject of research (Jansen et al., 2009; Barbosa and Feng, 2010; Bifet and Frank, 2010; Davidov et al., 2010; O’Connor et al., 2010; Pak and Paroubek, 2010; Tumasjen et al., 2010; Kouloumpis et al., 2011).

Another aspect of social media data such as Twitter messages is that it includes rich structured information about the individuals involved in the communication. For example, Twitter maintains information of who follows whom and re-tweets and tags inside of tweets provide discourse information. Modeling such structured information is important because: (i) it can lead to more accurate tools for extracting semantic information, and (ii) because it provides means for empirically studying properties of social interactions (e.g., we can study properties of persuasive language or what properties are associated with influential users).

We believe that a freely available, annotated corpus that can be used as a common testbed is needed in order to promote research that will lead to a better understanding of how sentiment is conveyed in tweets and texts.  Our primary goal in this task is to create such a resource: a corpus of tweets and texts with sentiment expressions marked with their contextual polarity and message-level polarity.  The few corpora with detailed opinion and sentiment annotation that have been made freely available, e.g., the MPQA corpus (Wiebe et al., 2005) of newswire data, have proved to be valuable resources for learning about the language of sentiment.  While a few twitter sentiment datasets have been created, they are either small and proprietary, such as the i-sieve corpus (Kouloumpis et al., 2011), or they rely on noisy labels obtained from emoticons or hashtags.  Furthermore, no twitter or text corpus with expression-level sentiment annotations has been made available so far.


II. Task Description

The task has two sub-tasks: an expression-level task and a message-level task.  Participants may choose to participate in either or both tasks.

Task A: Contextual Polarity Disambiguation
Given a message containing a marked instance of a word or a phrase, determine whether that instance is positive, negative or neutral in that context. The boundaries for the marked instance will be provided, i.e., this  is a classification task, NOT an entity recognition task.

Task B: Message Polarity Classification
Given a message, decide whether the message is of positive, negative, or neutral sentiment. For messages conveying both a positive and negative sentiment, whichever is the stronger sentiment should be chosen.


III. Data

We will create a corpus of 12-20K messages on a range of topics.  Topics will include a mixture of entities (e.g., Gadafi, Steve Jobs), products (e.g., kindle, android phone), and events (e.g., Japan earthquake, NHL playoffs).  Keywords and twitter hashtags will be used to identify messages relevant to the selected topic.

The message corpus will then be divided as follows:

trial data: 2000 twitter messages -- released

training data: 8000-12,000 twitter messages (includes the trial dataset) -- released

development data: 2,000 twitter messages (can be used for training too) -- released

test data #1: 2000-4000 twitter messages -- released

test data #2: 2000-4000 SMS messages -- released

All tweet datasets are sampled and annotated in the same way.

 

The development dataset is intended to be used as a development-time evaluation dataset as the participants develop their systems. However, the participants are free to use the dataset in any way they like, e.g., they can add it to their training dataset as well.

Participants should note that there will be two test datasets, one composed of twitter messages and another one composed of message data for which they would not be receiving explicit training data.  The purpose of having a separate test set of SMS messages is to see how well systems trained on twitter data will generalize to other types of message data.

The released datasets contain "objective" labels, which the participants are free to use on training as they wish. However, we recommend that for task A these labels be ignored since there will be no "objective" labels in the testing dataset. For task B, "objective" and "neutral" labels should be merged into "neutral"; the two labels will be also merged likewise for the test dataset. So, at test time, for both task A and task B, the systems will have to predict just three labels: positive, negative and neutral. However, while for task A neutral means just "neutral", for task B, neutral means "neutral OR objective".

 

IV. Evaluation

Each participating team will initially have access to the training data only.  Later, the unlabelled test data will be released.  After SemEval-3, the labels for the test data will be released as well.

The metric for evaluating the participants’ systems will be average F-measure (averaged F-positive and F-negative, and ignoring F-neutral; note that this does not make the task binary!), as well as F-measure for each class (positive, negative, neutral), which can be illuminating when comparing the performance of different systems.  We will ask the participants to submit their predictions, and the organizers will calculate the results for each participant. For each sub-task, systems will be ranked based on their average F-measure.  Separate rankings for each test dataset will be produced.

For each task and for each test dataset, each team may submit two runs: (1) Constrained - using the provided training data only; other resources, such as lexicons are allowed; however, it is not allowed to use additional tweets/SMS messages or additional sentences with sentiment annotations; and (2) Unconstrained - using additional data for training, e.g., additional tweets/SMS messages or additional sentences annotated for sentiment.

Teams will be asked to report what resources they have used for each run submitted.

 

V. Schedule

August 1, 2012   Trial data has been released
September 12, 2012  First Call for participation
January 10, 2013  Training Data (batch 1) released
February 15, 2013   Registration Deadline for Task Participants: http://www.cs.york.ac.uk/semeval-2013/index.php?id=registration
February 19, 2013  Full Training Data released

February 28, 2013  Development Data released (can be used also for training)

March 6, 2013 *Test* data, reformatted dev data, scorer and format checker have been released to the FTP server
March 15, 2013   Participants predictions due (end of evaluation period)
April 9, 2013   Paper submission deadline
April 29, 2013   Camera ready for system description papers due
Summer 2013   Workshop co-located with NAACL


VI. Organizers

Theresa Wilson   Johns Hopkins University, HLTCOE
Zornitsa Kozareva  University of Southern California, ISI
Preslav Nakov  Qatar Computing Research Institute, Qatar Foundation
Sara Rosenthal  Columbia University
Veselin Stoyanov Johns Hopkins University 
Alan Ritter  University of Washington


VII. References

Barbosa, L. and Feng, J. 2010. Robust sentiment detection on twitter from biased and noisy data.  Proceedings of Coling.
Bifet, A. and Frank, E. 2010. Sentiment knowledge discovery in twitter streaming data.  Proceedings of 14th International Conference on Discovery Science.
Davidov, D., Tsur, O., and Rappoport, A. 2010.  Enhanced sentiment learning using twitter hashtags and smileys.  Proceedings of Coling.
Jansen, B.J., Zhang, M., Sobel, K., and Chowdury, A. 2009.  Twitter power: Tweets as electronic word of mouth.  Journal of the American Society for Information Science and Technology 60(11):2169-2188.
Kouloumpis, E., Wilson, T., and Moore, J. 2011. Twitter Sentiment Analysis: The Good the Bad and the OMG! Proceedings of ICWSM.
O’Connor, B., Balasubramanyan, R., Routledge, B., and Smith, N. 2010.  From tweets to polls: Linking text sentiment to public opinion time series.  Proceedings of ICWSM.
Pak, A. and Paroubek, P. 2010.  Twitter as a corpus for sentiment analysis and opinion mining.  Proceedings of LREC.
Tumasjan, A., Sprenger, T.O., Sandner, P., and Welpe, I. 2010.  Predicting elections with twitter: What 140 characters reveal about political sentiment.  Proceedings of ICWSM.
Janyce Wiebe, Theresa Wilson and Claire Cardie (2005). Annotating expressions of opinions and emotions in language. Language Resources and Evaluation, volume 39, issue 2-3, pp. 165-210.


VIII. Contact Person

Theresa Wilson, Research Scientist
Human Language Technology Center of Excellence
Johns Hopkins University
810 Stieff Building
Baltimore, MD 21211

email: taw@jhu.edu
phone: 410-516-8244
