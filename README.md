# Meetup Venue Recommender

### Business Objective
The goal of this project is to build a **venue recommender** for *meetup group orgnizers*. On the one hand, this app could help *organizers* explore wonderful *venue* options that were liked by other fellows who have hosted similar events. On the other hand, this app could also serve as great advertising resorces for business *venues* and assist them to target potential clients. 

### Dataset
The dataset was fetched from the [*Meetup API*](https://www.meetup.com/meetup_api/) utilizing a [python API client](https://pypi.org/project/meetup-api/). In particular, I first identified the Meetup groups with more than 100 members in the top 30 largest [US cities](https://en.wikipedia.org/wiki/List_of_United_States_cities_by_population) and then gathered the meetup details associated with these groups between 2010 and 2018. 

The fetched raw dataset is around 20 GB and includes 100000 meetup groups with 0000 event details. 

### Challenges

* **Data Sparsity**  Similar to most recommendation problems, the meetup **Group x Venue Interation** matrix is extremely sparse, which means that usually each meetup group only hosted events in a very limited number of venues.


* **Demographic Restraint**: Different from many recommendation systems (e.g. movie/music recommendation, onlin purchase recommendation), demogrphic constraints are critical for a *venue recommender*, since it makes no sense to recommend a San Fransico Venue to a New York based Organizer(unless he/she is travelling). 


* **Cold Start**: Because this app is aimed to help meetup group orgnizers at all levels of experience (novice or expert), 

### Solutions

To tackle these challenges, I will buld a **hybrid recommender system** which combined *content*, *demographic* and *collaborative filtering*.


* **demographic filtering** all recommendations were limited to the same city of the query.
* **content filtering** documents of meetup group descriptions were used to measure the similarity among groups, and then venues that have hosted events for similar groups were recommended. In particular, a Recurrent Neuralnet work was first trained to classify meetup group categories, the output of the dense layer was used as the feature for measure group similarity.

* **collaborative filtering** (ongoing) collaborative filtering models with alternating least squares (ALS) optimization were trained. The interaction matrix is between meetup groups and venues, with the number of hosted events as implicit feedback.
