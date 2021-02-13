# classimb_fairness

Main idea of this project is to combine class imbalance + fairness handling strategies. Our goal is to learn a model which has a combined goal (LDA-RW), particularly with group+class specific weights.  

Data augmentation is one such approach which people have used to handle both (many papers have used it, both for handline data scarcity as well for debiasing datasets). We can build upon data augmentation solution. TODO: Find arguments why data augmentation is not sufficient, mainly because augmentation techniques assume availability of several resources, which may not be available for low-resource settings.

Baselines: 
-- class imbalance approaches which do not consider fairness aspects
-- Debiasing techniques which do not consider class imbalance
-- Data augmentation technique which considers both (still we will have empirical novelty, to evaluate them exactly for this purpose)


