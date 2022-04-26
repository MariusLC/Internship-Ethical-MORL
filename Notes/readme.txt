README : 

token git : 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


moral.ppo_train.py
python3 -m moral.ppo_train --lambd 0 0 1 1 --env 3
Paramètres : 
	+ --lambd : poids de l'expert sur les objectifs (livraisons, humains sauvés, tuiles nettoyées et vases cassés, pour delivery)
	+ --env : environnement d'apprentissage (2 = emergency, 3 = delivery)
Utilité : apprendre à des agents ppo dont on fixe les poids manuellement. (on apprend aux experts dont on va prendre les trajectoires pour l'AIRL)


utils.generate_demos_main.py
python3 -m utils.generate_demos_main --lambd 0 0 1 1 -env 3 --nbdemos 1000
Paramètres : 
	+ --lambd : poids de l'expert sur les objectifs (livraisons, humains sauvés, tuiles nettoyées et vases cassés, pour delivery)
	+ --env : environnement d'apprentissage (2 = emergency, 3 = delivery)
	+ --nbdemos : taille du set de trajectoires que l'on veut
Utilité : générer des trajectoires d'agents entrainés pour avoir un set de trajectoires leur correspondant (avant d'appliquer l'AIRL)


moral.airl_train.py
python3 -m moral.airl_train --lambd 0 0 1 1 --env 3
Paramètres : 
	+ --lambd : poids de l'expert sur les objectifs (livraisons, humains sauvés, tuiles nettoyées et vases cassés, pour delivery)
	+ --env : environnement d'apprentissage (2 = emergency, 3 = delivery)
Utilité : estimer les poids des experts à partir de leurs set de trajectoires respectif (avant d'appliquer le MORL)


moral.moral_train.py
python3 -m moral.moral_train --lambd 0 0 1 1 --env 3
Paramètres : 
	+ --lambd : poids de l'expert sur les objectifs (livraisons, humains sauvés, tuiles nettoyées et vases cassés, pour delivery)
	+ --env : environnement d'apprentissage (2 = emergency, 3 = delivery)
Utilité : estimer les poids de l'expert en preference learning à partir des questions que l'on lui pose et des fonctions de récompense estimées des experts.



IDEE GLOBALE ALGO


poids lambda d'un expert -> ppo_train.py -> expert entrainé, agent ppo ->  utils.generate_demos_main.py  -> démonstrations de l'expert -> airl.py -> poids de l'expert estimés + poids fixé de l'expert de preference learning -> MORL -> poids estimés de l'expert de preference learning -> comportements d'un expert ppo qui est l'aggreggation des fonctions de récompense des experts avec les poids de l'expert de preference learning
