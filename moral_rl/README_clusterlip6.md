se connecter :

	ssh lechapelier@cluster.lip6.fr


accéder au dépot depuis ma session :

	cd /net/big/lechapelier


lancer une tâche :
	
	oarsub -l "/nodes=1/core=24,walltime=01:00:00" "python3 test.py"
	-> retourne le JOB_ID, ex : JOB_ID=982376


se connecter directement au noeud, puis lancer une tâche :
	
	oarsub -I
	python3 test.py
	exit


annuler une tâche :
	
	oardel [JOB_ID]


visualiser les fichiers de logs/errors :

	cat OAR.[JOB_ID].stderr
	cat OAR.[JOB_ID].stdout

	ou 

	nano OAR.[JOB_ID].stderr
	-> nano permet également de modifier des fichiers à partir du terminal