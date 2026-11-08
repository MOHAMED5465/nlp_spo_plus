Maintenant que ta fondation en Recherche Opérationnelle (Partie 1) est robuste, codée et analysée, tu as prouvé que le pipeline de *Decision-Focused Learning* (PyEPO) fonctionne sur ton problème de routage (CVRP) et tu maîtrises le compromis entre les différentes relaxations (MTZ, DFJ, AP, Flow).

L'étape suivante est le cœur de ta contribution scientifique (Partie 2) : **remplacer les variables numériques globales par du texte localisé non structuré.**

Voici la feuille de route logique et rigoureuse pour y parvenir. Le prochain objectif immédiat est la **génération du jeu de données NLP**.

### 1. La conception du générateur de données (`genData_NLP`)

Tu ne peux plus utiliser ta matrice $B$ et ton vecteur global $x$. Il faut spatialiser l'information pour que le texte ait un sens géographique.

**A. Définition de l'état statique du graphe**

* Générer les coordonnées des $N$ clients et du dépôt.
* Calculer la matrice des distances euclidiennes de base $D$.

**B. Injection des variables latentes (La vérité terrain cachée)**
Pour chaque nœud $i$ (chaque client), tu dois échantillonner des états latents qui affecteront les coûts des arêtes reliées à ce nœud.

* *Exemple de variables pour le nœud $i$ :* `Météo` $\in$ {Clair, Pluie, Neige}, `Trafic` $\in$ {Fluide, Dense, Bloqué}, `Événement` $\in$ {Aucun, Manifestation, Travaux}.
* **La fonction de coût réel ($C^*$) :** Tu définis mathématiquement comment ces états interagissent pour créer le coût réel de l'arête $(i, j)$. Pour garantir la "mauvaise spécification" (misspecification) qui fait briller SPO+, cette fonction doit être complexe (ex: non-linéaire ou conditionnelle).
* *Exemple :* $C^*_{ij} = D_{ij} \times (1 + \text{Pénalité}(Trafic_i) \times \text{Pénalité}(Meteo_j))^2$



### 2. La génération du texte avec distracteurs (Le rôle du LLM "A")

C'est ici que tu construis ton dataset textuel de 250 instances.
Tu vas écrire un script (utilisant l'API OpenAI, Gemini, ou un modèle local via Ollama/vLLM) pour transformer les variables latentes de chaque nœud en texte ($T_i$).

* **Le Prompt système :** "Tu es un assistant de gestion de trafic urbain. Rédige un bref bulletin de situation pour un quartier ayant les caractéristiques suivantes : [Variables Latentes]. Ajoute une ou deux phrases de contexte local ou de faits divers qui n'ont AUCUN impact sur la logistique."
* **Les Distracteurs (Crucial pour la publication) :** Les phrases comme "Le maire a inauguré une nouvelle école" ou "Un concert de jazz se prépare sur la place" obligeront ton futur modèle d'optimisation (SPO+) à apprendre à *ignorer* ces informations (l'attention ira vers zéro), contrairement à une approche purement MSE.
* **Résultat attendu :** Pour chaque instance d'entraînement, tu as un graphe, une matrice de coût $C^*$, et une liste de $N$ textes (un par nœud).

### 3. La conception de l'Architecture Text-to-Cost (Le modèle "B")

Une fois tes données générées, tu devras construire le modèle Deep Learning qui remplacera ton MLP actuel.

* **L'Encodeur :** Un modèle pré-entraîné (ex: DeBERTa ou RoBERTa) figé avec un adaptateur LoRA pour éviter le surapprentissage sur tes 250 instances.
* **Le Forward Pass :**
1. Le texte du nœud $i$ ($T_i$) passe dans l'encodeur $\rightarrow$ embedding $e_i$.
2. Le texte du nœud $j$ ($T_j$) passe dans l'encodeur $\rightarrow$ embedding $e_j$.
3. Concaténation : $[e_i, e_j, D_{ij}]$.
4. Passage dans une tête de régression (un petit MLP) pour prédire $\hat{C}_{ij}$.



### 4. Le Fine-Tuning (SPO+ vs MSE sur le texte)

Tu réutiliseras exactement ta boucle d'entraînement actuelle, mais avec l'architecture Text-to-Cost.

* Tu utiliseras la relaxation **AP** ou **Flow** (puisque ton modèle Text-to-Cost sera lourd, la relaxation DFJ prendrait des jours entiers à s'entraîner).
* **L'hypothèse à valider :** La MSE va forcer l'encodeur DeBERTa à prêter attention à tout le texte pour deviner le coût exact. La perte SPO+ va forcer l'encodeur à ne lire que les mots décisifs ("Bloqué", "Neige") liés aux arêtes qui se trouvent sur la frontière de décision optimale.

---

**Par quoi veux-tu commencer aujourd'hui ?**
Nous pouvons soit définir mathématiquement tes variables latentes et la fonction $C^*$ de vérité terrain, soit concevoir directement le script et le prompt pour le LLM générateur de texte.