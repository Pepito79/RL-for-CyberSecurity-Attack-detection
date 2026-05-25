# Systeme de Detection d'Intrusion par Apprentissage par Renforcement (DQN)

## Table des matieres

1. [Vue d'ensemble du projet](#1-vue-densemble-du-projet)
2. [Architecture globale](#2-architecture-globale)
3. [Le dataset : CIC-IDS2017](#3-le-dataset--cic-ids2017)
4. [Script 1 : cleanner.py - Preprocessing des donnees](#4-script-1--cleannerpy---preprocessing-des-donnees)
5. [Script 2 : gymnasium_env.py - Environnement de Reinforcement Learning](#5-script-2--gymnasium_envpy---environnement-de-reinforcement-learning)
6. [Script 3 : RL_train.py - Entrainement de l'agent DQN](#6-script-3--rl_trainpy---entrainement-de-lagent-dqn)
7. [Script 4 : test_model.py - Evaluation formelle du modele](#7-script-4--test_modelpy---evaluation-formelle-du-modele)
8. [Script 5 : live_demo.py - Demonstration en temps reel](#8-script-5--live_demopy---demonstration-en-temps-reel)
9. [Pipeline d'execution complet](#9-pipeline-dexecution-complet)
10. [Fichiers generes](#10-fichiers-generes)
11. [Hyperparametres et choix de conception](#11-hyperparametres-et-choix-de-conception)

---

## 1. Vue d'ensemble du projet

Ce projet implemente un **Systeme de Detection d'Intrusion (IDS)** base sur le **Deep Q-Network (DQN)**, un algorithme d'apprentissage par renforcement profond.

### Le probleme resolu

Dans un reseau informatique, du trafic arrive en continu. Chaque paquet (ou flux reseau) est soit :
- **BENIGN** : trafic legitime d'un utilisateur normal
- **ATTACK** : trafic malveillant (DDoS, PortScan, attaques web, etc.)

L'agent RL apprend a prendre une **decision binaire** pour chaque flux :
- **Action 0 (PRODUCTION)** : laisser passer le trafic vers le serveur de production
- **Action 1 (HONEYPOT)** : rediriger le trafic vers un honeypot (piege a attaquants)

### Pourquoi le Reinforcement Learning ?

Contrairement a un classifieur supervise classique (Random Forest, SVM...), l'approche RL permet a l'agent de :
- Apprendre par **essai-erreur** via un systeme de recompenses/penalites
- S'adapter a des **politiques de securite asymetriques** (une attaque non detectee est bien plus grave qu'un faux positif)
- Evoluer potentiellement en **temps reel** face a de nouvelles menaces

---

## 2. Architecture globale

```
archive/                          <-- Donnees brutes CIC-IDS2017 (8 fichiers CSV)
    Monday-WorkingHours.pcap_ISCX.csv
    Tuesday-WorkingHours.pcap_ISCX.csv
    Wednesday-workingHours.pcap_ISCX.csv
    Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
    Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
    Friday-WorkingHours-Morning.pcap_ISCX.csv
    Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv

RL-for-CyberSecurity-Attack-detection/
    cleanner.py                   <-- Etape 1 : Preprocessing et nettoyage
    gymnasium_env.py              <-- Etape 2 : Environnement Gymnasium
    RL_train.py                   <-- Etape 3 : Entrainement DQN
    test_model.py                 <-- Etape 4 : Evaluation formelle
    live_demo.py                  <-- Etape 5 : Demo temps reel
    network_scaler.pkl            <-- Scaler sauvegarde (genere a l'entrainement)
    dqn_network_security_final.zip <-- Modele DQN entraine
    pyproject.toml                <-- Dependances Python
```

### Flux de donnees

```
Fichiers CSV bruts (archive/)
        |
        v
  [cleanner.py]  -->  dataset_TRAIN_clean.csv (80%)
        |              dataset_TEST_clean.csv  (20%)
        v
  [gymnasium_env.py]  -->  Environnement Gymnasium avec systeme de recompenses
        |
        v
  [RL_train.py]  -->  dqn_network_security_final.zip (modele sauvegarde)
        |              network_scaler.pkl (normalisation)
        |              models/ (checkpoints intermediaires)
        |              tensorboard_logs/ (metriques d'entrainement)
        v
  [test_model.py]  -->  Matrice de confusion + rapport de classification
  [live_demo.py]   -->  Simulation paquet par paquet avec verdict en direct
```

---

## 3. Le dataset : CIC-IDS2017

Le projet utilise le **CIC-IDS2017** (Canadian Institute for Cybersecurity), un dataset de reference en cybersecurite. Il contient du trafic reseau capture sur 5 jours avec des attaques injectees :

| Jour | Fichier | Contenu |
|------|---------|---------|
| Lundi | `Monday-WorkingHours` | Trafic 100% benin (baseline) |
| Mardi | `Tuesday-WorkingHours` | Attaques Brute Force (FTP, SSH) |
| Mercredi | `Wednesday-workingHours` | DoS (Slowloris, Hulk, GoldenEye), Heartbleed |
| Jeudi matin | `Thursday-Morning-WebAttacks` | Attaques web (XSS, SQL Injection, Brute Force) |
| Jeudi aprem | `Thursday-Afternoon-Infilteration` | Infiltration |
| Vendredi matin | `Friday-Morning` | Botnet (Ares) |
| Vendredi aprem | `Friday-Afternoon-DDos` | DDoS (LOIT) |
| Vendredi aprem | `Friday-Afternoon-PortScan` | PortScan |

Chaque ligne du CSV represente un **flux reseau** decrit par ~80 features (port, duree, taille des paquets, flags TCP, etc.) et un label indiquant le type de trafic.

---

## 4. Script 1 : `cleanner.py` - Preprocessing des donnees

### Role

Transforme les fichiers CSV bruts du dataset CIC-IDS2017 en deux fichiers propres, prets pour l'entrainement et le test.

### Fonctionnement detaille

#### 4.1. Selection des features (`FEATURES_TO_KEEP`)

Sur les ~80 features du dataset original, seules **20 features** sont conservees. Ce choix reduit le bruit et accelere l'entrainement :

| Feature | Signification | Pourquoi elle est utile |
|---------|---------------|------------------------|
| `Destination Port` | Port de destination du flux | Certains ports sont typiques d'attaques (ex: 22 pour SSH brute force) |
| `Flow Duration` | Duree totale du flux en microsecondes | Les attaques DDoS ont souvent des durees anormales |
| `Total Fwd Packets` | Nombre de paquets envoyes (client -> serveur) | Volume anormal = signe d'attaque |
| `Total Backward Packets` | Nombre de paquets reponse (serveur -> client) | Ratio forward/backward revele des patterns |
| `Total Length of Fwd Packets` | Taille totale des paquets forward | Attaques par injection = paquets volumineux |
| `Fwd Packet Length Max` | Taille max d'un paquet forward | Detecte les gros payloads malveillants |
| `Fwd Packet Length Mean` | Taille moyenne des paquets forward | Baseline du trafic normal |
| `Bwd Packet Length Max` | Taille max d'un paquet backward | Exfiltration de donnees = reponses volumineuses |
| `Flow Bytes/s` | Debit en octets par seconde | DDoS = debit anormalement eleve |
| `Flow Packets/s` | Debit en paquets par seconde | Idem, complementaire au debit en octets |
| `Flow IAT Mean` | Temps moyen entre les paquets du flux | Trafic automatise = IAT tres regulier |
| `Flow IAT Std` | Ecart-type du temps inter-paquets | Faible std = probable bot, pas un humain |
| `Flow IAT Max` | Temps max entre deux paquets | Detecte les connexions persistantes |
| `Fwd IAT Total` | Somme des temps inter-paquets forward | Profil temporel de l'emetteur |
| `Fwd Packets/s` | Paquets forward par seconde | Taux d'emission du client |
| `Packet Length Mean` | Longueur moyenne de tous les paquets | Signature globale du flux |
| `SYN Flag Count` | Nombre de flags SYN | SYN flood = DDoS classique |
| `RST Flag Count` | Nombre de flags RST | Port scan = beaucoup de RST (port ferme) |
| `PSH Flag Count` | Nombre de flags PSH | Injection de donnees = PSH frequent |
| `ACK Flag Count` | Nombre de flags ACK | Profil de la connexion TCP |

Une 21eme colonne **`Label`** est ajoutee : c'est la verite terrain (`BENIGN` ou le type d'attaque).

#### 4.2. Chargement par chunks (`chunksize=50000`)

```python
for chunk in pd.read_csv(f, chunksize=50000):
```

Les fichiers CSV font plusieurs centaines de Mo. Au lieu de tout charger en RAM d'un coup, pandas lit **50 000 lignes a la fois**. Chaque chunk est immediatement filtre pour ne garder que les 21 colonnes utiles, puis stocke. Cela evite les erreurs de memoire (OutOfMemoryError).

#### 4.3. Nettoyage des valeurs problematiques

```python
df_global = df_global.replace([np.inf, -np.inf], np.nan).fillna(0)
```

Certaines features comme `Flow Bytes/s` ou `Flow Packets/s` peuvent contenir :
- **`Inf` / `-Inf`** : quand la duree du flux est 0 (division par zero)
- **`NaN`** : quand une donnee est manquante

Ces valeurs sont remplacees par **0** pour eviter de casser le StandardScaler et le reseau de neurones.

#### 4.4. Split train/test (80/20)

```python
df_train, df_test = train_test_split(df_global, test_size=0.20, random_state=42)
```

- **80%** des donnees → `dataset_TRAIN_clean.csv` (entrainement)
- **20%** des donnees → `dataset_TEST_clean.csv` (test)
- `random_state=42` garantit un split **reproductible** : chaque execution produit exactement le meme decoupage
- `shuffle=True` (par defaut) melange les donnees avant le split, ce qui assure que les deux ensembles contiennent un mix de tous les jours et tous les types d'attaques

#### 4.5. Fichiers utilises

Le script ne prend que **5 fichiers sur 8** disponibles dans `archive/`. Les fichiers exclus sont :
- `Tuesday-WorkingHours` (Brute Force FTP/SSH)
- `Thursday-Afternoon-Infilteration` (Infiltration)
- `Friday-Morning` (Botnet)

Ce choix est delibere : le modele est entraine sur un sous-ensemble representatif d'attaques.

---

## 5. Script 2 : `gymnasium_env.py` - Environnement de Reinforcement Learning

### Role

Definit l'**environnement Gymnasium** dans lequel l'agent DQN evolue. C'est le coeur du systeme RL : il transforme le probleme de detection d'intrusion en un probleme de decision sequentielle.

### Concepts cles du Reinforcement Learning

```
                    observation (features du flux)
                           |
                           v
        +------------------+------------------+
        |           AGENT (DQN)               |
        |  Reseau de neurones qui decide      |
        +------------------+------------------+
                           |
                     action (0 ou 1)
                           |
                           v
        +------------------+------------------+
        |      ENVIRONNEMENT (NetworkSecurityEnv)    |
        |  Donne la recompense et le prochain flux   |
        +------------------+------------------+
                           |
                    reward + next_obs
                           |
                           v
                    (cycle suivant)
```

### Fonctionnement detaille

#### 5.1. Initialisation (`__init__`)

```python
def __init__(self, csv_path, is_training=True, scaler_path=None):
```

**Parametres :**
- `csv_path` : chemin vers le CSV nettoye (train ou test)
- `is_training` : flag qui controle le comportement du scaler
- `scaler_path` : chemin vers le fichier de normalisation

**Chargement des donnees :**

```python
self.df = pd.read_csv(csv_path)
self.labels = self.df["Label"].values        # Ex: ["BENIGN", "DDoS", "BENIGN", ...]
raw_features = self.df.drop(columns=["Label"]).values  # Matrice numerique (N x 20)
```

Le CSV est charge entierement en memoire. Les labels sont separes des features.

**Normalisation (StandardScaler) :**

La normalisation est **critique** pour un reseau de neurones. Les features ont des echelles tres differentes :
- `Destination Port` : 0 a 65535
- `Flow Duration` : 0 a plusieurs millions (microsecondes)
- `SYN Flag Count` : 0 a quelques dizaines

Le `StandardScaler` transforme chaque feature pour qu'elle ait une **moyenne de 0** et un **ecart-type de 1** :

```
x_normalise = (x - moyenne) / ecart_type
```

**Mode entrainement (`is_training=True`) :**
1. Le scaler **apprend** les parametres (moyenne, ecart-type) sur les donnees d'entrainement avec `fit_transform()`
2. Le scaler est **sauvegarde** dans `network_scaler.pkl` avec `joblib.dump()`

**Mode evaluation (`is_training=False`) :**
1. Le scaler est **charge** depuis `network_scaler.pkl` avec `joblib.load()`
2. Les donnees de test sont transformees avec les **memes parametres** que l'entrainement via `transform()`

C'est essentiel : si on recalculait le scaler sur les donnees de test, les valeurs normalisees seraient differentes et le modele ne fonctionnerait pas correctement.

**Espace d'actions et d'observations :**

```python
self.action_space = spaces.Discrete(2)         # 2 actions possibles : 0 ou 1
self.observation_space = spaces.Box(            # Vecteur continu de 20 features
    low=-np.inf, high=np.inf,
    shape=(num_features,), dtype=np.float32
)
```

- `Discrete(2)` : l'agent choisit entre 0 (PRODUCTION) et 1 (HONEYPOT)
- `Box(shape=(20,))` : l'observation est un vecteur de 20 nombres reels (les features normalisees)

#### 5.2. Reset (`reset`)

```python
def reset(self, seed=None, options=None):
    self.current_step = 0
    obs = self.features[self.current_step].astype(np.float32)
    return obs, {}
```

Remet l'environnement au debut du dataset. Retourne la premiere observation (le premier flux reseau normalise).

#### 5.3. Step (`step`) - Le systeme de recompenses

C'est la fonction la plus importante. A chaque etape, l'agent recoit une observation, choisit une action, et l'environnement retourne une **recompense**.

```python
def step(self, action):
    true_label = self.labels[self.current_step]

    if true_label == "BENIGN":
        reward = 1.0 if action == 0 else -2.0
    else:
        reward = 5.0 if action == 1 else -20.0
```

**Matrice de recompenses :**

| | Action 0 (PRODUCTION) | Action 1 (HONEYPOT) |
|---|---|---|
| **Trafic BENIGN** | **+1.0** (correct) | **-2.0** (faux positif) |
| **Trafic ATTACK** | **-20.0** (fuite !) | **+5.0** (correct) |

**Justification de l'asymetrie :**

- **+1.0 pour BENIGN → PRODUCTION** : recompense de base pour une decision correcte
- **-2.0 pour BENIGN → HONEYPOT** : penalite moderee pour un faux positif (un utilisateur legitime serait bloque, c'est genant mais pas dangereux)
- **+5.0 pour ATTACK → HONEYPOT** : forte recompense pour avoir detecte une attaque (5x plus que la classification correcte du benin, pour encourager la detection)
- **-20.0 pour ATTACK → PRODUCTION** : penalite **massive** pour une attaque non detectee (c'est le pire scenario : l'attaque atteint le serveur de production). Ce ratio 20:1 par rapport a la recompense de base force l'agent a ne **jamais** laisser passer une attaque, meme au prix de quelques faux positifs

Cette asymetrie reflete la realite : en cybersecurite, **rater une attaque est infiniment plus grave que bloquer un utilisateur legitime par erreur**.

**Fin de l'episode :**

```python
self.current_step += 1
terminated = self.current_step >= len(self.df)
```

L'episode se termine quand tous les flux du dataset ont ete traites. L'agent parcourt le dataset **sequentiellement**, un flux a la fois.

---

## 6. Script 3 : `RL_train.py` - Entrainement de l'agent DQN

### Role

Entraine un agent **Deep Q-Network (DQN)** sur l'environnement de securite reseau.

### Qu'est-ce que DQN ?

DQN (Deep Q-Network) est un algorithme de Reinforcement Learning qui utilise un **reseau de neurones profond** pour approximer la **fonction Q** :

```
Q(s, a) = recompense esperee en prenant l'action a dans l'etat s, puis en suivant la politique optimale
```

L'agent apprend a estimer, pour chaque flux reseau observe (etat `s`), **quelle action** (0=Production ou 1=Honeypot) maximise la recompense totale future.

### Fonctionnement detaille

#### 6.1. Detection du materiel

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Si un GPU NVIDIA (CUDA) est disponible, l'entrainement sera accelere. Sinon, il tourne sur le CPU.

#### 6.2. Monitor wrapper

```python
env = Monitor(env, log_file)
```

Le `Monitor` de Stable-Baselines3 **enveloppe** l'environnement pour enregistrer automatiquement dans `training_stats.csv` :
- La recompense cumulee de chaque episode
- La longueur de chaque episode (nombre de steps)
- Le temps ecoule

#### 6.3. Checkpoints

```python
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path=str(BASE_DIR / "models"),
    name_prefix="dqn_ids_model",
)
```

Tous les **50 000 steps**, le modele est sauvegarde automatiquement dans `models/`. Cela permet de :
- Reprendre l'entrainement en cas de crash
- Comparer les performances a differentes etapes

Les fichiers generes sont nommes `dqn_ids_model_50000_steps.zip`, `dqn_ids_model_100000_steps.zip`, etc.

#### 6.4. Configuration du DQN

```python
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=5e-4,
    buffer_size=100000,
    learning_starts=5000,
    batch_size=256,
    train_freq=1,
    gradient_steps=1,
    tau=0.005,
    gamma=0.99,
    exploration_fraction=0.20,
    exploration_final_eps=0.05,
    device=device,
    verbose=1,
    tensorboard_log=str(BASE_DIR / "tensorboard_logs"),
)
```

**Explication de chaque hyperparametre :**

| Parametre | Valeur | Role |
|-----------|--------|------|
| `MlpPolicy` | - | Reseau de neurones "Multi-Layer Perceptron" (feedforward classique, 2 couches cachees de 64 neurones par defaut) |
| `learning_rate` | `5e-4` (0.0005) | Vitesse d'apprentissage. Trop haut = instable, trop bas = trop lent |
| `buffer_size` | `100000` | Taille du **Replay Buffer** : memoire qui stocke les 100 000 dernieres experiences (s, a, r, s'). L'agent reapprend a partir d'experiences passees, ce qui stabilise l'entrainement |
| `learning_starts` | `5000` | L'agent attend d'avoir accumule 5000 experiences avant de commencer a apprendre. Cela garantit un minimum de diversite dans le buffer |
| `batch_size` | `256` | A chaque mise a jour, 256 experiences sont tirees aleatoirement du buffer pour calculer le gradient. Plus c'est grand, plus le gradient est stable mais plus c'est lent |
| `train_freq` | `1` | Le reseau est mis a jour **a chaque step** (pas tous les N steps). Apprentissage agressif |
| `gradient_steps` | `1` | 1 mise a jour de gradient par step. Equilibre entre vitesse et stabilite |
| `tau` | `0.005` | Taux de mise a jour **soft** du target network. DQN utilise deux reseaux : un "online" (qui apprend) et un "target" (qui fournit des cibles stables). Le target est mis a jour lentement : `target = 0.005 * online + 0.995 * target` |
| `gamma` | `0.99` | Facteur de **discount**. Controle l'importance des recompenses futures. 0.99 signifie que l'agent regarde loin dans le futur (mais ici chaque decision est quasi-independante, donc c'est surtout une convention) |
| `exploration_fraction` | `0.20` | Pendant les premiers 20% des steps (40 000 sur 200 000), l'agent explore en prenant des actions aleatoires (strategie epsilon-greedy). Le taux d'exploration diminue lineairement de 1.0 a `exploration_final_eps` |
| `exploration_final_eps` | `0.05` | Apres la phase d'exploration, l'agent garde 5% de hasard. Cela evite de rester coince dans une politique sous-optimale |
| `verbose` | `1` | Affiche les metriques d'entrainement dans la console |

#### 6.5. Boucle d'entrainement

```python
model.learn(total_timesteps=200000, callback=checkpoint_callback, log_interval=1)
```

L'agent s'entraine pendant **200 000 steps**. A chaque step :
1. L'agent observe le flux courant (20 features normalisees)
2. Il choisit une action (0 ou 1) via epsilon-greedy
3. L'environnement retourne la recompense et le flux suivant
4. L'experience est stockee dans le replay buffer
5. Un batch de 256 experiences est tire du buffer
6. Le gradient est calcule et le reseau est mis a jour

#### 6.6. Sauvegarde finale

```python
model.save(str(BASE_DIR / "dqn_network_security_final"))
```

Le modele final est sauvegarde dans `dqn_network_security_final.zip`. Ce fichier contient :
- Les poids du reseau de neurones
- La configuration des hyperparametres
- L'architecture du reseau

---

## 7. Script 4 : `test_model.py` - Evaluation formelle du modele

### Role

Evalue le modele entraine sur le **dataset de test** (jamais vu pendant l'entrainement) et produit des metriques de performance standard.

### Fonctionnement detaille

#### 7.1. Inference sur le dataset de test

```python
obs, info = env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    agent_actions.append(action)
    true_label = env.labels[env.current_step]
    ground_truth.append(0 if true_label == "BENIGN" else 1)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

- `deterministic=True` : l'agent utilise sa politique apprise **sans aucune exploration aleatoire** (mode evaluation pur)
- Pour chaque flux, on enregistre l'**action de l'agent** et la **verite terrain** (0 = benin, 1 = attaque)

#### 7.2. Matrice de confusion

```python
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
```

La matrice de confusion est le tableau standard d'evaluation d'un classifieur binaire :

```
                    Prediction de l'agent
                    PRODUCTION (0)     HONEYPOT (1)
Realite  BENIGN        TN                 FP
         ATTACK        FN                 TP
```

| Metrique | Signification | Consequence en securite |
|----------|---------------|------------------------|
| **TN** (True Negative) | Trafic benin correctement autorise | OK - l'utilisateur accede normalement au service |
| **FP** (False Positive) | Trafic benin envoye au honeypot par erreur | Genant - un utilisateur legitime est bloque |
| **FN** (False Negative) | Attaque non detectee, envoyee en production | **CRITIQUE** - l'attaque atteint le serveur reel |
| **TP** (True Positive) | Attaque correctement redirigee vers le honeypot | OK - l'attaque est isolee et peut etre etudiee |

#### 7.3. Rapport de classification (sklearn)

```python
classification_report(y_true, y_pred, target_names=["Production (Valid)", "Honeypot (Threat)"])
```

Genere automatiquement les metriques par classe :
- **Precision** : parmi les flux envoyes au honeypot, quel % etait vraiment des attaques ?
- **Recall** : parmi toutes les attaques, quel % a ete detecte ?
- **F1-score** : moyenne harmonique de precision et recall
- **Support** : nombre d'echantillons par classe

---

## 8. Script 5 : `live_demo.py` - Demonstration en temps reel

### Role

Simule le fonctionnement de l'IDS **en temps reel**, paquet par paquet, avec un affichage lisible des decisions.

### Fonctionnement detaille

#### 8.1. Boucle de traitement

```python
for i in range(num_packets):
    action, _ = model.predict(obs, deterministic=True)
    true_label = env.labels[env.current_step]
    is_attack = true_label != "BENIGN"
```

Pour chaque paquet (jusqu'a `num_packets` = 10 000 par defaut) :
1. Le modele predit l'action
2. Le label reel est recupere pour verification
3. La decision est comparee a la verite terrain

#### 8.2. Classification des resultats

Chaque decision tombe dans l'une des 4 categories :

| Categorie | Action | Realite | Verdict |
|-----------|--------|---------|---------|
| `bon_autorise` | PRODUCTION | BENIGN | Correct - True Negative |
| `mauvais_bloque` | HONEYPOT | ATTACK | Correct - True Positive |
| `bon_bloque` | HONEYPOT | BENIGN | Erreur - False Positive |
| `mauvais_autorise` | PRODUCTION | ATTACK | **DANGER** - False Negative |

#### 8.3. Affichage

```
STEP     | DECISION        | VERITE     | RESULTAT
-------------------------------------------------------
0        | PRODUCTION      | SAFE       | BON AUTORISE
10       | HONEYPOT        | ATTACK     | MAUVAIS BLOQUE
20       | PRODUCTION      | ATTACK     | MAUVAIS AUTORISE (FUITE!)
```

L'affichage se fait toutes les **10 lignes** pour la lisibilite (les 50 premiers sont affiches un par un).

#### 8.4. Bilan final

A la fin, un resume est affiche avec :
- Le nombre de decisions dans chaque categorie
- La **precision globale** : `(TN + TP) / total * 100`

---

## 9. Pipeline d'execution complet

### Etape 1 : Preprocessing
```bash
cd RL-for-CyberSecurity-Attack-detection
python cleanner.py
```
**Entree :** 5 fichiers CSV bruts dans `archive/`
**Sortie :** `dataset_TRAIN_clean.csv` et `dataset_TEST_clean.csv`
**Duree estimee :** 1 a 5 minutes selon la machine

### Etape 2 : Entrainement
```bash
python RL_train.py
```
**Entree :** `dataset_TRAIN_clean.csv`
**Sortie :** `dqn_network_security_final.zip`, `network_scaler.pkl`, checkpoints dans `models/`
**Duree estimee :** 10 minutes a 2 heures (selon GPU/CPU)

### Etape 3 : Evaluation
```bash
python test_model.py
```
**Entree :** `dataset_TEST_clean.csv` + modele entraine
**Sortie :** Matrice de confusion et rapport de classification (console)

### Etape 4 : Demo (optionnel)
```bash
python live_demo.py
```
**Entree :** `dataset_TEST_clean.csv` + modele entraine
**Sortie :** Simulation paquet par paquet avec verdicts (console)

---

## 10. Fichiers generes

| Fichier | Genere par | Description |
|---------|-----------|-------------|
| `dataset_TRAIN_clean.csv` | `cleanner.py` | Dataset d'entrainement nettoye (80% des donnees, ~20 colonnes) |
| `dataset_TEST_clean.csv` | `cleanner.py` | Dataset de test nettoye (20% des donnees) |
| `network_scaler.pkl` | `gymnasium_env.py` | Parametres de normalisation (moyenne et ecart-type de chaque feature) serialises avec joblib |
| `dqn_network_security_final.zip` | `RL_train.py` | Modele DQN entraine (poids du reseau, config, architecture) |
| `training_stats.csv` | `RL_train.py` (Monitor) | Log des recompenses et longueurs d'episodes pendant l'entrainement |
| `models/dqn_ids_model_*_steps.zip` | `RL_train.py` (Checkpoint) | Sauvegardes intermediaires tous les 50 000 steps |
| `tensorboard_logs/` | `RL_train.py` | Logs TensorBoard pour visualiser les courbes d'entrainement |

---

## 11. Hyperparametres et choix de conception

### Pourquoi DQN plutot qu'un autre algorithme RL ?

- **Espace d'actions discret** : DQN est concu pour les problemes a actions discretes (ici 2 actions). PPO ou A2C fonctionneraient aussi, mais DQN est plus simple et performant pour des actions binaires.
- **Replay Buffer** : DQN reutilise des experiences passees, ce qui est tres efficace quand les donnees sont sequentielles (un dataset de flux reseau).
- **Stabilite** : le target network et le replay buffer rendent l'entrainement plus stable qu'un Q-learning tabulaire.

### Pourquoi StandardScaler ?

- Les reseaux de neurones convergent beaucoup plus vite quand les entrees sont centrees-reduites
- Sans normalisation, les features a grande echelle (comme `Flow Duration` en microsecondes) domineraient les autres dans le gradient

### Pourquoi un split 80/20 global plutot que par jour ?

- Un split par jour introduirait un biais : certains types d'attaques n'apparaissent qu'un seul jour
- Le melange global garantit que train et test contiennent **tous les types d'attaques** dans des proportions similaires
- `random_state=42` rend le split reproductible

### Pourquoi les recompenses sont-elles asymetriques ?

- En securite, le **cout d'un faux negatif** (attaque non detectee) est infiniment superieur au **cout d'un faux positif** (utilisateur bloque)
- Le ratio -20 vs -2 force l'agent a prioriser la detection d'attaques, quitte a avoir quelques faux positifs
- Le ratio +5 vs +1 renforce positivement la detection d'attaques plus que la classification du benin

### Adaptation des chemins

Tous les scripts utilisent `pathlib.Path` avec `BASE_DIR = Path(__file__).resolve().parent` pour calculer les chemins **relativement au script**. Cela rend le projet portable : il fonctionne sur n'importe quelle machine (Windows, Linux, macOS) sans modifier les chemins.

```
BASE_DIR = RL-for-CyberSecurity-Attack-detection/   (dossier des scripts)
DATA_DIR = archive/                                   (dossier des CSV bruts, un niveau au-dessus)
```
