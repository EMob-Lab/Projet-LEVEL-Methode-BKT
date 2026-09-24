import warnings

import numpy as np
import pandas as pd
import scipy.linalg
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans


class funFEM:
    """
    Implémentation du modèle funFEM pour la classification de données fonctionnelles.

    explication de l'algorithme et des paramètres :
    - cluster_interval : intervalle de nombres de clusters à tester (par exemple, (2, 6) pour tester de 2 à 6 clusters)
    - model : modèle de covariance à utiliser (par exemple, "AkjBk")
    - crit : critère de sélection du modèle (par exemple, "bic")
    - init : méthode d'initialisation (par exemple, "kmeans")
    - Tinit : initialisation des paramètres de transformation (par exemple, None)
    - maxit : nombre maximum d'itérations pour l'algorithme EM
    - eps : tolérance pour la convergence de l'algorithme EM
    - disp : affichage des informations de progression (True ou False)
    - random_state : graine pour la reproductibilité des résultats (par exemple, None)

    explication des MODELS :
    - DkBk : modèle avec covariance discriminante par cluster et bruit par cluster
    - DkB : modèle avec covariance discriminante par cluster et bruit global
    - DBk : modèle avec covariance discriminante globale et bruit par cluster
    - DB : modèle avec covariance discriminante globale et bruit global
    - AkjBk : modèle avec covariance discriminante par cluster et bruit par cluster, avec contraintes sur les valeurs propres
    - AkjB : modèle avec covariance discriminante par cluster et bruit global, avec contraintes sur les valeurs propres
    - AkBk : modèle avec covariance discriminante par cluster et bruit par cluster, avec contraintes sur les valeurs propres
    - AkB : modèle avec covariance discriminante par cluster et bruit global, avec contraintes sur les valeurs propres
    - AjBk : modèle avec covariance discriminante par cluster et bruit par cluster, avec contraintes sur les valeurs propres
    - AjB : modèle avec covariance discriminante par cluster et bruit global, avec contraintes sur les valeurs propres
    - ABk : modèle avec covariance discriminante globale et bruit par cluster, avec contraintes sur les valeurs propres
    - AB : modèle avec covariance discriminante globale et bruit global, avec contraintes sur les valeurs propres

    explication des CRITERIA :
    - bic : critère d'information bayésien
    - aic : critère d'information d'Akaike
    - icl : critère d'information de classification

    explication des INIT :
    - user : initialisation par l'utilisateur (Tinit doit être fourni)
    - random : initialisation aléatoire
    - kmeans : initialisation par K-means
    - hclust : initialisation par clustering hiérarchique

    exemple d'utilisation :
    >>> import numpy as np
    >>> from FunFEM import funFEM
    >>> # Génération de données simulées
    >>> np.random.seed(0)
    >>> n_samples = 100
    >>> n_features = 10
    >>> X = np.random.rand(n_samples, n_features)
    >>> W = np.eye(n_features)  # Matrice d'identité pour l'exemple
    >>> # Création de l'objet funFEM
    >>> model = funFEM(cluster_interval=(2, 4), model="AkjBk", crit="bic", init="kmeans", disp=True)
    >>> # Ajustement du modèle aux données
    >>> model.fit(X, W)
    >>> # Affichage des résultats
    >>> print("Best model:", model.model)
    >>> print("Best number of clusters:", model.K)
    """

    MODELS = [
        "DkBk",
        "DkB",
        "DBk",
        "DB",
        "AkjBk",
        "AkjB",
        "AkBk",
        "AkB",
        "AjBk",
        "AjB",
        "ABk",
        "AB",
        "all",
    ]
    CRITERIA = ["bic", "aic", "icl"]
    INIT = ["user", "random", "kmeans", "hclust"]

    def __init__(
        self,
        cluster_interval=(2, 6),
        model="AkjBk",
        crit="bic",
        init="kmeans",
        Tinit=None,
        maxit=50,
        eps=1e-6,
        disp=False,
        random_state=None,
        em_step_max_retries=5,
    ):
        if isinstance(model, str):
            model = [model]
        for m in model:
            if m not in self.MODELS:
                raise ValueError(f"model should be in {self.MODELS}")
        if crit not in self.CRITERIA:
            raise ValueError(f"crit should be in {self.CRITERIA}")
        if init not in self.INIT:
            raise ValueError(f"init should be in {self.INIT}")

        if isinstance(cluster_interval, int):
            self.K_list = [cluster_interval]
        elif isinstance(cluster_interval, tuple) and len(cluster_interval) == 2:
            self.K_list = list(range(cluster_interval[0], cluster_interval[1] + 1))
        else:
            self.K_list = list(cluster_interval)
        for k in self.K_list:
            if k < 2:
                raise ValueError("K=1 is not allowed!")
        if em_step_max_retries < 1:
            raise ValueError("em_step_max_retries must be at least 1")

        self.model_list = (
            [m for m in self.MODELS if m != "all"] if "all" in model else model
        )
        self.crit = crit
        self.init = init
        self.Tinit = [] if Tinit is None else Tinit
        self.maxit = maxit
        self.eps = eps
        self.disp = disp
        self.em_step_max_retries = em_step_max_retries
        self.random_state = random_state

        self.best_model_ = None
        self.all_models_ = {}
        self.allCriterions = None
        self._W = None
        self._V = None
        self._prms = None

    # ── criteria ─────────────────────────────────────────────────────
    def _criteria(self, loglik, T, prms, n):
        K = prms["K"]
        p = prms["p"]
        model = prms["model"]
        comp = {
            "DkBk": (K - 1)
            + K * (K - 1)
            + (K - 1) * (p - K / 2)
            + K * K * (K - 1) // 2
            + K,
            "DkB": (K - 1)
            + K * (K - 1)
            + (K - 1) * (p - K / 2)
            + K * K * (K - 1) // 2
            + 1,
            "DBk": (K - 1) + K * (K - 1) + (K - 1) * (p - K / 2) + K * (K - 1) // 2 + K,
            "DB": (K - 1) + K * (K - 1) + (K - 1) * (p - K / 2) + K * (K - 1) // 2 + 1,
            "AkjBk": (K - 1) + K * (K - 1) + (K - 1) * (p - K / 2) + K * K,
            "AkjB": (K - 1) + K * (K - 1) + (K - 1) * (p - K / 2) + K * (K - 1) + 1,
            "AkBk": (K - 1) + K * (K - 1) + (K - 1) * (p - K / 2) + 2 * K,
            "AkB": (K - 1) + K * (K - 1) + (K - 1) * (p - K / 2) + K + 1,
            "AjBk": (K - 1) + K * (K - 1) + (K - 1) * (p - K / 2) + (K - 1) + K,
            "AjB": (K - 1) + K * (K - 1) + (K - 1) * (p - K / 2) + (K - 1) + 1,
            "ABk": (K - 1) + K * (K - 1) + (K - 1) * (p - K / 2) + K + 1,
            "AB": (K - 1) + K * (K - 1) + (K - 1) * (p - K / 2) + 2,
        }[model]
        aic = loglik - comp
        bic = loglik - 0.5 * comp * np.log(n)
        T_eps = np.clip(T, 1e-6, None)
        icl = loglik - 0.5 * comp * np.log(n) - np.sum(T * np.log(T_eps))
        return {"aic": aic, "bic": bic, "icl": icl, "nbprm": comp}

    # ── E‑step ──────────────────────────────────────────────────────
    def _estep(self, prms, fd_coefs, U):
        Y = fd_coefs
        n, p = Y.shape
        K = prms["K"]
        D = prms["D"]
        prop = prms["prop"]
        d = K - 1
        QQ = np.zeros((n, K))
        for k in range(K):
            bk = D[k, p - 1, p - 1]
            mY = prms["my"][k, :]
            YY = Y - mY
            projYY = YY @ U @ U.T
            X_proj = projYY @ U
            rss = np.sum((YY - projYY) ** 2, axis=1)

            if d == 1:
                quad_form = (1.0 / max(D[k, 0, 0], 1e-12)) * np.sum(X_proj**2, axis=1)
                log_det = np.log(max(D[k, 0, 0], 1e-12))
            else:
                D_disc = D[k, :d, :d]
                inv_Sigma = scipy.linalg.pinv(D_disc)
                quad_form = np.sum((X_proj @ inv_Sigma) * X_proj, axis=1)
                log_det = np.log(max(scipy.linalg.det(D_disc), 1e-30))

            QQ[:, k] = (
                quad_form
                + (1.0 / max(bk, 1e-12)) * rss
                + (p - d) * np.log(max(bk, 1e-12))
                + log_det
                - 2.0 * np.log(max(prop[k], 1e-12))
                + p * np.log(2.0 * np.pi)
            )

        A = -0.5 * QQ
        A_max = np.max(A, axis=1)
        loglik = np.sum(np.log(np.sum(np.exp(A - A_max[:, None]), axis=1)) + A_max)
        T = np.exp(A - A_max[:, None])
        T = T / np.sum(T, axis=1, keepdims=True)
        return T, loglik

    # ── F‑step (generalised problem in the original metric W) ────────
    def _fstep(self, fd_coefs, basis_inprod, T):
        if np.min(np.sum(T, axis=0)) <= 1:
            raise ValueError("One cluster is almost empty!")
        Y = fd_coefs
        n, p = Y.shape
        K = T.shape[1]
        d = K - 1
        W = basis_inprod
        nk = np.sum(T, axis=0)
        Ttilde = T / np.sqrt(np.maximum(nk, 1e-10))[None, :]
        GtGW = Y.T @ Y @ W
        GtTTtGW = Y.T @ Ttilde @ Ttilde.T @ Y @ W
        try:
            M = scipy.linalg.solve(GtGW, GtTTtGW)
        except scipy.linalg.LinAlgError:
            M = scipy.linalg.pinv(GtGW) @ GtTTtGW
        U_raw, s, _ = scipy.linalg.svd(M)
        U = U_raw[:, :d]
        return U

    # ── M‑step ──────────────────────────────────────────────────────
    def _mstep(self, fd_coefs, basis_inprod, U, T, model):
        Y = fd_coefs
        n, p = Y.shape
        K = T.shape[1]
        d = K - 1
        U_proj = basis_inprod @ U  # R: U = t(W) %*% U
        X = Y @ U_proj
        C_overall = np.cov(Y, rowvar=False, bias=False)

        mu = np.zeros((K, d))
        m = np.zeros((K, p))
        prop = np.zeros(K)
        D = np.zeros((K, p, p))

        per_cluster_b = {"DkBk", "DBk", "AkjBk", "AjBk", "AkBk", "ABk"}

        for k in range(K):
            nk = np.sum(T[:, k])
            prop[k] = nk / n
            mu[k, :] = np.sum(T[:, k : k + 1] * X, axis=0) / nk
            m[k, :] = np.sum(T[:, k : k + 1] * Y, axis=0) / nk
            YY = Y - m[k, :]
            Ck = (YY.T @ (T[:, k : k + 1] * YY)) / max(nk - 1, 1)
            Dk_disc = U_proj.T @ (Ck @ U_proj)
            C_disc = U_proj.T @ (C_overall @ U_proj)

            # discriminant covariance
            if model in ("DkBk", "DkB"):
                D[k, :d, :d] = Dk_disc
            elif model in ("DBk", "DB"):
                D[k, :d, :d] = C_disc
            elif model in ("AkjBk", "AkjB"):
                D[k, :d, :d] = np.diag(np.diag(Dk_disc)) if d > 1 else Dk_disc
            elif model in ("AjBk", "AjB"):
                D[k, :d, :d] = np.diag(np.diag(C_disc)) if d > 1 else C_disc
            elif model in ("AkBk", "AkB"):
                val = np.trace(Dk_disc) / d if d > 0 else np.trace(Dk_disc)
                D[k, :d, :d] = val * np.eye(d) if d > 1 else np.array([[val]])
            elif model in ("ABk", "AB"):
                val = np.trace(C_disc) / d if d > 0 else np.trace(C_disc)
                D[k, :d, :d] = val * np.eye(d) if d > 1 else np.array([[val]])

            # noise variance b (matches R's bk <= 0 -> 0.001)
            if model in per_cluster_b:
                bk = (np.trace(Ck) - np.trace(Dk_disc)) / (p - d)
            else:
                bk = (np.trace(C_overall) - np.trace(C_disc)) / (p - d)
            if bk <= 0:
                bk = 0.001
            np.fill_diagonal(D[k, d:, d:], bk)

        return {
            "K": K,
            "p": p,
            "mean": mu,
            "my": m,
            "prop": prop,
            "D": D,
            "model": model,
        }

    # ── EM main ─────────────────
    def _funfem_main(self, fd_coefs, basis_inprod, K, model="AkjBk"):
        Y = fd_coefs
        n, p = Y.shape
        Lobs = [-np.inf] * (self.maxit + 1)

        # ── Initialisation with retry on empty clusters ─
        MAX_RETRIES = self.em_step_max_retries
        T = None
        V = None
        use_random = False  # once we fall back, stay on random
        for attempt in range(MAX_RETRIES):
            if use_random or self.init == "random":
                rng = np.random.default_rng(
                    self.random_state + attempt * 100
                    if self.random_state is not None
                    else None
                )
                T = rng.multinomial(1, [1 / K] * K, size=n)
            elif self.init == "user" and len(self.Tinit) > 0:
                T = self.Tinit
            elif self.init == "kmeans":
                rstate = (
                    self.random_state + attempt
                    if self.random_state is not None
                    else None
                )
                mod = KMeans(n_clusters=K, n_init=10, random_state=rstate).fit(Y)
                T = np.zeros((n, K))
                T[np.arange(n), mod.labels_] = 1
            elif self.init == "hclust":
                Z = linkage(Y, method="ward")
                labels = fcluster(Z, K, criterion="maxclust") - 1
                T = np.zeros((n, K))
                T[np.arange(n), labels] = 1

            # Check for near-empty clusters
            col_sums = np.sum(T, axis=0)
            if np.min(col_sums) <= 1:
                if attempt < MAX_RETRIES - 1:
                    use_random = True
                    continue
                else:
                    raise ValueError(
                        f"K={K}: could not avoid empty clusters "
                        f"after {MAX_RETRIES} attempts."
                    )

            try:
                V = self._fstep(fd_coefs, basis_inprod, T)
                break  # success
            except ValueError:
                if attempt < MAX_RETRIES - 1:
                    use_random = True
                    continue
                else:
                    raise

        # ── EM loop ─
        prms = self._mstep(fd_coefs, basis_inprod, V, T, model)
        T, loglik = self._estep(prms, fd_coefs, V)
        Lobs[0] = loglik

        Linf_new = Lobs[0]
        final_i = 0
        for i in range(self.maxit):
            V = self._fstep(fd_coefs, basis_inprod, T)
            prms = self._mstep(fd_coefs, basis_inprod, V, T, model)
            T, loglik = self._estep(prms, fd_coefs, V)
            Lobs[i + 1] = loglik

            if i >= 1 and Lobs[i + 1] < Lobs[i] - 1e-8:
                if self.disp:
                    print("Warning: LL decreased. Stopping.")
                final_i = i
                break
            if i >= 1:
                acc = (
                    (Lobs[i + 1] - Lobs[i]) / (Lobs[i] - Lobs[i - 1])
                    if (Lobs[i] - Lobs[i - 1]) != 0
                    else 0
                )
                Linf_old = Linf_new
                Linf_new = (
                    Lobs[i] + 1 / (1 - acc) * (Lobs[i + 1] - Lobs[i])
                    if (1 - acc) != 0
                    else Lobs[i + 1]
                )
                if abs(Linf_new - Linf_old) < self.eps or np.isnan(Linf_new):
                    final_i = i
                    break
            final_i = i

        cls = np.argmax(T, axis=1)
        crit = self._criteria(Lobs[final_i + 1], T, prms, n)
        U_orig = basis_inprod.T @ V  # R: U = t(W) %*% V
        return {
            "model": model,
            "K": K,
            "cls": cls,
            "P": T,
            "prms": prms,
            "U": U_orig,
            "V": V,
            "aic": crit["aic"],
            "bic": crit["bic"],
            "icl": crit["icl"],
            "loglik": Lobs[1 : final_i + 2],
            "ll": Lobs[final_i + 1],
            "nbprm": crit["nbprm"],
        }

    # ── fit ─────────────────────────────────────────────────────────
    def fit(self, X, W):
        n, p = X.shape
        self._W = np.asarray(W, dtype=float)

        all_criterions = []
        n_skipped = 0
        pbar = None
        if self.disp:
            try:
                from tqdm.auto import tqdm

                total_iters = len(self.K_list) * len(self.model_list)
                pbar = tqdm(
                    total=total_iters, desc=f"EM ({self.crit.upper()})", leave=False
                )
            except ImportError:
                print("Computing models... (install 'tqdm' for a progress bar)")

        for k in self.K_list:
            for m in self.model_list:
                try:
                    res = self._funfem_main(X, self._W, k, model=m)
                    self.all_models_[(k, m)] = res
                    all_criterions.append(
                        {
                            "K": k,
                            "model": m,
                            "bic": res["bic"],
                            "aic": res["aic"],
                            "icl": res["icl"],
                            "nbprm": res["nbprm"],
                            "ll": res["ll"],
                        }
                    )
                except Exception as e:
                    n_skipped += 1
                    if self.disp and n_skipped <= 10:
                        print(f"  [skip] K={k:2d} {m:6s}: {e}")
                if pbar:
                    pbar.update(1)
        if pbar:
            pbar.close()

        if n_skipped > 0 and self.disp:
            print(f"  ({n_skipped} combos skipped)")

        if not all_criterions:
            raise ValueError("No reliable results (all K×model combos failed)!")

        self.allCriterions = pd.DataFrame(all_criterions)
        best_idx = self.allCriterions[self.crit].idxmax()
        best_row = self.allCriterions.loc[best_idx]
        best_k = int(best_row["K"])
        best_m = best_row["model"]

        if self.disp:
            pivot = self.allCriterions.pivot(
                index="model", columns="K", values=self.crit
            )
            print(f"\n --- funFEM Results ({self.crit.upper()}) --- ")
            try:
                from IPython.display import display as ipyd

                ipyd(pivot.round(2))
            except ImportError:
                print(pivot.round(2).to_string())
            print(
                f"\nBest: {best_m}  K={best_k}  "
                f"({self.crit.upper()}={best_row[self.crit]:.2f})\n"
            )

        self.best_model_ = self.all_models_[(best_k, best_m)]
        self.U = self.best_model_["U"]  # = W @ V, as in R
        self._V = self.best_model_["V"]  # raw discriminant directions (for predict)
        self.prms = self.best_model_["prms"]

        self.K = best_k
        self.model = best_m
        self.cls = self.best_model_["cls"]
        self.aic = self.best_model_["aic"]
        self.bic = self.best_model_["bic"]
        self.icl = self.best_model_["icl"]
        self.loglik = self.best_model_["loglik"]
        self.ll = self.best_model_["ll"]
        return self

    def predict_proba(self, X):
        if self._V is None:
            raise ValueError("Model must be fitted first.")
        T, _ = self._estep(self.prms, X, self._V)
        return T

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
