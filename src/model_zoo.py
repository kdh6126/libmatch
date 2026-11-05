"""일곱 모델의 정의. exp16(실측)과 exp82(격자)가 같은 것을 쓴다.

두 실험이 각자 복사본을 들면 하이퍼파라미터가 조용히 갈라지고, 그때 논문은
"같은 일곱 모델"이라고 쓰면서 실제로는 두 벌을 비교하게 된다. 정의를 한 곳에
두고 양쪽이 가져다 쓴다.

모델마다 필요한 표현이 다르다. 능선 회귀는 정렬 one-hot 또는 ESM 임베딩을,
나무 계열과 최근접 이웃은 자리별 토큰을, 합성곱망은 토큰 열을 받는다. 분할은
전부 같다 -- 표현이 달라도 같은 학습·보류 집합을 본다.
"""
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

sys.path.insert(0, "src")
from exp2_search import fit_cnn, cnn_score                      # noqa: E402

NAMES = ["ridge_onehot", "boosted_trees", "random_forest", "knn_hamming",
         "cnn_maxpool", "ridge_esm", "ridge_esm+onehot"]


def models(Xs, tok, Xesm, tr, te, ytr, seed):
    """Every model sees the same split; each gets the representation it needs."""
    out = {}
    r = Ridge(alpha=1.0, solver="lsqr").fit(Xs[tr], ytr)
    out["ridge_onehot"] = Xs[te] @ r.coef_ + r.intercept_

    h = HistGradientBoostingRegressor(max_iter=300, learning_rate=.08,
                                      random_state=seed).fit(tok[tr], ytr)
    out["boosted_trees"] = h.predict(tok[te])

    f = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1,
                              random_state=seed).fit(tok[tr], ytr)
    out["random_forest"] = f.predict(tok[te])

    k = KNeighborsRegressor(n_neighbors=10, metric="hamming",
                            n_jobs=-1).fit(tok[tr], ytr)
    out["knn_hamming"] = k.predict(tok[te])

    c, mu, sd = fit_cnn(tok[tr], ytr, seed=seed, vocab=22)
    out["cnn_maxpool"] = cnn_score(c, tok[te], mu, sd)

    if Xesm is not None:
        re = Ridge(alpha=50.0, solver="lsqr").fit(Xesm[tr], ytr)
        out["ridge_esm"] = re.predict(Xesm[te])
        hb = np.hstack([Xesm, Xs.toarray()]) if Xs.shape[1] < 3000 else None
        if hb is not None:
            rh = Ridge(alpha=50.0, solver="lsqr").fit(hb[tr], ytr)
            out["ridge_esm+onehot"] = rh.predict(hb[te])
    return out
