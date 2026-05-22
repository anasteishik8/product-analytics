"""
Ядерная регрессия Надарая–Ватсона и локально-линейная регрессия.

Реализует непараметрические методы регрессии на основе ядерного сглаживания.
Основной метод — оценка Надарая–Ватсона:

    m(x) = sum_i K_h(x - x_i) * y_i / sum_i K_h(x - x_i)

где K_h(u) = K(u/h) / h — ядерная функция с шириной окна h.

Для многомерных x используется произведение одномерных ядер (product kernel)
или гауссовское ядро RBF:

    K(x, x_i) = exp(-||x - x_i||^2 / (2 * h^2))

Примечание производительности:
    При d > 10 и n < 500 вычисление матрицы расстояний O(n^2 * d) может
    занимать несколько секунд. Для n=222, d=15 — приемлемо.

Ссылки:
    [1] Nadaraya E.A. (1964). On estimating regression.
    [2] Watson G.S. (1964). Smooth regression analysis.
    [3] Fan J., Gijbels I. (1996). Local Polynomial Modelling.
"""
from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
from sklearn.model_selection import KFold, cross_val_score

DEFAULT_SEED = 42
DEFAULT_BANDWIDTHS = np.logspace(-2, 1, 20)  # от 0.01 до 10, 20 точек


def gaussian_kernel(u: np.ndarray) -> np.ndarray:
    """Гауссовское ядро K(u) = exp(-u^2 / 2) / sqrt(2*pi).

    Parameters
    ----------
    u : np.ndarray
        Нормированные расстояния u = ||x - x_i|| / h.

    Returns
    -------
    np.ndarray
        Значения ядра той же формы, что и входной массив.

    Examples
    --------
    >>> gaussian_kernel(np.array([0.0, 1.0]))
    array([0.39894228, 0.24197072])
    """
    return np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)


def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    """Ядро Эпанечникова K(u) = 0.75 * (1 - u^2) * I(|u| <= 1).

    Оптимальное в смысле минимума среднеквадратичной ошибки (MSE)
    среди симметричных неотрицательных ядер.

    Parameters
    ----------
    u : np.ndarray
        Нормированные расстояния u = ||x - x_i|| / h.

    Returns
    -------
    np.ndarray
        Значения ядра (нули для |u| > 1).

    Examples
    --------
    >>> epanechnikov_kernel(np.array([0.0, 0.5, 1.5]))
    array([0.75  , 0.5625, 0.    ])
    """
    result = 0.75 * (1 - u ** 2)
    result[np.abs(u) > 1] = 0.0
    return np.maximum(result, 0.0)


def triangular_kernel(u: np.ndarray) -> np.ndarray:
    """Треугольное ядро K(u) = (1 - |u|) * I(|u| <= 1).

    Parameters
    ----------
    u : np.ndarray
        Нормированные расстояния u = ||x - x_i|| / h.

    Returns
    -------
    np.ndarray
        Значения ядра (нули для |u| > 1).

    Examples
    --------
    >>> triangular_kernel(np.array([0.0, 0.5, 1.5]))
    array([1.  , 0.5 , 0.  ])
    """
    result = 1.0 - np.abs(u)
    result[np.abs(u) > 1] = 0.0
    return np.maximum(result, 0.0)


_KERNEL_MAP: dict[str, callable] = {
    "gaussian": gaussian_kernel,
    "epanechnikov": epanechnikov_kernel,
    "triangular": triangular_kernel,
}


def _compute_distances(X_query: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    """Вычисляет попарные евклидовы расстояния между точками.

    Parameters
    ----------
    X_query : np.ndarray, shape (m, d)
        Точки запроса.
    X_train : np.ndarray, shape (n, d)
        Обучающие точки.

    Returns
    -------
    np.ndarray, shape (m, n)
        Матрица расстояний dist[i, j] = ||X_query[i] - X_train[j]||.
    """
    # Используем broadcasting для эффективного вычисления
    diff = X_query[:, np.newaxis, :] - X_train[np.newaxis, :, :]  # (m, n, d)
    return np.sqrt(np.sum(diff ** 2, axis=-1))  # (m, n)


class NadarayaWatsonRegressor:
    """Оценщик Надарая–Ватсона с выбором ширины окна через кросс-валидацию.

    Оценивает условное математическое ожидание E[Y|X=x] с помощью
    взвешенного среднего обучающих меток, где веса определяются
    ядерной функцией расстояний.

    Формула:
        m(x) = sum_i K_h(||x - x_i||) * y_i / sum_i K_h(||x - x_i||)

    При нулевой сумме весов (точка далеко от всех обучающих примеров)
    возвращает среднее значение по обучающей выборке.

    Parameters
    ----------
    kernel : str, optional
        Тип ядра: 'gaussian' (по умолчанию), 'epanechnikov', 'triangular'.
    bandwidth : float, optional
        Ширина окна h > 0. По умолчанию 1.0.

    Attributes
    ----------
    X_train_ : np.ndarray
        Обучающие признаки после вызова fit().
    y_train_ : np.ndarray
        Обучающие метки после вызова fit().
    best_bandwidth_ : float или None
        Ширина окна после bandwidth_cv(); None до вызова.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.linspace(0, 5, 50).reshape(-1, 1)
    >>> y = np.sin(X.ravel()) + 0.1 * np.random.randn(50)
    >>> model = NadarayaWatsonRegressor(kernel='gaussian', bandwidth=0.5)
    >>> model.fit(X, y)
    >>> preds = model.predict(X)
    >>> preds.shape
    (50,)
    """

    def __init__(self, kernel: str = "gaussian", bandwidth: float = 1.0) -> None:
        if kernel not in _KERNEL_MAP:
            raise ValueError(f"Неизвестное ядро '{kernel}'. Допустимые: {list(_KERNEL_MAP.keys())}")
        if bandwidth <= 0:
            raise ValueError(f"Ширина окна должна быть > 0, получено {bandwidth}")
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.best_bandwidth_: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NadarayaWatsonRegressor":
        """Сохраняет обучающие данные (ленивое обучение).

        Ядерная регрессия — непараметрический метод, поэтому обучение
        сводится к запоминанию обучающей выборки.

        Parameters
        ----------
        X : np.ndarray, shape (n, d) или (n,)
            Обучающие признаки.
        y : np.ndarray, shape (n,)
            Целевые значения.

        Returns
        -------
        NadarayaWatsonRegressor
            Возвращает self для цепочки вызовов.
        """
        X_arr = np.array(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        self.X_train_ = X_arr
        self.y_train_ = np.array(y, dtype=float)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказывает значения для новых точек.

        Parameters
        ----------
        X : np.ndarray, shape (m, d) или (m,)
            Точки для предсказания.

        Returns
        -------
        np.ndarray, shape (m,)
            Предсказанные значения.

        Raises
        ------
        RuntimeError
            Если модель не обучена (fit не вызывался).
        """
        if self.X_train_ is None:
            raise RuntimeError("Модель не обучена. Вызовите fit() перед predict().")

        X_arr = np.array(X)
        X_query = X_arr.reshape(-1, 1) if X_arr.ndim == 1 else X_arr
        kernel_fn = _KERNEL_MAP[self.kernel]

        # Матрица расстояний (m, n)
        distances = _compute_distances(X_query, self.X_train_)

        # Нормированные расстояния
        u = distances / self.bandwidth

        # Веса
        weights = kernel_fn(u)  # (m, n)

        # Взвешенное среднее
        weight_sums = weights.sum(axis=1)  # (m,)
        fallback = np.mean(self.y_train_)  # среднее при нулевых весах

        predictions = np.where(
            weight_sums > 1e-12,
            (weights @ self.y_train_) / weight_sums,
            fallback,
        )
        return predictions

    def bandwidth_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: Union[KFold, int] = 5,
        bandwidths: Optional[np.ndarray] = None,
        random_state: int = DEFAULT_SEED,
    ) -> float:
        """Выбирает оптимальную ширину окна через кросс-валидацию (минимизация MSE).

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Признаки.
        y : np.ndarray, shape (n,)
            Целевые значения.
        cv : KFold или int, optional
            Стратегия кросс-валидации. По умолчанию KFold(5).
        bandwidths : np.ndarray, optional
            Сетка ширин окна. По умолчанию logspace(-2, 1, 20).
        random_state : int, optional
            Зерно генератора. По умолчанию 42.

        Returns
        -------
        float
            Оптимальная ширина окна (сохраняется в self.best_bandwidth_).
        """
        if bandwidths is None:
            bandwidths = DEFAULT_BANDWIDTHS.copy()

        if isinstance(cv, int):
            cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)

        best_h = bandwidths[0]
        best_mse = np.inf

        for h in bandwidths:
            self.bandwidth = h
            mse_scores = []

            for train_idx, val_idx in cv.split(X):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                self.fit(X_tr, y_tr)
                preds = self.predict(X_val)
                mse = np.mean((y_val - preds) ** 2)
                mse_scores.append(mse)

            mean_mse = np.mean(mse_scores)
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_h = h

        self.bandwidth = best_h
        self.best_bandwidth_ = best_h
        return best_h

    def get_params(self, deep: bool = True) -> dict:
        """Возвращает параметры для совместимости со sklearn API."""
        return {"kernel": self.kernel, "bandwidth": self.bandwidth}

    def set_params(self, **params) -> "NadarayaWatsonRegressor":
        """Устанавливает параметры для совместимости со sklearn API."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class LocalLinearRegressor:
    """Локально-линейная регрессия (Local Linear Regression).

    В каждой точке запроса x решает взвешенную МНК задачу:

        min_{a, b} sum_i K_h(||x - x_i||) * (y_i - a - b^T (x_i - x))^2

    Предсказание: m(x) = a* (константный член взвешенной регрессии).

    По сравнению с оценщиком Надарая–Ватсона устраняет смещение
    на границах области наблюдений.

    Parameters
    ----------
    kernel : str, optional
        Тип ядра: 'gaussian' (по умолчанию), 'epanechnikov', 'triangular'.
    bandwidth : float, optional
        Ширина окна h > 0. По умолчанию 1.0.

    Notes
    -----
    При малом bandwidth некоторые локальные задачи могут быть вырожденными
    (нет обучающих точек в окне). В этом случае возвращается глобальное среднее.

    Примечание производительности:
        Для каждой точки запроса решается отдельная задача МНК O(n * d^2).
        При n_query=222, n_train=177, d=15 — порядка n_query итераций,
        каждая с матрицей (n_train × d+1). Работает несколько секунд.
    """

    def __init__(self, kernel: str = "gaussian", bandwidth: float = 1.0) -> None:
        if kernel not in _KERNEL_MAP:
            raise ValueError(f"Неизвестное ядро '{kernel}'. Допустимые: {list(_KERNEL_MAP.keys())}")
        if bandwidth <= 0:
            raise ValueError(f"Ширина окна должна быть > 0, получено {bandwidth}")
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.best_bandwidth_: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LocalLinearRegressor":
        """Сохраняет обучающие данные (ленивое обучение).

        Parameters
        ----------
        X : np.ndarray, shape (n, d) или (n,)
        y : np.ndarray, shape (n,)

        Returns
        -------
        LocalLinearRegressor
        """
        X_arr = np.array(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        self.X_train_ = X_arr
        self.y_train_ = np.array(y, dtype=float)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказывает значения для новых точек.

        Parameters
        ----------
        X : np.ndarray, shape (m, d) или (m,)

        Returns
        -------
        np.ndarray, shape (m,)
        """
        if self.X_train_ is None:
            raise RuntimeError("Модель не обучена. Вызовите fit() перед predict().")

        X_arr = np.array(X)
        X_query = X_arr.reshape(-1, 1) if X_arr.ndim == 1 else X_arr
        kernel_fn = _KERNEL_MAP[self.kernel]
        n_train = self.X_train_.shape[0]
        d = self.X_train_.shape[1]
        fallback = np.mean(self.y_train_)

        predictions = np.zeros(len(X_query))

        for i, x in enumerate(X_query):
            # Расстояния от x до всех обучающих точек
            diffs = self.X_train_ - x  # (n_train, d)
            distances = np.sqrt(np.sum(diffs ** 2, axis=1))  # (n_train,)
            u = distances / self.bandwidth
            w = kernel_fn(u)  # (n_train,)

            if w.sum() < 1e-12:
                predictions[i] = fallback
                continue

            # Дизайн-матрица [1, (x_i - x)] размером (n_train, d+1)
            Z = np.hstack([np.ones((n_train, 1)), diffs])  # (n_train, d+1)

            # Взвешенная МНК: beta = (Z^T W Z)^{-1} Z^T W y
            W_diag = np.diag(w)
            ZtWZ = Z.T @ W_diag @ Z  # (d+1, d+1)
            ZtWy = Z.T @ W_diag @ self.y_train_  # (d+1,)

            try:
                beta = np.linalg.solve(ZtWZ, ZtWy)
                predictions[i] = beta[0]  # константный член = предсказание в x
            except np.linalg.LinAlgError:
                # Сингулярная матрица — возвращаем взвешенное среднее
                predictions[i] = np.dot(w, self.y_train_) / w.sum()

        return predictions

    def bandwidth_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: Union[KFold, int] = 5,
        bandwidths: Optional[np.ndarray] = None,
        random_state: int = DEFAULT_SEED,
    ) -> float:
        """Выбирает оптимальную ширину окна через кросс-валидацию.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
        y : np.ndarray, shape (n,)
        cv : KFold или int, optional
        bandwidths : np.ndarray, optional
            Сетка ширин окна. По умолчанию logspace(-2, 1, 20).
        random_state : int, optional

        Returns
        -------
        float
            Оптимальная ширина окна.
        """
        if bandwidths is None:
            bandwidths = DEFAULT_BANDWIDTHS.copy()

        if isinstance(cv, int):
            cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)

        best_h = bandwidths[0]
        best_mse = np.inf

        for h in bandwidths:
            self.bandwidth = h
            mse_scores = []

            for train_idx, val_idx in cv.split(X):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                self.fit(X_tr, y_tr)
                preds = self.predict(X_val)
                mse = np.mean((y_val - preds) ** 2)
                mse_scores.append(mse)

            mean_mse = np.mean(mse_scores)
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_h = h

        self.bandwidth = best_h
        self.best_bandwidth_ = best_h
        return best_h

    def get_params(self, deep: bool = True) -> dict:
        """Возвращает параметры для совместимости со sklearn API."""
        return {"kernel": self.kernel, "bandwidth": self.bandwidth}

    def set_params(self, **params) -> "LocalLinearRegressor":
        """Устанавливает параметры для совместимости со sklearn API."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
