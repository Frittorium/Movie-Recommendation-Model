import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


class HybridRecommender(nn.Module):
    def __init__(self, n_users, n_movies, n_user_num, n_movie_num, emb_dim=32, hidden=128, dropout=0.2,
                 n_occ=21, n_reg=11):
        super().__init__()
        self.register_buffer("user_cat", torch.zeros(n_users, 2, dtype=torch.long))
        self.register_buffer("user_num", torch.zeros(n_users, n_user_num))
        self.register_buffer("movie_num", torch.zeros(n_movies, n_movie_num))
        self.register_buffer("global_mean", torch.tensor(0.0))
        self.user_emb, self.movie_emb = nn.Embedding(n_users, emb_dim), nn.Embedding(n_movies, emb_dim)
        self.user_bias, self.movie_bias = nn.Embedding(n_users, 1), nn.Embedding(n_movies, 1)
        self.occ_emb, self.reg_emb = nn.Embedding(n_occ, 8), nn.Embedding(n_reg, 4)
        in_dim = 2 * emb_dim + 8 + 4 + n_user_num + n_movie_num
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden // 2, 1))

    def forward(self, u, m):
        ue, me = self.user_emb(u), self.movie_emb(m)
        cat = self.user_cat[u]
        x = torch.cat([ue, me, self.occ_emb(cat[:, 0]), self.reg_emb(cat[:, 1]), self.user_num[u], self.movie_num[m]], 1)
        return (self.global_mean + self.user_bias(u).squeeze(-1) + self.movie_bias(m).squeeze(-1)
                + (ue * me).sum(1) + self.mlp(x).squeeze(-1))


class MovieRecommender:
    def __init__(self, art_dir=ART_DIR):
        needed = ["model.pt", "model_config.json", "movies_lookup.csv", "users_lookup.csv", "ratings_clean.csv.gz"]
        missing = [f for f in needed if not os.path.exists(os.path.join(art_dir, f))]
        if missing:
            raise FileNotFoundError(f"Missing artifacts in '{art_dir}': {missing}. Run train_model.py first.")
        self.config = json.load(open(os.path.join(art_dir, "model_config.json")))
        c = self.config
        self.model = HybridRecommender(c["n_users"], c["n_movies"], c["n_user_num"], c["n_movie_num"],
                                       c["emb_dim"], c["hidden"], c["dropout"])
        self.model.load_state_dict(torch.load(os.path.join(art_dir, "model.pt"), map_location="cpu"))
        self.model.eval()

        self.movies = pd.read_csv(os.path.join(art_dir, "movies_lookup.csv"))
        self.movies["GenreList"] = self.movies.Genres.str.split("|")
        self.users = pd.read_csv(os.path.join(art_dir, "users_lookup.csv")).set_index("UserID")
        ratings = pd.read_csv(os.path.join(art_dir, "ratings_clean.csv.gz"))
        self._user_ratings = {u: g.set_index("MovieID").Rating for u, g in ratings.groupby("UserID")}
        self._mid2row = {m: i for i, m in enumerate(self.movies.MovieID)}
        self.genres = sorted({g for gl in self.movies.GenreList for g in gl})

    # ------------------------------------------------------------------ helpers
    @property
    def user_id_range(self):
        return int(self.users.index.min()), int(self.users.index.max())

    def has_user(self, user_id):
        return int(user_id) in self.users.index

    def user_profile(self, user_id):
        r = self.users.loc[int(user_id)]
        return {"gender": "Male" if r.Gender == "M" else "Female", "age_group": r.AgeGroup,
                "occupation": r.OccupationName, "n_ratings": int(r.n_ratings)}

    @torch.no_grad()
    def _score(self, user_id, movie_idx):
        u_idx = int(self.users.loc[int(user_id), "u_idx"])
        m = torch.as_tensor(np.asarray(movie_idx), dtype=torch.long)
        u = torch.full_like(m, u_idx)
        return self.model(u, m).clamp(1, 5).numpy()  # ratings live on a 1-5 scale

    # ------------------------------------------------------------------ public API
    def predict_rating(self, user_id, movie_id):
        """Predicted 1-5 rating for one (user, movie) pair."""
        row = self._mid2row[int(movie_id)]
        return float(self._score(user_id, [self.movies.m_idx.iat[row]])[0])

    def actual_rating(self, user_id, movie_id):
        """The rating the user really gave (or None)."""
        r = self._user_ratings.get(int(user_id))
        return None if r is None or int(movie_id) not in r.index else int(r.loc[int(movie_id)])

    def recommend(self, user_id, n=10, genre=None, min_ratings=20):
        """Top-n unseen movies by predicted rating. `min_ratings` hides films with too little evidence."""
        df = self.movies.copy()
        df["Predicted"] = self._score(user_id, df.m_idx.values)
        seen = self._user_ratings.get(int(user_id), pd.Series(dtype=float)).index
        df = df[~df.MovieID.isin(seen) & (df.train_count >= min_ratings)]
        if genre and genre != "Any":
            df = df[df.GenreList.apply(lambda g: genre in g)]
        return df.sort_values("Predicted", ascending=False).head(n)[
            ["MovieID", "Title", "Year", "Genres", "Predicted", "train_count"]].reset_index(drop=True)

    def user_top_rated(self, user_id, n=8):
        """Movies this user rated highest (context for judging the recommendations)."""
        r = self._user_ratings.get(int(user_id))
        if r is None:
            return pd.DataFrame(columns=["Title", "Year", "Genres", "Rating"])
        # ties are broken by movie popularity so the list is stable and sensible
        df = self.movies.set_index("MovieID").loc[r.index].assign(Rating=r.values)
        return df.sort_values(["Rating", "train_count"], ascending=False).head(n)[
            ["Title", "Year", "Genres", "Rating"]].reset_index(drop=True)

    def search_movies(self, query, limit=100):
        q = query.strip().lower()
        if not q:
            return self.movies.head(0)
        hit = self.movies[self.movies.Title.str.lower().str.contains(q, regex=False)]
        return hit.sort_values("train_count", ascending=False).head(limit)[
            ["MovieID", "Title", "Year", "Genres", "avg_rating", "train_count"]].reset_index(drop=True)


if __name__ == "__main__":
    rec = MovieRecommender()
    print("Model:", rec.config["model_name"], "| test RMSE:", round(rec.config["metrics"]["test_RMSE"], 4))
    uid = rec.user_id_range[0]
    print("User", uid, rec.user_profile(uid))
    print(rec.recommend(uid, n=5).to_string(index=False))