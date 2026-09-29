"""
gui_app.py  --  Tkinter front-end for the saved MovieLens-1M model.

The GUI only collects input and displays output; all model work happens in recommender.py.
Run:  python gui_app.py      (after train_model.py has produced ./artifacts)
"""
import sys
import tkinter as tk
from tkinter import messagebox, ttk

from recommender import MovieRecommender


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MovieLens-1M Recommender")
        self.geometry("1000x680")
        try:
            self.rec = MovieRecommender()
        except Exception as e:  # missing artifacts, corrupt files, ...
            messagebox.showerror("Cannot load model", str(e))
            self.destroy()
            sys.exit(1)
        self.matches = None
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)
        nb.add(self._recommend_tab(nb), text="  Recommendations  ")
        nb.add(self._predict_tab(nb), text="  Predict a rating  ")
        c = self.rec.config
        ttk.Label(self, text=f"Model: {c['model_name']}   |   test RMSE {c['metrics']['test_RMSE']:.3f}   "
                             f"MAE {c['metrics']['test_MAE']:.3f}", foreground="#555").pack(pady=(0, 6))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _tree(parent, cols, widths, height):
        tv = ttk.Treeview(parent, columns=[c for c, _ in cols], show="headings", height=height)
        for (c, head), w in zip(cols, widths):
            tv.heading(c, text=head)
            tv.column(c, width=w, anchor="w" if c in ("title", "genres") else "center")
        return tv

    def _read_user(self, var):
        try:
            uid = int(var.get())
        except ValueError:
            messagebox.showwarning("Invalid input", "User ID must be a whole number.")
            return None
        if not self.rec.has_user(uid):
            lo, hi = self.rec.user_id_range
            messagebox.showwarning("Unknown user", f"User ID must be between {lo} and {hi}.")
            return None
        return uid

    # ------------------------------------------------------------------ tab 1
    def _recommend_tab(self, nb):
        f = ttk.Frame(nb, padding=10)
        top = ttk.Frame(f); top.pack(fill="x")
        lo, hi = self.rec.user_id_range
        self.uid1, self.topn, self.genre, self.minr = (tk.StringVar(value=str(lo)), tk.StringVar(value="10"),
                                                       tk.StringVar(value="Any"), tk.StringVar(value="20"))
        for i, (label, w) in enumerate([("User ID", ttk.Spinbox(top, from_=lo, to=hi, textvariable=self.uid1, width=8)),
                                        ("Top N", ttk.Spinbox(top, from_=1, to=50, textvariable=self.topn, width=5)),
                                        ("Genre", ttk.Combobox(top, values=["Any"] + self.rec.genres,
                                                               textvariable=self.genre, width=14, state="readonly")),
                                        ("Min ratings", ttk.Spinbox(top, from_=1, to=500, textvariable=self.minr, width=6))]):
            ttk.Label(top, text=label).grid(row=0, column=2 * i, padx=(8, 2))
            w.grid(row=0, column=2 * i + 1)
        ttk.Button(top, text="Recommend", command=self.on_recommend).grid(row=0, column=8, padx=12)
        self.profile_lbl = ttk.Label(f, text="Enter a user ID and press Recommend.", font=("TkDefaultFont", 10, "bold"))
        self.profile_lbl.pack(anchor="w", pady=(10, 4))

        ttk.Label(f, text="Recommended (movies this user has not rated)").pack(anchor="w")
        self.reco_tv = self._tree(f, [("rank", "#"), ("title", "Title"), ("year", "Year"), ("genres", "Genres"),
                                      ("pred", "Predicted"), ("pop", "# ratings")], [40, 300, 60, 300, 80, 80], 12)
        self.reco_tv.pack(fill="both", expand=True)
        ttk.Label(f, text="For context: this user's highest-rated movies").pack(anchor="w", pady=(10, 0))
        self.fav_tv = self._tree(f, [("title", "Title"), ("year", "Year"), ("genres", "Genres"), ("rating", "Rating")],
                                 [340, 60, 340, 80], 7)
        self.fav_tv.pack(fill="both", expand=True)
        self.bind("<Return>", lambda e: self.on_recommend())
        return f

    def on_recommend(self):
        uid = self._read_user(self.uid1)
        if uid is None:
            return
        try:
            n, minr = int(self.topn.get()), int(self.minr.get())
        except ValueError:
            messagebox.showwarning("Invalid input", "Top N and Min ratings must be whole numbers.")
            return
        p = self.rec.user_profile(uid)
        self.profile_lbl.config(text=f"User {uid}: {p['gender']}, {p['age_group']}, {p['occupation']}, "
                                     f"{p['n_ratings']} ratings")
        for tv in (self.reco_tv, self.fav_tv):
            tv.delete(*tv.get_children())
        recs = self.rec.recommend(uid, n=n, genre=self.genre.get(), min_ratings=minr)
        if recs.empty:
            messagebox.showinfo("No results", "No movies match these filters.")
        for i, r in recs.iterrows():
            self.reco_tv.insert("", "end", values=(i + 1, r.Title, r.Year, r.Genres.replace("|", ", "),
                                                   f"{r.Predicted:.2f}", int(r.train_count)))
        for _, r in self.rec.user_top_rated(uid).iterrows():
            self.fav_tv.insert("", "end", values=(r.Title, r.Year, r.Genres.replace("|", ", "), int(r.Rating)))

    # ------------------------------------------------------------------ tab 2
    def _predict_tab(self, nb):
        f = ttk.Frame(nb, padding=10)
        top = ttk.Frame(f); top.pack(fill="x")
        lo, hi = self.rec.user_id_range
        self.uid2, self.query = tk.StringVar(value=str(lo)), tk.StringVar()
        ttk.Label(top, text="User ID").grid(row=0, column=0, padx=(0, 2))
        ttk.Spinbox(top, from_=lo, to=hi, textvariable=self.uid2, width=8).grid(row=0, column=1)
        ttk.Label(top, text="Movie title contains").grid(row=0, column=2, padx=(16, 2))
        e = ttk.Entry(top, textvariable=self.query, width=30); e.grid(row=0, column=3)
        e.bind("<Return>", lambda ev: (self.on_search(), "break")[1])
        ttk.Button(top, text="Search", command=self.on_search).grid(row=0, column=4, padx=8)

        self.search_tv = self._tree(f, [("title", "Title"), ("year", "Year"), ("genres", "Genres"),
                                        ("avg", "Avg rating"), ("pop", "# ratings")], [340, 60, 300, 90, 80], 12)
        self.search_tv.pack(fill="both", expand=True, pady=10)
        ttk.Button(f, text="Predict rating for selected movie", command=self.on_predict).pack()
        self.result_lbl = ttk.Label(f, text="", font=("TkDefaultFont", 20, "bold"))
        self.result_lbl.pack(pady=(14, 2))
        self.detail_lbl = ttk.Label(f, text="", justify="center")
        self.detail_lbl.pack()
        return f

    def on_search(self):
        self.search_tv.delete(*self.search_tv.get_children())
        self.matches = self.rec.search_movies(self.query.get())
        if self.matches.empty:
            messagebox.showinfo("No results", "No movie title matches that text.")
            return
        for i, r in self.matches.iterrows():
            self.search_tv.insert("", "end", iid=str(i), values=(r.Title, r.Year, r.Genres.replace("|", ", "),
                                                                 f"{r.avg_rating:.2f}" if r.avg_rating == r.avg_rating else "-",
                                                                 int(r.train_count)))

    def on_predict(self):
        uid = self._read_user(self.uid2)
        if uid is None:
            return
        sel = self.search_tv.selection()
        if not sel or self.matches is None:
            messagebox.showinfo("Select a movie", "Search for a movie and select it in the list first.")
            return
        row = self.matches.iloc[int(sel[0])]
        pred = self.rec.predict_rating(uid, row.MovieID)
        actual = self.rec.actual_rating(uid, row.MovieID)
        stars = "★" * int(round(pred)) + "☆" * (5 - int(round(pred)))
        self.result_lbl.config(text=f"{pred:.2f} / 5   {stars}")
        self.detail_lbl.config(text=f"{row.Title} ({row.Year}) for user {uid}\n" +
                                    (f"This user actually rated it {actual}." if actual is not None
                                     else "This user has not rated this movie."))


if __name__ == "__main__":
    App().mainloop()