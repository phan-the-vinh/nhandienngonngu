# digits/fallback

`mlp_large_v2a_pre_digits_final.joblib` is the byte-identical fallback copy of
the former production model (`models/digits/production/mlp_large_v2a.joblib`).
SHA-256: `d64acc7bc44449c7186f70b221d123581f786efdbd53885edc2ae70bf0968ad3`.

The default digit model is now `models/digits/production/digits_final.joblib`.
To run the fallback without changing the default, pass its path with
`python run.py --mode digits --model-path models/digits/fallback/mlp_large_v2a_pre_digits_final.joblib`.
