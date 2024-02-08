from thesis.mas import algorithm

"""
How to use

Ensure `line_profiler` is installe (`python -m pip install line_profiler`).
Decorate the functions you're interested in with `@profile` from `line_profiler`.
Run `kernprof -l thesis/profile.py` to profile the code.
Then run `python -m line_profiler profile.py.lprof` to see the results.
"""
algorithm.run(
    algorithm.MASConfig(
        model_name="solu-1l",
        sample_overlap=256,
        num_max_samples=32,
        sample_length_pre=192,
        sample_length_post=64,
        samples_to_check=256,
    )
)
