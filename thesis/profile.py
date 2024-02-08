from thesis.mas import algorithm

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
