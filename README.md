# Machine Learning Institute - Week 6 - Fine tuning and rlhf

This week, we are experimenting with reinforcement learning on a large language model.

Using this blog post as a starting point, we will be experimenting with:

* Performing some fine tuning on a base model (to e.g. perform summarization)
* Attempting to train a reward model on top of a base model
* Performing RLHF using PPO and evaluating the fine tuning

# Set-up

* Install the [git lfs](https://git-lfs.com/) extension **before cloning this repository**
* Install the [uv package manager](https://docs.astral.sh/uv/getting-started/installation/)

Then install dependencies with:

```bash
uv sync --all-packages --dev
```

# Model Training

Run the following, with an optional `--model "model_name"` parameter

```bash
uv run -m model.start_train
```

# Run streamlit app

```bash
uv run streamlit run streamlit/app.py
```
