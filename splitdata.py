import logging
from pathlib import Path

import pandas as pd


SEED = 42
TEST_FRACTION = 0.2
DEMOGRAPHIC_COLUMNS = ("race", "sex", "gender", "age_cat")


def log_split_comparison(train: pd.DataFrame, test: pd.DataFrame) -> None:
	logger = logging.getLogger(__name__)
	logger.info("Split row counts: train=%d, test=%d", len(train), len(test))

	columns = [column for column in DEMOGRAPHIC_COLUMNS if column in train.columns]
	if not columns:
		logger.warning("No recognized demographic columns found.")
	for column in columns:
		train_dist = train[column].fillna("<missing>").value_counts(normalize=True)
		test_dist = test[column].fillna("<missing>").value_counts(normalize=True)
		comparison = pd.concat(
			[train_dist.rename("train_pct"), test_dist.rename("test_pct")], axis=1
		).fillna(0)
		logger.info(
			"%s distribution (percent):\n%s", column, (comparison * 100).round(2)
		)

	if "score_text" not in train.columns:
		logger.warning("Column 'score_text' not found.")
		return

	train_dist = train["score_text"].value_counts(normalize=True, dropna=False)
	test_dist = test["score_text"].value_counts(normalize=True, dropna=False)
	score_comparison = pd.concat(
		[train_dist.rename("train_pct"), test_dist.rename("test_pct")], axis=1
	).fillna(0)
	logger.info(
		"score_text distribution (percent):\n%s",
		(score_comparison * 100).round(2),
	)


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
	logger = logging.getLogger(__name__)
	root = Path(__file__).resolve().parent
	source = root / "raw_data" / "cox-violent-parsed.csv"
	output_dir = root / "data"

	if not source.is_file():
		raise FileNotFoundError(f"Input CSV not found: {source}")
	data = pd.read_csv(source)
	if len(data) < 2:
		raise ValueError("At least two rows are required for a train/test split.")

	test = data.sample(frac=TEST_FRACTION, random_state=SEED)
	train = data.drop(index=test.index)
	output_dir.mkdir(parents=True, exist_ok=True)
	train_path = output_dir / "train.csv"
	test_path = output_dir / "test.csv"
	train.to_csv(train_path, index=False)
	test.to_csv(test_path, index=False)

	logger.info("Created reproducible split using seed=%d.", SEED)
	logger.info("Saved train file: %s", train_path)
	logger.info("Saved test file: %s", test_path)
	log_split_comparison(train, test)


if __name__ == "__main__":
	main()
