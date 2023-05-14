from h2o import h2o
from genetic_algorithm import genetic_algorithm

h2o.init()


def import_datasets():
    train_df = h2o.import_file('./datasets/train.csv')
    test_df = h2o.import_file('./datasets/test.csv')
    return train_df, test_df


def main():
    # import datasets
    train_df, test_df = import_datasets()
    train_df = train_df.drop('Id')

    # train test split
    train_split_df, test_split_df = train_df.split_frame(ratios=[.75], seed=1234)

    # call genetic algorithm
    genetic_algorithm(train_split_df, test_split_df)


if __name__ == "__main__":
    main()
