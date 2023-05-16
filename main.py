import json
import os

import pandas as pd
from h2o import h2o
from h2o.estimators import H2ORandomForestEstimator, H2ODeepLearningEstimator
from matplotlib import pyplot as plt
from genetic_algorithm import genetic_algorithm

h2o.init()


def import_datasets():
    train_df = h2o.import_file('./datasets/train.csv')
    test_df = h2o.import_file('./datasets/test.csv')
    return train_df, test_df


def generate_kaggle_sample(chromosome, train_split_df, test_split_df, test_df):
    # train model and generate kaggle_sample.csv
    if chromosome.model_type == 'RF':
        model = H2ORandomForestEstimator(**chromosome.params)
    else:
        model = H2ODeepLearningEstimator(**chromosome.params)

    predictors = train_split_df.col_names[1:]
    response = train_split_df.col_names[0]
    model.train(x=predictors, y=response, training_frame=train_split_df, validation_frame=test_split_df)
    print(model.show())

    ids_col = test_df.as_data_frame()['Id'].tolist()
    test_df = test_df.drop('Id')
    predictions = model.predict(test_df).as_data_frame()['predict'].tolist()
    predictions = [round(value) for value in predictions]
    df = pd.DataFrame({'Id': ids_col, 'label': predictions})
    df.to_csv(os.path.join('./run_analysis', 'kaggle_sample.csv'), index=False)


def generate_analysis(df, historic, chromosome):
    # Create the "run_analysis" folder if it doesn't exist
    folder_path = 'run_analysis'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # fitness_evolution plot
    fig1, ax1 = plt.subplots()
    ax1.plot(df.generation, df.best_fitness, color='blue', marker='o')
    ax1.set_xlabel("generation", fontsize=14)
    ax1.set_ylabel("best fitness", color="blue", fontsize=14)

    # fitness_distribution plot
    fig2, ax2 = plt.subplots()
    ax2.hist(list(map(lambda x: x.fitness, historic)))
    ax2.set_xlabel("Fitness", fontsize=14)
    ax2.set_ylabel("Frequency", fontsize=14)

    # best model
    params = {key: str(value) for key, value in chromosome.params.items()}
    df_chromosome = pd.DataFrame(params, index=[0])
    df_chromosome['model_type'] = chromosome.model_type

    # Save the Markdown text to a file
    markdown_filename = os.path.join(folder_path, 'analysis.md')
    with open(markdown_filename, 'w') as file:
        file.write("# Genetic Algorithm Analysis \n\n")
        file.write(f"### Best Fitness p/Generation \n\n{df.to_markdown()}\n\n")
        file.write(f"### Fitness evolution\n\n![Plot](fitness_evolution.png)\n\n")
        file.write(f"### Distribution of fitness\n\n![Plot](fitness_distribution.png)\n\n")
        file.write(f"### Model \n\n{df_chromosome.to_markdown()}\n\n")

    # Save the plots as image files
    plot1 = os.path.join(folder_path, 'fitness_evolution.png')
    fig1.savefig(plot1)
    plot2 = os.path.join(folder_path, 'fitness_distribution.png')
    fig2.savefig(plot2)


def main():
    # import config
    with open('config.json', 'r') as file:
        conf = json.load(file)

    # import datasets
    train_df, test_df = import_datasets()
    train_df = train_df.drop('Id')

    # train test split
    train_split_df, test_split_df = train_df.split_frame(ratios=[.75], seed=1234)

    # call genetic algorithm
    df_analysis, historic = genetic_algorithm(train_split_df, test_split_df, conf)

    # generate kaggle sample
    generate_kaggle_sample(historic[0], train_split_df, test_split_df, test_df)

    # generate analysis
    generate_analysis(df_analysis, historic, historic[0])


if __name__ == "__main__":
    main()
