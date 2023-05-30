import json
import os
import random
import shutil
import time

import pandas as pd
from h2o import h2o
from matplotlib import pyplot as plt
from genetic_algorithm import genetic_algorithm, calculate_fitness
from send_email import send_email

h2o.init(max_mem_size='16g')


def import_datasets():
    train_df = h2o.import_file('./datasets/train.csv')
    test_df = h2o.import_file('./datasets/test.csv')
    return train_df, test_df


def generate_kaggle_sample(chromosome, test_split_df, test_df, run_id):
    model = h2o.load_model(f'runs_history/{run_id}/best_model')

    print("VERIFY IF MATCHES WITH B: " + str(chromosome.fitness))
    print("VERIFY IF MATCHES WITH A: " + str(calculate_fitness(model, test_split_df)))
    print(model.show())

    ids_col = test_df.as_data_frame()['Id'].tolist()
    test_df = test_df.drop('Id')
    predictions = model.predict(test_df).as_data_frame()['predict'].tolist()
    predictions = [round(value) for value in predictions]
    df = pd.DataFrame({'Id': ids_col, 'label': predictions})
    df.to_csv(os.path.join(f'runs_history/{run_id}', 'kaggle_sample.csv'), index=False)


def generate_analysis(df, historic, chromosome, run_id, elapsed_time, conf):
    # fitness and metrics plot
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel("generation", fontsize=14)
    line1 = ax1.plot(df.generation, df.best_fitness, color='blue', marker='o', label='max fitness')
    line2 = ax1.plot(df.generation, df.best_accuracy, color='red', marker='o', label='max accuracy')
    line3 = ax1.plot(df.generation, df.best_precision, color='green', marker='o', label='max precision')
    line4 = ax1.plot(df.generation, df.best_recall, color='pink', marker='o', label='max recall')
    line5 = ax1.plot(df.generation, df.best_f1, color='yellow', marker='o', label='max f1')
    fig1.tight_layout()
    ax1.legend(handles=line1 + line2 + line3 + line4 + line5, loc='upper left')

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
    markdown_filename = os.path.join(f'runs_history/{run_id}', 'analysis.md')
    with open(markdown_filename, 'w') as file:
        file.write("# Genetic Algorithm Analysis\n\n")
        # overall metrics
        file.write(f"**Elapsed time:** {elapsed_time} seconds  \n")
        file.write(f"**Convergence speed:** {3} seconds  \n")
        file.write(f"**Elitism:** {False if conf['reproduction_specs']['keep_ratio'] == 0 else False}  \n")
        file.write(f"**Used dataset ratio:** {conf['model_training']['dataset_ratio']}  \n")
        file.write(f"**Train data ratio:** {conf['model_training']['train_data_ratio']}  \n")
        file.write(f"**Test data ratio:** {1 - conf['model_training']['train_data_ratio']}  \n")
        # charts
        file.write(f"### Metrics maximum value per generation \n\n{df.to_markdown()}\n\n")
        file.write(f"### Metrics maximum value per generation (line chart) \n\n![Plot](fitness_evolution.png)\n\n")
        file.write(f"### Fitness distribution \n\n![Plot](fitness_distribution.png)\n\n")
        file.write(f"### Model \n\n{df_chromosome.to_markdown()}\n\n")

        # Save the plots as image files
        plot1 = os.path.join(f'runs_history/{run_id}', 'fitness_evolution.png')
        fig1.savefig(plot1)
        plot2 = os.path.join(f'runs_history/{run_id}', 'fitness_distribution.png')
        fig2.savefig(plot2)


def main():
    # create runs_history_folder
    if not os.path.exists('runs_history'):
        os.makedirs('runs_history')

    # create run folder
    run_id = sum(os.path.isdir(os.path.join('runs_history', name)) for name in os.listdir('runs_history'))
    new_folder_path = os.path.join('runs_history', str(run_id))
    os.makedirs(new_folder_path)

    # create logs file
    os.makedirs(os.path.dirname(f'runs_history/{run_id}/run_logs.txt'), exist_ok=True)

    # import config
    with open('config.json', 'r') as file:
        conf = json.load(file)
        shutil.copy('config.json', f'runs_history/{run_id}')

    # import datasets
    train_df, test_df = import_datasets()
    train_df = train_df.drop('Id')

    # select only 50% of the train data
    train_df, not_used_df = train_df.split_frame(ratios=[conf["model_training"]["dataset_ratio"]], seed=random.randint(
        0, 1000))

    # train data test split
    train_split_df, test_split_df = train_df.split_frame(ratios=[conf["model_training"]["train_data_ratio"]],
                                                         seed=random.randint(0, 1000))

    # encode 'label' column as categorical for multinomial classification
    train_split_df['label'] = train_split_df['label'].asfactor()
    test_split_df['label'] = test_split_df['label'].asfactor()

    # call genetic algorithm
    start_time = time.time()
    df_analysis, historic = genetic_algorithm(train_split_df, test_split_df, conf, run_id)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # generate kaggle sample
    generate_kaggle_sample(historic[0], test_split_df, test_df, run_id)

    # generate analysis
    generate_analysis(df_analysis, historic, historic[0], run_id, elapsed_time, conf)

    # send email with best fitness
    send_email(historic[0].fitness)


if __name__ == "__main__":
    main()
