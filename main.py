import json
import os
import random
import shutil
import time

import pandas as pd
from h2o import h2o
from matplotlib import pyplot as plt
from genetic_algorithm import genetic_algorithm
from send_email import send_email

h2o.init(max_mem_size='16g')


def import_datasets():
    train_df = h2o.import_file('./datasets/train.csv')
    test_df = h2o.import_file('./datasets/test.csv')
    return train_df, test_df


def generate_kaggle_sample(test_df, run_id, model):
    ids_col = test_df.as_data_frame()['Id'].tolist()
    test_df = test_df.drop('Id')
    predictions = model.predict(test_df).as_data_frame()['predict'].tolist()
    predictions = [round(value) for value in predictions]
    df = pd.DataFrame({'Id': ids_col, 'label': predictions})
    df.to_csv(os.path.join(f'runs_history/{run_id}', 'kaggle_sample.csv'), index=False)


def metrics(historic, num_of_chromosomes_per_generation):
    avg_fitness = []
    rf_per_generation = []
    dl_per_generation = []
    gbm_per_generation = []

    fitness_sum = 0
    rf_counter = 0
    dl_counter = 0
    gbm_counter = 0
    chromosome_counter = 0
    generation = 0
    for index in range(len(historic)):
        chromosome_counter += 1
        fitness_sum += historic[index].fitness

        if historic[index].model_type == 'RF':
            rf_counter += 1
        elif historic[index].model_type == 'DL':
            dl_counter += 1
        else:
            gbm_counter += 1

        if chromosome_counter == num_of_chromosomes_per_generation:
            avg_fitness.append(fitness_sum / num_of_chromosomes_per_generation)
            rf_per_generation.append(rf_counter)
            dl_per_generation.append(dl_counter)
            gbm_per_generation.append(gbm_counter)
            fitness_sum = 0
            rf_counter = 0
            dl_counter = 0
            gbm_counter = 0
            chromosome_counter = 0
            generation += 1

    return avg_fitness, rf_per_generation, dl_per_generation, gbm_per_generation


def generate_analysis(df, historic, chromosome, run_id, elapsed_time, execution_time_per_generation, model,
                      train_split_df, conf):
    # Metrics maximum value per generation
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel("Generation", fontsize=14)
    line1 = ax1.plot(df.generation, df.best_fitness, color='blue', marker='o', label='max fitness')
    line2 = ax1.plot(df.generation, df.best_accuracy, color='red', marker='o', label='max accuracy')
    line3 = ax1.plot(df.generation, df.best_precision, color='green', marker='o', label='max precision')
    line4 = ax1.plot(df.generation, df.best_recall, color='pink', marker='o', label='max recall')
    line5 = ax1.plot(df.generation, df.best_f1, color='yellow', marker='o', label='max f1')
    fig1.tight_layout()
    ax1.legend(handles=line1 + line2 + line3 + line4 + line5, loc='upper left')

    # Fitness distribution
    fig2, ax2 = plt.subplots()
    ax2.hist(list(map(lambda x: x.fitness, historic)))
    ax2.set_xlabel("Fitness", fontsize=14)
    ax2.set_ylabel("Frequency", fontsize=14)

    avg_fitness, rf_per_generation, dl_per_generation, gbm_per_generation = metrics(historic,
                                                                                    conf['pop_specs']['pop_size'])

    # Average fitness per generation
    generations = range(len(avg_fitness))
    fig3, ax3 = plt.subplots()
    ax3.plot(generations, avg_fitness)
    ax3.set_xlabel("Generation", fontsize=14)
    ax3.set_ylabel("Average Fitness", fontsize=14)

    # Number of models of each type per generation
    fig4, ax4 = plt.subplots()
    ax4.set_xlabel("Generation", fontsize=14)
    line1 = ax4.plot(generations, rf_per_generation, color='blue', marker='o', label='random forest')
    line2 = ax4.plot(generations, dl_per_generation, color='red', marker='o', label='deep learning')
    line3 = ax4.plot(generations, gbm_per_generation, color='green', marker='o', label='gradient boost machine')
    fig4.tight_layout()
    ax4.legend(handles=line1 + line2 + line3, loc='upper left')

    # Execution time per generation
    fig5, ax5 = plt.subplots()
    ax5.plot(generations, execution_time_per_generation)
    ax5.set_xlabel("Generation", fontsize=14)
    ax5.set_ylabel("Execution time in seconds", fontsize=14)

    # Best model
    params = {key: str(value) for key, value in chromosome.params.items()}
    df_chromosome = pd.DataFrame(params, index=[0])
    df_chromosome['model_type'] = chromosome.model_type
    columns = ['model_type'] + list(df_chromosome.columns[:-1])
    df_chromosome = df_chromosome.reindex(columns=columns)

    # Logloss
    learning_curve = model.scoring_history()
    df1 = pd.DataFrame(learning_curve)

    # Save the Markdown text to a file
    markdown_filename = os.path.join(f'runs_history/{run_id}', 'analysis.md')
    with open(markdown_filename, 'w') as file:
        file.write("# Genetic Algorithm Analysis\n\n")
        # overall metrics
        file.write(f"**Elapsed time:** {round(elapsed_time, 2)} seconds  \n")
        file.write(f"**Convergence speed:** {3} seconds  \n")
        file.write(f"**Elitism:** {False if conf['reproduction_specs']['keep_ratio'] == 0 else True}  \n")
        file.write(f"**Used dataset ratio:** {conf['model_training']['dataset_ratio']}  \n")
        file.write(f"**Train data ratio:** {conf['model_training']['train_data_ratio']}  \n")
        file.write(f"**Test data ratio:** {1 - conf['model_training']['train_data_ratio']}  \n")
        # charts
        file.write(f"### Metrics maximum value per generation \n\n{df.to_markdown(index=False)}\n\n")
        file.write(
            f"### Metrics maximum value per generation (line chart) \n\n![Plot]("
            f"metrics_max_value_per_generations.png)\n\n")
        file.write(f"### Fitness distribution \n\n![Plot](fitness_distribution.png)\n\n")
        file.write(f"### Average fitness per generation \n\n![Plot](avg_fitness_per_generation.png)\n\n")
        file.write(f"### Number of models of each type per generation \n\n![Plot](models_types_per_generation.png)\n\n")
        file.write(f"### Execution time per generation \n\n![Plot](execution_time_per_generation.png)\n\n")
        # model
        file.write(f"# Best Solution Analysis\n\n{df_chromosome.to_markdown(index=False)}\n\n")

        # Save the plots as image files
        plot1 = os.path.join(f'runs_history/{run_id}', 'metrics_max_value_per_generations.png')
        fig1.savefig(plot1)
        plot2 = os.path.join(f'runs_history/{run_id}', 'fitness_distribution.png')
        fig2.savefig(plot2)
        plot3 = os.path.join(f'runs_history/{run_id}', 'avg_fitness_per_generation.png')
        fig3.savefig(plot3)
        plot4 = os.path.join(f'runs_history/{run_id}', 'models_types_per_generation.png')
        fig4.savefig(plot4)
        plot5 = os.path.join(f'runs_history/{run_id}', 'execution_time_per_generation.png')
        fig5.savefig(plot5)


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
    df_analysis, historic, execution_time_per_generation = genetic_algorithm(train_split_df, test_split_df, conf,
                                                                             run_id)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # get best model
    model = h2o.load_model(f'runs_history/{run_id}/best_model')

    # generate kaggle sample
    generate_kaggle_sample(test_df, run_id, model)

    # generate analysis
    generate_analysis(df_analysis, historic, historic[0], run_id, elapsed_time, execution_time_per_generation, model,
                      train_split_df, conf)

    # send email with best fitness
    send_email(historic[0].fitness)


if __name__ == "__main__":
    main()
