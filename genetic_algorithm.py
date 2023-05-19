import copy
import random
import pandas
from h2o import h2o
from utils import convert_percentages_to_values, compute_class_weights, generate_random_value_for_param, save_model, \
    write_log, generate_second_crossover_point
from h2o.estimators import H2ORandomForestEstimator, H2ODeepLearningEstimator, H2OGradientBoostingEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Chromosome:
    def __init__(self, model_type, params):
        self.model_type = copy.deepcopy(model_type)
        self.params = copy.deepcopy(params)
        self.fitness = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None

    def __eq__(self, other):
        if len(self.params) != len(other.params):
            return False
        if self.model_type != other.model_type:
            return False

        for i in range(len(self.params)):
            if self.params[i] != other.params[i]:
                return False
        return True

    def __str__(self):
        return str({'model_type': self.model_type, 'params': self.params}) + ' | Fitness: ' + str(self.fitness)


def init_pop(conf):
    pop = []
    algo1, algo2, algo3 = convert_percentages_to_values(conf['pop_specs']['pop_size'], conf['pop_specs']['RF_pop_rate'],
                                                        conf['pop_specs']['DL_pop_rate'],
                                                        conf['pop_specs']['GBM_pop_rate'])
    # RF
    for i in range(algo1):
        params = {'seed': random.randint(0, 1000000)}
        for param in conf['RF'].keys():
            params[param] = generate_random_value_for_param(conf['RF'][param])
        pop.append(Chromosome('RF', params))

    # DL
    for i in range(algo2):
        params = {'seed': random.randint(0, 1000000)}
        for param in conf['DL'].keys():
            params[param] = generate_random_value_for_param(conf['DL'][param])
        pop.append(Chromosome('DL', params))

    # GBM
    for i in range(algo3):
        params = {'seed': random.randint(0, 1000000)}
        for param in conf['GBM'].keys():
            params[param] = generate_random_value_for_param(conf['GBM'][param])
        pop.append(Chromosome('GBM', params))

    return pop


def train_model(chromosome, train_df, test_df):
    predictors = train_df.col_names[1:]
    response = train_df.col_names[0]

    if chromosome.model_type == 'RF':
        model = H2ORandomForestEstimator(**chromosome.params)
    elif chromosome.model_type == 'DL':
        model = H2ODeepLearningEstimator(**chromosome.params)
    else:
        model = H2OGradientBoostingEstimator(**chromosome.params)

    model.train(x=predictors, y=response, training_frame=train_df, validation_frame=test_df)
    return model


def calculate_fitness(model, test_df):
    true_values = test_df.as_data_frame()['label'].tolist()
    predictions = model.predict(test_df).as_data_frame()['predict'].tolist()

    # Calculate class weights based on class imbalance
    class_weights = compute_class_weights(true_values)

    # Calculate weighted metrics
    accuracy = accuracy_score(true_values, predictions, sample_weight=class_weights)
    precision = precision_score(true_values, predictions, average='weighted', sample_weight=class_weights,
                                zero_division=0)
    recall = recall_score(true_values, predictions, average='weighted', sample_weight=class_weights, zero_division=0)
    f1 = f1_score(true_values, predictions, average='weighted', sample_weight=class_weights, zero_division=0)

    # Calculate fitness score
    fitness_score = (accuracy + precision + recall + f1) / 4.0
    return fitness_score, accuracy, precision, recall, f1


def fitness(pop, train_df, test_df, best_fitness, run_id):
    for chromosome in pop:
        if chromosome.fitness is None:
            # noinspection PyBroadException
            try:
                model = train_model(chromosome, train_df, test_df)
                fitness_score, accuracy, precision, recall, f1 = calculate_fitness(model, test_df)
                chromosome.fitness = fitness_score
                chromosome.accuracy = accuracy
                chromosome.precision = precision
                chromosome.recall = recall
                chromosome.f1 = f1

                if chromosome.fitness > best_fitness:
                    best_fitness = chromosome.fitness
                    save_model(model, run_id)

                h2o.remove(model)
            except Exception:
                chromosome.fitness = 0
                chromosome.accuracy = 0
                chromosome.precision = 0
                chromosome.recall = 0
                chromosome.f1 = 0

        write_log(f'----- {chromosome.model_type}: {chromosome.fitness} -----', run_id)
        write_log(f'accuracy: {chromosome.accuracy}', run_id)
        write_log(f'precision: {chromosome.accuracy}', run_id)
        write_log(f'recall: {chromosome.accuracy}', run_id)
        write_log(f'f1: {chromosome.accuracy}', run_id)
    return best_fitness


def select(pop):
    pop.sort(key=lambda sol: sol.fitness, reverse=True)
    return pop[:int(round(len(pop) / 2, 0))]


def mutate(chromosome, conf):
    new_chromosome = copy.deepcopy(chromosome)

    if conf['reproduction_specs']['randomize_num_of_genes_to_mutate']:
        genes = list(new_chromosome.params.keys())
        genes_to_mutate = random.sample(genes, random.randint(1, len(genes)))

        for gene in genes_to_mutate:
            if gene == 'seed':
                new_chromosome.params['seed'] = random.randint(0, 1000000)
            else:
                new_chromosome.params[gene] = generate_random_value_for_param(conf[new_chromosome.model_type][gene])
    else:
        gene_to_mutate = random.choice(list(new_chromosome.params.keys()))
        if gene_to_mutate == 'seed':
            new_chromosome.params['seed'] = random.randint(0, 1000000)
        else:
            new_chromosome.params[gene_to_mutate] = generate_random_value_for_param(
                conf[new_chromosome.model_type][gene_to_mutate])

    return new_chromosome


def crossover(chromosome_a, chromosome_b):
    num_of_genes = len(chromosome_a.params)
    params = list(chromosome_a.params.keys())
    num_of_crossover_points = random.randint(1, 2)

    if num_of_crossover_points == 2:
        point_a = random.randint(2, num_of_genes - 1)
        # noinspection PyBroadException
        try:
            point_b = generate_second_crossover_point(num_of_genes, point_a)
        except Exception:
            point_b = num_of_genes

        child_a = copy.deepcopy(chromosome_a)
        child_b = copy.deepcopy(chromosome_b)

        for i in range(0, point_a):
            child_a.params[params[i]] = chromosome_b.params[params[i]]
            child_b.params[params[i]] = chromosome_a.params[params[i]]

        for i in range(point_a, point_b):
            child_a.params[params[i]] = chromosome_a.params[params[i]]
            child_b.params[params[i]] = chromosome_b.params[params[i]]

        if point_b != num_of_genes:
            for i in range(point_b, num_of_genes):
                child_a.params[params[i]] = chromosome_b.params[params[i]]
                child_b.params[params[i]] = chromosome_a.params[params[i]]

        return child_a, child_b
    else:
        point = random.randint(2, num_of_genes - 1)
        child_a = copy.deepcopy(chromosome_a)
        child_b = copy.deepcopy(chromosome_b)

        for i in range(0, point):
            child_a.params[params[i]] = chromosome_b.params[params[i]]
            child_b.params[params[i]] = chromosome_a.params[params[i]]

        for i in range(point, num_of_genes):
            child_a.params[params[i]] = chromosome_a.params[params[i]]
            child_b.params[params[i]] = chromosome_b.params[params[i]]

        return child_a, child_b


def reproduce(new_pop, selected, mutation, _crossover, conf):
    while mutation != 0:
        mutation -= 1
        mutated = mutate(random.choice(selected), conf)
        mutated.fitness = None
        new_pop.append(mutated)

    while _crossover > 0:
        _crossover -= 2

        parents = random.sample(selected, 2)
        while parents[0].model_type != parents[1].model_type:
            parents = random.sample(selected, 2)

        child_a, child_b = crossover(parents[0], parents[1])
        child_a.fitness = None
        child_b.fitness = None

        if _crossover >= 0:
            new_pop.append(child_a)
            new_pop.append(child_b)
        elif random.randint(0, 1) == 1:
            new_pop.append(child_a)
        else:
            new_pop.append(child_b)


def genetic_algorithm(train_df, test_df, conf, run_id):
    pop = init_pop(conf)
    historic = []
    best_fitness = 0
    df = pandas.DataFrame(columns=['generation', 'best_fitness'])
    to_keep, mutation, _crossover = convert_percentages_to_values(len(pop), conf['reproduction_specs']['keep_rate'],
                                                                  conf['reproduction_specs']['mutation_rate'],
                                                                  conf['reproduction_specs']['crossover_rate'])

    for i in range(0, conf['num_of_gens']):
        write_log(f'### Generation {str(i + 1)}', run_id)
        historic = historic + pop
        best_fitness = fitness(pop, train_df, test_df, best_fitness, run_id)
        pop.sort(key=lambda chromosome: chromosome.fitness, reverse=True)

        write_log(f'### Best Fitness for Generation {str(i + 1)}: {str(pop[0].fitness)}', run_id)

        row = pandas.DataFrame(
            {'generation': [i], 'best_fitness': [pop[0].fitness], 'best_accuracy': [pop[0].accuracy],
             'best_precision': [pop[0].precision], 'best_recall': [pop[0].recall], 'best_f1': [pop[0].f1]})
        df = pandas.concat([df, row])

        new_pop = pop[:to_keep]
        selected = select(pop)
        reproduce(new_pop, selected, mutation, _crossover, conf)
        pop = new_pop

    historic.sort(key=lambda chromosome: chromosome.fitness, reverse=True)
    return df, historic
