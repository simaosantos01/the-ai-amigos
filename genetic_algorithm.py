import copy
import random
from utils import convert_percentages_to_values, compute_class_weights, generate_random_value_for_param
from h2o.estimators import H2ORandomForestEstimator, H2ODeepLearningEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

conf = {
    'num_of_gens': 50,
    'pop_specs': {
        'pop_size': 100,
        'RF_pop_rate': 50,
        'DL_pop_rate': 50,
    },
    'reproduction_specs': {
        'keep_rate': 80,
        'mutation_rate': 10,
        'randomize_num_of_genes_to_mutate': False,
        'crossover_rate': 10,
        'crossover_points': [
            20, 80
        ]
    },
    'RF': {
        'ntrees': {
            'min_value': 10,
            'max_value': 50,
            'type': 'integer'
        },
        'max_depth': {
            'min_value': 0,
            'max_value': 20,
            'type': 'integer'
        },
        'min_rows': {
            'min_value': 1,
            'max_value': 20,
            'type': 'integer'
        },
        'sample_rate': {
            'min_value': 0,
            'max_value': 1,
            'type': 'float'
        },
        'col_sample_rate_per_tree': {
            'min_value': 0,
            'max_value': 1,
            'type': 'float'
        }
    },
    'DL': {
        'activation': [
            'tanh',
            'tanh_with_dropout',
            'rectifier',
            'rectifier_with_dropout',
            'maxout',
            'maxout_with_dropout'
        ],
        'hidden': [
            [100, 100],
            [200, 200]
        ],
        'input_dropout_ratio': [
            0,
            0.1,
            0.2
        ],
        'l1': [
            0,
            1e-5
        ],
        'l2': [
            0,
            1e-5
        ],
        'epochs': {
            'min_value': 10,
            'max_value': 1000,
            'type': 'integer'
        }
    }
}


class Chromosome:
    def __init__(self, model_type, params):
        self.model_type = copy.deepcopy(model_type)
        self.params = copy.deepcopy(params)
        self.fitness = None

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


def init_pop():
    pop = []
    algo1, algo2, algo3 = convert_percentages_to_values(conf['pop_specs']['pop_size'], conf['pop_specs']['RF_pop_rate'],
                                                        conf['pop_specs']['DL_pop_rate'], 0)
    # RF
    for i in range(algo1):
        params = {}
        for param in conf['RF'].keys():
            params[param] = generate_random_value_for_param(conf['RF'][param])
        pop.append(Chromosome('RF', params))

    # DL
    for i in range(algo2):
        params = {}
        for param in conf['DL'].keys():
            params[param] = generate_random_value_for_param(conf['DL'][param])
        pop.append(Chromosome('DL', params))

    return pop


def calculate_fitness(chromosome, train_df, test_df):
    # predictors = train_df.col_names[1:]
    # response = train_df.col_names[0]

    # if chromosome.model_type == 'RF':
    #    model = H2ORandomForestEstimator(**chromosome.params)
    # else:
    #     model = H2ODeepLearningEstimator(**chromosome.params)

    # model.train(x=predictors, y=response, training_frame=train_df, validation_frame=test_df)

    # true_values = test_df.as_data_frame()['label'].tolist()
    # predictions = model.predict(test_df).as_data_frame()['predict'].tolist()
    # predictions = [round(value) for value in predictions]

    # Calculate class weights based on class imbalance
    # class_weights = compute_class_weights(true_values)

    # Calculate weighted metrics
    # accuracy = accuracy_score(true_values, predictions, sample_weight=class_weights)
    # precision = precision_score(true_values, predictions, average='weighted', sample_weight=class_weights)
    # recall = recall_score(true_values, predictions, average='weighted', sample_weight=class_weights)
    # f1 = f1_score(true_values, predictions, average='weighted', sample_weight=class_weights)

    # Calculate weighted fitness score
    # fitness_score = (accuracy + precision + recall + f1) / 4.0
    # return fitness_score
    return random.randint(0, 100)


def fitness(pop, train_df, test_df):
    for chromosome in pop:
        chromosome.fitness = calculate_fitness(chromosome, train_df, test_df)


def select(pop):
    pop.sort(key=lambda sol: sol.fitness, reverse=True)
    return pop[:int(round(len(pop) / 2, 0))]


def mutate(chromosome):
    new_chromosome = copy.deepcopy(chromosome)

    if conf['reproduction_specs']['randomize_num_of_genes_to_mutate']:
        genes = list(new_chromosome.params.keys())
        genes_to_mutate = random.sample(genes, random.randint(1, len(genes)))

        for gene in genes_to_mutate:
            new_chromosome.params[gene] = generate_random_value_for_param(conf[new_chromosome.model_type][gene])
    else:
        gene_to_mutate = random.choice(list(new_chromosome.params.keys()))
        new_chromosome.params[gene_to_mutate] = generate_random_value_for_param(
            conf[new_chromosome.model_type][gene_to_mutate])

    return new_chromosome


def crossover(chromosome_a, chromosome_b):
    num_of_genes = len(chromosome_a.params)
    params = list(chromosome_a.params.keys())

    if 0 <= 1 < len(conf['reproduction_specs']['crossover_points']):
        point_a = int((conf['reproduction_specs']['crossover_points'][0] * num_of_genes / 100))
        point_b = int((conf['reproduction_specs']['crossover_points'][1] * num_of_genes / 100))
        child_a = copy.deepcopy(chromosome_a)
        child_b = copy.deepcopy(chromosome_b)

        for i in range(0, point_a):
            child_a.params[params[i]] = chromosome_b.params[params[i]]
            child_b.params[params[i]] = chromosome_a.params[params[i]]

        for i in range(point_a, point_b):
            child_a.params[params[i]] = chromosome_a.params[params[i]]
            child_b.params[params[i]] = chromosome_b.params[params[i]]

        for i in range(point_b, num_of_genes):
            child_a.params[params[i]] = chromosome_b.params[params[i]]
            child_b.params[params[i]] = chromosome_a.params[params[i]]

        return child_a, child_b
    else:
        point = int((conf['reproduction_specs']['crossover_points'][0] * num_of_genes / 100))
        child_a = copy.deepcopy(chromosome_a)
        child_b = copy.deepcopy(chromosome_b)

        for i in range(0, point):
            child_a.params[params[i]] = chromosome_b.params[params[i]]
            child_b.params[params[i]] = chromosome_a.params[params[i]]

        for i in range(point, num_of_genes):
            child_a.params[params[i]] = chromosome_a.params[params[i]]
            child_b.params[params[i]] = chromosome_b.params[params[i]]

        return child_a, child_b


def reproduce(new_pop, selected, mutation, _crossover):
    while mutation != 0:
        mutation -= 1
        new_pop.append(mutate(random.choice(selected)))

    while _crossover != 0:
        _crossover -= 2

        parents = random.sample(selected, 2)
        while parents[0].model_type != parents[1].model_type:
            parents = random.sample(selected, 2)

        child_a, child_b = crossover(parents[0], parents[1])

        if _crossover >= 0:
            new_pop.append(child_a)
            new_pop.append(child_b)
        elif random.randint(0, 1) == 1:
            new_pop.append(child_a)
        else:
            new_pop.append(child_b)


def genetic_algorithm(train_df, test_df):
    pop = init_pop()
    fitness(pop, train_df, test_df)

    for i in range(0, conf['num_of_gens']):
        fitness(pop, train_df, test_df)
        pop.sort(key=lambda chromosome: chromosome.fitness, reverse=True)

        print('======= Generation ' + str(i) + ' =====')
        print('best_fitness: ' + str(pop[0].fitness))
        to_keep, mutation, _crossover = convert_percentages_to_values(len(pop), conf['reproduction_specs']['keep_rate'],
                                                                      conf['reproduction_specs']['mutation_rate'],
                                                                      conf['reproduction_specs']['crossover_rate'])
        new_pop = pop[:to_keep]
        selected = select(pop)
        reproduce(new_pop, selected, mutation, _crossover)
        pop = new_pop
