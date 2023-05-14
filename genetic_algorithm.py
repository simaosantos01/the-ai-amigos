import copy
import random
from utils import convert_percentages_to_values, compute_class_weights
from h2o.estimators import H2ORandomForestEstimator, H2ODeepLearningEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

conf = {
    "pop_size": 50,
    "rf_pop_percentage": 50,
    "dl_pop_percentage": 50
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

    def get_genes(self):
        if self.model_type == 1:
            return {'model_type': 'RF', 'max_depth': self.params[0], 'min_rows': self.params[1]}
        else:
            return {'model_type': 'DL', 'epochs': self.params[1], 'adaptive_rate': self.params[1]}

    def __str__(self):
        return str(self.get_genes()) + ' | Fitness: ' + str(self.fitness)


def init_pop():
    pop = []
    algo1, algo2, algo3 = convert_percentages_to_values(conf['pop_size'], conf['rf_pop_percentage'],
                                                        conf['dl_pop_percentage'], 0)

    for i in range(algo1):
        params = [random.randint(1, 20), random.randint(1, 30)]
        pop.append(Chromosome(1, params))

    for i in range(algo2):
        params = [random.randint(1, 10), random.randint(0, 1)]
        pop.append(Chromosome(2, params))

    return pop


def calculate_fitness(genes, train_df, test_df):
    predictors = train_df.col_names[1:]
    response = train_df.col_names[0]

    if genes['model_type'] == 'RF':
        model = H2ORandomForestEstimator(max_depth=genes['max_depth'], min_rows=genes['min_rows'])
    else:
        model = H2ODeepLearningEstimator(epochs=genes['epochs'], adaptive_rate=genes['adaptive_rate'])

    model.train(x=predictors, y=response, training_frame=train_df, validation_frame=test_df)

    true_values = test_df.as_data_frame()['label'].tolist()
    predictions = model.predict(test_df).as_data_frame()['predict'].tolist()
    predictions = [round(value) for value in predictions]

    # Calculate class weights based on class imbalance
    class_weights = compute_class_weights(true_values)

    # Calculate weighted metrics
    accuracy = accuracy_score(true_values, predictions, sample_weight=class_weights)
    precision = precision_score(true_values, predictions, average='weighted', sample_weight=class_weights)
    recall = recall_score(true_values, predictions, average='weighted', sample_weight=class_weights)
    f1 = f1_score(true_values, predictions, average='weighted', sample_weight=class_weights)

    # Calculate weighted fitness score
    fitness_score = (accuracy + precision + recall + f1) / 4.0
    return fitness_score


def fitness(pop, train_df, test_df):
    for chromosome in pop:
        genes = chromosome.get_genes()
        chromosome.fitness = calculate_fitness(genes, train_df, test_df)


def genetic_algorithm(train_df, test_df):
    pop = init_pop()
    print(calculate_fitness(pop[0].get_genes(), train_df, test_df))

