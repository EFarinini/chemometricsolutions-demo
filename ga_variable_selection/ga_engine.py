"""
Genetic Algorithm Engine for Variable Selection
==============================================

Implementation of Riccardo Leardi's genetic algorithms optimized for
Streamlit execution with NumPy vectorization.

Based on:
- Leardi, R. (2000). Application of genetic algorithms to feature selection
  under full validation conditions and to outlier detection.
- MATLAB implementations: GAPLSR, GALDA, GADIST, GAMAHAL
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Iterator, Optional
import warnings


class GeneticAlgorithm:
    """
    Genetic Algorithm for variable selection with backward stepwise elimination.

    This implementation follows Leardi's approach with:
    - Probabilistic population initialization
    - Single-point crossover
    - Bit-flip mutation
    - Backward stepwise elimination refinement
    - Multiple independent runs for robustness

    Parameters
    ----------
    dataset : np.ndarray
        Input data matrix of shape (n_samples, n_variables)
    fitness_fn : callable
        Fitness evaluation function with signature:
        fitness_fn(X, y, selected_indices) -> float
    config : dict
        Configuration parameters:
        - runs: Number of independent GA runs (default: 20)
        - population_size: Number of chromosomes per generation (default: 20)
        - evaluations: Number of fitness evaluations per run (default: 50)
        - mutation_prob: Probability of bit mutation (default: 0.01)
        - crossover_prob: Probability of crossover (default: 0.5)
        - cv_groups: Cross-validation folds (default: 3)
        - min_vars: Minimum variables to select (default: 1)
        - max_vars: Maximum variables to select (default: None, uses n_vars/2)
    y : np.ndarray, optional
        Target variable (for supervised methods)
    random_seed : int, optional
        Random seed for reproducibility. Used for multiple independent runs
        with different random initializations (Leardi's approach)

    Attributes
    ----------
    n_samples_ : int
        Number of samples in dataset
    n_variables_ : int
        Number of variables in dataset
    selection_frequency_ : np.ndarray
        Count of selections per variable across all runs
    best_chromosome_ : np.ndarray
        Best chromosome found (binary array)
    best_fitness_ : float
        Best fitness score achieved
    fitness_history_ : list
        Evolution of fitness scores over runs
    """

    def __init__(
        self,
        dataset: np.ndarray,
        fitness_fn: Callable,
        config: Dict,
        y: Optional[np.ndarray] = None,
        random_seed: Optional[int] = None
    ):
        # Set random seed for reproducibility (for multiple independent runs)
        if random_seed is not None:
            np.random.seed(random_seed)

        # Validate dataset
        if not isinstance(dataset, np.ndarray):
            dataset = np.array(dataset)

        if dataset.ndim != 2:
            raise ValueError(f"Dataset must be 2D array, got shape {dataset.shape}")

        if np.any(np.isnan(dataset)) or np.any(np.isinf(dataset)):
            raise ValueError("Dataset contains NaN or Inf values")

        self.dataset = dataset
        self.fitness_fn = fitness_fn
        self.y = y
        self.random_seed = random_seed

        # Dataset dimensions (MUST be defined BEFORE _validate_config)
        self.n_samples_, self.n_variables_ = dataset.shape

        # Validate configuration (uses self.n_variables_)
        self.config = self._validate_config(config)

        # Results storage
        self.selection_frequency_ = np.zeros(self.n_variables_, dtype=int)
        self.best_chromosome_ = None
        self.best_fitness_ = -np.inf
        self.fitness_history_ = []
        self.chromosome_library_ = []
        self.fitness_library_ = []

        # Runtime control
        self._stop_flag = False
        self._current_evaluation = 0
        self._total_evaluations = self.config['runs'] * self.config['evaluations']

    def _validate_config(self, config: Dict) -> Dict:
        """Validate and set default configuration parameters."""
        default_config = {
            'runs': 20,
            'population_size': 20,
            'evaluations': 50,
            'mutation_prob': 0.01,
            'crossover_prob': 0.5,
            'cv_groups': 3,
            'min_vars': 1,
            'max_vars': None,
            'backward_step_interval': 10,
            'elite_size': 2
        }

        # Merge with defaults
        validated = {**default_config, **config}

        # Validate ranges
        if validated['runs'] < 1:
            raise ValueError("runs must be >= 1")
        if validated['population_size'] < 4:
            raise ValueError("population_size must be >= 4")
        if validated['evaluations'] < 10:
            raise ValueError("evaluations must be >= 10")
        if not 0 < validated['mutation_prob'] <= 1:
            raise ValueError("mutation_prob must be in (0, 1]")
        if not 0 < validated['crossover_prob'] <= 1:
            raise ValueError("crossover_prob must be in (0, 1]")
        if validated['cv_groups'] < 2:
            raise ValueError("cv_groups must be >= 2")

        # Set max_vars if not specified
        if validated['max_vars'] is None:
            validated['max_vars'] = max(validated['min_vars'], self.n_variables_ // 2)

        return validated

    def initialize_population(self) -> np.ndarray:
        """
        Initialize population with probabilistic variable selection.

        Uses smoothed probabilities based on variable autocorrelation
        (important for spectral data). Variables near each other have
        correlated selection probabilities.

        Returns
        -------
        population : np.ndarray
            Binary array of shape (population_size, n_variables)
        """
        population_size = self.config['population_size']
        n_vars = self.n_variables_
        min_vars = self.config['min_vars']
        max_vars = self.config['max_vars']

        # Base probability: aim for average of min_vars to max_vars
        avg_vars = (min_vars + max_vars) / 2
        base_prob = avg_vars / n_vars

        # Add smooth variation (for spectral data autocorrelation)
        # Create probability wave that varies along variable index
        x = np.linspace(0, 4 * np.pi, n_vars)
        prob_variation = 0.2 * np.sin(x) + 0.1 * np.cos(2 * x)
        probabilities = np.clip(base_prob + prob_variation, 0.05, 0.95)

        # Generate population
        population = np.zeros((population_size, n_vars), dtype=bool)

        for i in range(population_size):
            # Random selection based on probabilities
            chromosome = np.random.rand(n_vars) < probabilities

            # Ensure minimum variables selected
            n_selected = np.sum(chromosome)
            if n_selected < min_vars:
                # Randomly activate variables to reach minimum
                inactive = np.where(~chromosome)[0]
                to_activate = np.random.choice(
                    inactive,
                    size=min_vars - n_selected,
                    replace=False
                )
                chromosome[to_activate] = True

            # Ensure maximum not exceeded
            elif n_selected > max_vars:
                # Randomly deactivate variables
                active = np.where(chromosome)[0]
                to_deactivate = np.random.choice(
                    active,
                    size=n_selected - max_vars,
                    replace=False
                )
                chromosome[to_deactivate] = False

            population[i] = chromosome

        return population

    def crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single-point crossover between two parent chromosomes.

        Parameters
        ----------
        parent1, parent2 : np.ndarray
            Parent chromosomes (binary arrays)

        Returns
        -------
        offspring1, offspring2 : np.ndarray
            Two offspring chromosomes
        """
        n_vars = len(parent1)

        # Random crossover point
        crossover_point = np.random.randint(1, n_vars)

        # Create offspring
        offspring1 = np.concatenate([
            parent1[:crossover_point],
            parent2[crossover_point:]
        ])
        offspring2 = np.concatenate([
            parent2[:crossover_point],
            parent1[crossover_point:]
        ])

        # Ensure at least one variable selected
        if not np.any(offspring1):
            offspring1[np.random.randint(n_vars)] = True
        if not np.any(offspring2):
            offspring2[np.random.randint(n_vars)] = True

        return offspring1, offspring2

    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Apply bit-flip mutation to chromosome.

        Parameters
        ----------
        chromosome : np.ndarray
            Input chromosome (binary array)

        Returns
        -------
        mutated : np.ndarray
            Mutated chromosome
        """
        mutated = chromosome.copy()
        mutation_prob = self.config['mutation_prob']

        # Random bit flips
        mutation_mask = np.random.rand(len(mutated)) < mutation_prob
        mutated[mutation_mask] = ~mutated[mutation_mask]

        # Ensure at least one variable selected
        if not np.any(mutated):
            mutated[np.random.randint(len(mutated))] = True

        return mutated

    def evaluate(self, chromosome: np.ndarray) -> float:
        """
        Evaluate fitness of a chromosome.

        Parameters
        ----------
        chromosome : np.ndarray
            Binary chromosome indicating selected variables

        Returns
        -------
        fitness : float
            Fitness score (higher is better)
        """
        selected_indices = np.where(chromosome)[0]

        if len(selected_indices) == 0:
            return 0.0

        try:
            # Call fitness function
            if self.y is not None:
                fitness = self.fitness_fn(
                    self.dataset[:, selected_indices],
                    self.y,
                    selected_indices
                )
            else:
                # For unsupervised methods (e.g., Mahalanobis)
                fitness = self.fitness_fn(
                    self.dataset[:, selected_indices],
                    selected_indices
                )

            # Ensure numeric return
            fitness = float(fitness)

            # Check for invalid values
            if np.isnan(fitness) or np.isinf(fitness):
                return 0.0

            return fitness

        except Exception as e:
            warnings.warn(f"Fitness evaluation failed: {str(e)}", RuntimeWarning)
            return 0.0

    def _tournament_selection(
        self,
        population: np.ndarray,
        fitness_scores: np.ndarray,
        tournament_size: int = 3
    ) -> np.ndarray:
        """
        Select parent using tournament selection.

        Parameters
        ----------
        population : np.ndarray
            Current population
        fitness_scores : np.ndarray
            Fitness scores for population
        tournament_size : int
            Number of individuals in tournament

        Returns
        -------
        parent : np.ndarray
            Selected parent chromosome
        """
        # Random individuals for tournament
        indices = np.random.choice(
            len(population),
            size=tournament_size,
            replace=False
        )

        # Select best from tournament
        tournament_fitness = fitness_scores[indices]
        winner_idx = indices[np.argmax(tournament_fitness)]

        return population[winner_idx].copy()

    def run(self) -> Dict:
        """
        Execute genetic algorithm with multiple runs.

        Returns
        -------
        results : dict
            Dictionary containing:
            - selected_variables: Indices of selected variables
            - selection_frequency: Selection count per variable
            - best_fitness: Best fitness achieved
            - best_chromosome: Best chromosome found
            - fitness_history: Evolution of fitness scores
        """
        # Reset results
        self.selection_frequency_ = np.zeros(self.n_variables_, dtype=int)
        self.best_fitness_ = -np.inf
        self.fitness_history_ = []
        self.chromosome_library_ = []
        self.fitness_library_ = []

        # Execute multiple runs
        for run_idx in range(self.config['runs']):
            if self._stop_flag:
                break

            run_result = self._single_run(run_idx)

            # Update global best
            if run_result['best_fitness'] > self.best_fitness_:
                self.best_fitness_ = run_result['best_fitness']
                self.best_chromosome_ = run_result['best_chromosome'].copy()

            # Update selection frequency
            self.selection_frequency_[run_result['best_chromosome']] += 1

            # Store in library
            self.chromosome_library_.append(run_result['best_chromosome'])
            self.fitness_library_.append(run_result['best_fitness'])

            # Store history
            self.fitness_history_.append({
                'run': run_idx,
                'best_fitness': run_result['best_fitness'],
                'n_variables': np.sum(run_result['best_chromosome']),
                'generation_history': run_result['generation_history']
            })

        # Compile results
        results = self.get_results()
        return results

    def _single_run(self, run_idx: int) -> Dict:
        """
        Execute single GA run.

        Parameters
        ----------
        run_idx : int
            Run index

        Returns
        -------
        result : dict
            Results from this run
        """
        # Initialize population
        population = self.initialize_population()

        # Evaluate initial population
        fitness_scores = np.array([
            self.evaluate(chromosome)
            for chromosome in population
        ])

        # Track best in run
        best_idx = np.argmax(fitness_scores)
        best_chromosome = population[best_idx].copy()
        best_fitness = fitness_scores[best_idx]

        generation_history = [best_fitness]

        # Evolution loop
        n_generations = self.config['evaluations'] // self.config['population_size']
        backward_interval = self.config['backward_step_interval']

        for gen in range(n_generations):
            if self._stop_flag:
                break

            # Create new population
            new_population = []
            elite_size = self.config['elite_size']

            # Elitism: keep best chromosomes
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())

            # Generate offspring
            while len(new_population) < self.config['population_size']:
                # Parent selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)

                # Crossover
                if np.random.rand() < self.config['crossover_prob']:
                    offspring1, offspring2 = self.crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1.copy(), parent2.copy()

                # Mutation
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)

                new_population.extend([offspring1, offspring2])

            # Trim to population size
            population = np.array(new_population[:self.config['population_size']])

            # Evaluate new population
            fitness_scores = np.array([
                self.evaluate(chromosome)
                for chromosome in population
            ])

            # Update best
            gen_best_idx = np.argmax(fitness_scores)
            if fitness_scores[gen_best_idx] > best_fitness:
                best_fitness = fitness_scores[gen_best_idx]
                best_chromosome = population[gen_best_idx].copy()

            generation_history.append(best_fitness)

            # Backward stepwise elimination (every N generations)
            if (gen + 1) % backward_interval == 0:
                improved_chromosome, improved_fitness = self._backward_stepwise(
                    best_chromosome, best_fitness
                )
                if improved_fitness > best_fitness:
                    best_fitness = improved_fitness
                    best_chromosome = improved_chromosome.copy()
                    # Replace worst in population
                    worst_idx = np.argmin(fitness_scores)
                    population[worst_idx] = best_chromosome
                    fitness_scores[worst_idx] = best_fitness

            self._current_evaluation = (run_idx * n_generations + gen + 1)

        return {
            'best_chromosome': best_chromosome,
            'best_fitness': best_fitness,
            'generation_history': generation_history
        }

    def _backward_stepwise(
        self,
        chromosome: np.ndarray,
        current_fitness: float
    ) -> Tuple[np.ndarray, float]:
        """
        Backward stepwise elimination to refine variable selection.

        Iteratively removes variables that don't contribute to fitness.

        Parameters
        ----------
        chromosome : np.ndarray
            Starting chromosome
        current_fitness : float
            Current fitness score

        Returns
        -------
        improved_chromosome : np.ndarray
            Refined chromosome
        improved_fitness : float
            Improved fitness score
        """
        improved_chromosome = chromosome.copy()
        improved_fitness = current_fitness

        selected_vars = np.where(improved_chromosome)[0]

        # Minimum variables constraint
        min_vars = self.config['min_vars']

        max_iterations = 50
        improvement_threshold = 0.001  # 0.1% minimum improvement

        for _ in range(max_iterations):
            if len(selected_vars) <= min_vars:
                break

            # Try removing each variable
            best_removal = None
            best_removal_fitness = improved_fitness

            # Randomize order to avoid bias
            var_order = np.random.permutation(selected_vars)

            for var_idx in var_order:
                # Create test chromosome without this variable
                test_chromosome = improved_chromosome.copy()
                test_chromosome[var_idx] = False

                # Evaluate
                test_fitness = self.evaluate(test_chromosome)

                # Check if improvement
                if test_fitness > best_removal_fitness * (1 + improvement_threshold):
                    best_removal = var_idx
                    best_removal_fitness = test_fitness

            # Apply best removal if found
            if best_removal is not None:
                improved_chromosome[best_removal] = False
                improved_fitness = best_removal_fitness
                selected_vars = np.where(improved_chromosome)[0]
            else:
                # No improvement found, stop
                break

        return improved_chromosome, improved_fitness

    def get_results(self) -> Dict:
        """
        Compile final results.

        Returns
        -------
        results : dict
            Dictionary containing all results
        """
        if self.best_chromosome_ is None:
            raise RuntimeError("GA has not been run yet. Call run() first.")

        selected_variables = np.where(self.best_chromosome_)[0]

        return {
            'selected_variables': selected_variables,
            'selection_frequency': self.selection_frequency_,
            'best_fitness': self.best_fitness_,
            'best_chromosome': self.best_chromosome_,
            'fitness_history': self.fitness_history_,
            'chromosome_library': self.chromosome_library_,
            'fitness_library': self.fitness_library_,
            'n_runs_completed': len(self.fitness_history_)
        }

    def run_with_progress(self) -> Iterator[Dict]:
        """
        Execute GA with progress updates for Streamlit.

        Yields progress dictionaries during execution for real-time
        display in Streamlit interface.

        Yields
        ------
        progress : dict
            Progress update containing:
            - run: Current run index
            - generation: Current generation
            - evaluation: Total evaluations completed
            - best_fitness: Best fitness so far
            - n_variables: Number of variables in best solution
            - selected_vars: List of selected variable indices
            - progress_percent: Overall progress percentage (0-100)
        """
        # Reset results
        self.selection_frequency_ = np.zeros(self.n_variables_, dtype=int)
        self.best_fitness_ = -np.inf
        self.fitness_history_ = []
        self.chromosome_library_ = []
        self.fitness_library_ = []
        self._stop_flag = False
        self._current_evaluation = 0

        # Execute multiple runs with progress reporting
        for run_idx in range(self.config['runs']):
            if self._stop_flag:
                break

            # Initialize population
            population = self.initialize_population()
            fitness_scores = np.array([
                self.evaluate(chromosome)
                for chromosome in population
            ])

            # Track best in run
            best_idx = np.argmax(fitness_scores)
            best_chromosome = population[best_idx].copy()
            best_fitness = fitness_scores[best_idx]
            generation_history = [best_fitness]

            # Update global best
            if best_fitness > self.best_fitness_:
                self.best_fitness_ = best_fitness
                self.best_chromosome_ = best_chromosome.copy()

            # Yield initial progress
            yield {
                'run': run_idx,
                'generation': 0,
                'evaluation': self._current_evaluation,
                'best_fitness': self.best_fitness_,
                'n_variables': int(np.sum(self.best_chromosome_)),
                'selected_vars': np.where(self.best_chromosome_)[0].tolist(),
                'progress_percent': (self._current_evaluation / self._total_evaluations) * 100
            }

            # Evolution loop
            n_generations = self.config['evaluations'] // self.config['population_size']
            backward_interval = self.config['backward_step_interval']

            for gen in range(n_generations):
                if self._stop_flag:
                    break

                # Create new population (same as run() method)
                new_population = []
                elite_size = self.config['elite_size']

                elite_indices = np.argsort(fitness_scores)[-elite_size:]
                for idx in elite_indices:
                    new_population.append(population[idx].copy())

                while len(new_population) < self.config['population_size']:
                    parent1 = self._tournament_selection(population, fitness_scores)
                    parent2 = self._tournament_selection(population, fitness_scores)

                    if np.random.rand() < self.config['crossover_prob']:
                        offspring1, offspring2 = self.crossover(parent1, parent2)
                    else:
                        offspring1, offspring2 = parent1.copy(), parent2.copy()

                    offspring1 = self.mutate(offspring1)
                    offspring2 = self.mutate(offspring2)

                    new_population.extend([offspring1, offspring2])

                population = np.array(new_population[:self.config['population_size']])
                fitness_scores = np.array([
                    self.evaluate(chromosome)
                    for chromosome in population
                ])

                # Update best
                gen_best_idx = np.argmax(fitness_scores)
                if fitness_scores[gen_best_idx] > best_fitness:
                    best_fitness = fitness_scores[gen_best_idx]
                    best_chromosome = population[gen_best_idx].copy()

                    if best_fitness > self.best_fitness_:
                        self.best_fitness_ = best_fitness
                        self.best_chromosome_ = best_chromosome.copy()

                generation_history.append(best_fitness)

                # Backward stepwise
                if (gen + 1) % backward_interval == 0:
                    improved_chromosome, improved_fitness = self._backward_stepwise(
                        best_chromosome, best_fitness
                    )
                    if improved_fitness > best_fitness:
                        best_fitness = improved_fitness
                        best_chromosome = improved_chromosome.copy()

                        if improved_fitness > self.best_fitness_:
                            self.best_fitness_ = improved_fitness
                            self.best_chromosome_ = best_chromosome.copy()

                        worst_idx = np.argmin(fitness_scores)
                        population[worst_idx] = best_chromosome
                        fitness_scores[worst_idx] = best_fitness

                self._current_evaluation = run_idx * n_generations + gen + 1

                # Yield progress every 5 generations
                if (gen + 1) % 5 == 0 or gen == n_generations - 1:
                    yield {
                        'run': run_idx,
                        'generation': gen + 1,
                        'evaluation': self._current_evaluation,
                        'best_fitness': self.best_fitness_,
                        'n_variables': int(np.sum(self.best_chromosome_)),
                        'selected_vars': np.where(self.best_chromosome_)[0].tolist(),
                        'progress_percent': (self._current_evaluation / self._total_evaluations) * 100
                    }

            # Store run results
            self.selection_frequency_[best_chromosome] += 1
            self.chromosome_library_.append(best_chromosome)
            self.fitness_library_.append(best_fitness)
            self.fitness_history_.append({
                'run': run_idx,
                'best_fitness': best_fitness,
                'n_variables': np.sum(best_chromosome),
                'generation_history': generation_history
            })

        # Final yield with complete results
        yield {
            'run': self.config['runs'] - 1,
            'generation': n_generations,
            'evaluation': self._total_evaluations,
            'best_fitness': self.best_fitness_,
            'n_variables': int(np.sum(self.best_chromosome_)),
            'selected_vars': np.where(self.best_chromosome_)[0].tolist(),
            'progress_percent': 100.0,
            'completed': True
        }

    def stop(self):
        """Stop the GA execution (for Streamlit stop button)."""
        self._stop_flag = True
