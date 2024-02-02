from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Union
import numpy as np
import jax.numpy as jnp


class NEAlgorithm(ABC):
    """Interface of all Neuro-evolution algorithms in EvoJAX."""

    pop_size: int

    @abstractmethod
    def ask(self) -> jnp.ndarray:
        """Ask the algorithm for a population of parameters.

        Returns
            A Jax array of shape (population_size, param_size).
        """
        raise NotImplementedError()

    @abstractmethod
    def tell(self, fitness: Union[np.ndarray, jnp.ndarray]) -> None:
        """Report the fitness of the population to the algorithm.

        Args:
            fitness - The fitness scores array.
        """
        raise NotImplementedError()

    def save_state(self) -> Any:
        """Optionally, save the state of the algorithm.

        Returns
            Saved state.
        """
        return None

    def load_state(self, saved_state: Any) -> None:
        """Optionally, load the saved state of the algorithm.

        Args:
            saved_states - The result of self.save_states().
        """
        pass

    @property
    def best_params(self) -> jnp.ndarray:
        raise NotImplementedError()

    @best_params.setter
    def best_params(self, params: Union[np.ndarray, jnp.ndarray]) -> None:
        raise NotImplementedError()


class QualityDiversityMethod(NEAlgorithm):
    """Quality diversity method."""

    params_lattice: jnp.ndarray
    fitness_lattice: jnp.ndarray
    occupancy_lattice: jnp.ndarray

    @abstractmethod
    def observe_bd(self, bd: Dict[str, jnp.ndarray]) -> None:
        raise NotImplementedError()
