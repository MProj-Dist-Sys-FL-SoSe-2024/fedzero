from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Set

import gurobipy as grb
import numpy as np
import pandas as pd

from fedzero.config import (TIMESTEP_IN_MIN, MAX_ROUND_IN_MIN, GUROBI_ENV, MIN_LOCAL_EPOCHS,
                            ENABLE_BROWN_CLIENTS_DURING_TIME_WINDOW, TIME_WINDOW_LOWER_BOUND, TIME_WINDOW_UPPER_BOUND)
from fedzero.entities import PowerDomainApi, ClientLoadApi, Client
from fedzero.oort import OortSelector
from fedzero.utility import UtilityJudge
from fedzero.config import BROWN_CLIENTS_ALLOWANCE, BROWN_CLIENTS_BUDGET_PERCENTAGE, BROWN_CLIENTS_NUMBER_PERCENTAGE, BROWN_EXCLUSION_UPDATE

_sum = grb.quicksum


class SelectionStrategy(ABC):
    def __init__(self, clients_per_round: int):
        self.clients_per_round = clients_per_round

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def select(
        self, power_domain_api: PowerDomainApi, client_load_api: ClientLoadApi, round_number: int, now: datetime
    ) -> Optional[pd.DataFrame]:
        """Selects the participating clients for a FL training round and decides on the duration.

        Args:
            power_domain_api: Power domain time series api.
            client_load_api: Client load time series api.
            round_number: Index of current round (used for warming up the utility judges)
            now: Current fedzero time

        Returns:
            None if no solution could be found. Otherwise, a DataFrame with the expected batches where
            - rows determine the participating clients
            - columns determine the time slot
        """


class RandomSelectionStrategy(SelectionStrategy):
    def __init__(self,
                 clients_per_round: int,
                 use_forecasts: bool = False,
                 min_epochs: Optional[int] = None,  # only used with use_forecasts
                 seed: Optional[int] = None):
        super().__init__(clients_per_round)
        self.use_forecasts = use_forecasts
        self.min_epochs = min_epochs
        self.rng = np.random.default_rng(seed=seed)

    def __repr__(self):
        return f"random{'_fc' if self.use_forecasts else ''}"

    def select(self, power_domain_api: PowerDomainApi, client_load_api: ClientLoadApi,
               round_number: int, now: datetime, allow_brown_clients=False) -> Optional[pd.DataFrame]:
        """Selects <CLIENTS_PER_ROUND> randomly if they have energy and capacity
        """
        clients = _filterby_current_capacity_and_energy(power_domain_api, client_load_api, now)
        if self.use_forecasts:
            clients = _filterby_forecasted_capacity_and_energy(power_domain_api, client_load_api, clients, now,
                                                               int(MAX_ROUND_IN_MIN / TIMESTEP_IN_MIN), self.min_epochs)
        if len(clients) < self.clients_per_round:
            return None

        selected_clients = self.rng.choice(clients, self.clients_per_round, replace=False)
        return pd.DataFrame(1, index=selected_clients, columns=[now + timedelta(minutes=TIMESTEP_IN_MIN)])


class FedZeroSelectionStrategy(SelectionStrategy):
    def __init__(self,
                 clients_per_round: int,
                 utility_judge: UtilityJudge,
                 alpha: float,
                 exclusion_factor: float,
                 min_epochs: float,
                 max_epochs: float,
                 seed: Optional[int] = None):
        super().__init__(clients_per_round)
        self.utility_judge = utility_judge
        self.alpha = alpha
        self.exclusion_factor = exclusion_factor
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.rng = np.random.default_rng(seed=seed)

        self.excluded_clients: List[Client] = []
        self.cycle_active_clients: Set[Client] = set()
        self.cycle_start: Optional[datetime] = None
        self.cycle_participation_mean = 0

    @property
    def exclusion_factor(self):
        if BROWN_CLIENTS_ALLOWANCE and BROWN_EXCLUSION_UPDATE:
            return self._brown_exclusion_factor
        else:
            return self._exclusion_factor
    
    @exclusion_factor.setter
    def exclusion_factor(self, value):
        self._exclusion_factor = value
        self._brown_exclusion_factor = self._exclusion_factor - min(0.25, self._exclusion_factor * 0.25)

    def __repr__(self):
        return f"fedzero_a{self.alpha}_e{self.exclusion_factor}"

    def select(self, power_domain_api: PowerDomainApi, client_load_api: ClientLoadApi,
               round_number: int, now: datetime) -> Optional[pd.DataFrame]:
        TRANSITION_PERIOD_H = 12
        wallah = self.cycle_participation_mean
        if self.cycle_start is None:
            self.cycle_start = now
        elif self.cycle_start + timedelta(hours=24) <= now:
            self.cycle_start = now
            self.cycle_participation_mean = np.mean([c.participated_rounds for c in self.cycle_active_clients])
            self.cycle_active_clients = set()
            print("############################################################")
            print(f"### NEW CYCLE! MEAN: {self.cycle_participation_mean} ###")
            print("############################################################")
        elif self.cycle_start + timedelta(hours=24 - TRANSITION_PERIOD_H) <= now:
            current_mean = np.mean([c.participated_rounds for c in self.cycle_active_clients])
            factor = (now - (self.cycle_start + timedelta(hours=24 - TRANSITION_PERIOD_H))).seconds / 3600 / TRANSITION_PERIOD_H
            wallah = self.cycle_participation_mean + (current_mean - self.cycle_participation_mean) * factor
            print(f"Cycle mean: {self.cycle_participation_mean:.2f}, Current mean: {current_mean:.2f} factor: {factor}, result: {wallah} ###")

        # Filter clients if we are not inside the time window where brown clients are allowed
        clients = _filterby_current_capacity_and_energy(power_domain_api, client_load_api, now)
        # update set of active clients for current cycle. Takes union of current set of active clients and the
        # set of clients that are currently available and have enough energy and capacity
        self.cycle_active_clients = self.cycle_active_clients.union(clients)
        # when alpha is set to true, the list of clients that are excluded is updated based on the clients
        # statistical utility and number of rounds they have participated in
        if self.alpha:
            self._update_excluded_clients(clients, round_number, wallah)

            # filter out clients that are in the excluded clients list
            clients = [client for client in clients if client not in self.excluded_clients]

        utility = self.utility_judge.utility()
        for d in range(1, int(MAX_ROUND_IN_MIN / TIMESTEP_IN_MIN) + 1):
            filtered_clients = _filterby_forecasted_capacity_and_energy(power_domain_api, client_load_api, clients, now, d, self.min_epochs)
            filtered_clients_capacity = _filterby_current_capacity(client_load_api, clients, now, d, self.min_epochs)
            if len(filtered_clients) < self.clients_per_round:
                continue
            solution = self._optimal_selection(power_domain_api, client_load_api, filtered_clients, utility, d=d, now=now)
            if solution is None:
                continue
            if ENABLE_BROWN_CLIENTS_DURING_TIME_WINDOW and TIME_WINDOW_LOWER_BOUND <= round_number <= TIME_WINDOW_UPPER_BOUND:
                unused_green_clients = [_client for _client in filtered_clients if _client not in solution.index]
                # Calc Energy Series
                batches = solution.sum(axis=1)
                energy = pd.Series()
                for index, item in batches.items():
                    index: Client
                    energy[index] = item * index.energy_per_batch
                # Define upper energy limit and lower client limit
                limit = round(energy.sum() * BROWN_CLIENTS_BUDGET_PERCENTAGE)
                brown_clients = [
                                 _client for _client in client_load_api.get_clients() if
                                 (_client not in filtered_clients_capacity)
                                 and (_client not in self.excluded_clients)
                                ]
                brown_clients.extend(unused_green_clients)
                min_brown_clients = min(len(brown_clients), max(1, self.clients_per_round * BROWN_CLIENTS_NUMBER_PERCENTAGE))
                brown_solution = self._brown_selection(client_load_api, brown_clients, utility, d=d, l=limit, min_clients=min_brown_clients, now=now)
                if brown_solution is None or len(brown_solution.index) < min_brown_clients:
                    continue
                # Calc Brown Energy Series
                brown_batches = brown_solution.sum(axis=1)
                brown_energy = pd.Series()
                for index, item in brown_batches.items():
                    index: Client
                    index.is_brown = True
                    brown_energy[index] = item * index.energy_per_batch
                # Sum Brown Energy
                brown_energy_sum = brown_energy.sum()
                if not (int(brown_energy_sum) <= limit * 1.01):
                    raise RuntimeError(f"Brown Energy Limit Exceeded with {int(brown_energy_sum)} of {limit * 1.01}")
                solution = pd.concat([solution, brown_solution])
            if solution is not None:
                return solution
        return None  # if no solution found before max round durationself.round_prefer_duration

    def _update_excluded_clients(self, clients: List[Client], round_number: int, wallah) -> None:
        participants = {client for client in clients if client.participated_in_last_round(round_number)}
        if not participants:
            return

        print("--- FedZero Exclusion ------------------------")
        utility_threshold = np.quantile([client.statistical_utility() for client in participants], self.exclusion_factor)
        print(f"| Excluding {int(len(participants) * self.exclusion_factor)} clients below statistical utility {utility_threshold:.12}.")
        for client in participants:
            if client.statistical_utility() <= utility_threshold:
                self.excluded_clients.append(client)

        print(f"| Excluded clients after add: {len(self.excluded_clients)}")
        for i, client in enumerate(self.excluded_clients):
            participated_rounds = client.participated_rounds - wallah
            if participated_rounds > 0:
                probability = min(self.alpha * 1 / participated_rounds, 1)
            else:
                probability = 1

            print(f"| #{i} {client.name}: {client.participated_rounds} part -> {probability:.0%} ...", end="")
            if self.rng.random() <= probability:
                print(" SUCCESS")
                if client in self.excluded_clients:
                    self.excluded_clients.remove(client)
            else:
                print("")
        print(f"| Excluded clients after remove: {len(self.excluded_clients)}")
        print("----------------------------------------------")

    def _brown_selection(self,
                         client_load_api: ClientLoadApi,
                         clients: List[Client],
                         utility: Dict[Client, float],
                         d: int,
                         l: int,
                         min_clients: int,
                         now: datetime):
        model = grb.Model(name="Brown Client Selection Model", env=GUROBI_ENV)

        m_alloc = {(c, t): model.addVar(lb=0, ub=client_load_api.forecast(now + timedelta(minutes=TIMESTEP_IN_MIN * t), duration_in_timesteps=1, client_name=c.name).iloc[0]) for c in clients for t in range(d)}
        b = {c: model.addVar(vtype=grb.GRB.BINARY) for c in clients}

        model.addConstr(_sum(m_alloc[client, t] * client.energy_per_batch for client in clients for t in range(d)) <= l)

        for client in clients:
            min_batches = (client.batches_per_epoch * self.min_epochs) + 1
            max_batches = client.batches_per_epoch * self.max_epochs
            model.addGenConstrIndicator(b[client], True, min_batches <= _sum(m_alloc[client, t] for t in range(d)))
            model.addGenConstrIndicator(b[client], True, max_batches >= _sum(m_alloc[client, t] for t in range(d)))
            model.addGenConstrIndicator(b[client], False, 0 >= _sum(m_alloc[client, t] for t in range(d)))

        model.addConstr(_sum(b[c] for c in clients) >= min_clients)

        model.ModelSense = grb.GRB.MAXIMIZE
        model.setObjective(_sum(b[c] * utility[c] * m_alloc[c, t] for c in clients for t in range(d)))
        model.optimize()

        if model.Status == grb.GRB.INFEASIBLE:
            return None

        df = pd.DataFrame([var.X for var in m_alloc.values()], index=pd.MultiIndex.from_tuples(m_alloc.keys()))
        df = df.unstack(level=1)
        selected_clients = pd.Series([var.X for var in b.values()], index=b.keys()).sort_index()
        df = df[np.isclose(selected_clients, 1)]

        df.columns = pd.date_range(start=now + pd.DateOffset(minutes=TIMESTEP_IN_MIN), periods=d, freq=f"{TIMESTEP_IN_MIN}T")
        return df.sort_index()

    def _optimal_selection(self,
                           power_domain_api: PowerDomainApi,
                           client_load_api: ClientLoadApi,
                           clients: List[Client],
                           utility: Dict[Client, float],
                           d: int,
                           now: datetime):
        model = grb.Model(name="MIP Model", env=GUROBI_ENV)

        # defining the decision variables for a Gurobi optimization model, which will be used to allocate resources
        # to clients over time in an optimal way
        m_alloc = {(c, t): model.addVar(
            lb=0,  # lower bound
            ub=client_load_api.forecast(now + timedelta(minutes=TIMESTEP_IN_MIN * t),  # upper bound
                                        duration_in_timesteps=1,
                                        client_name=c.name)
            .iloc[0]) for c in clients for t in range(d)}
        b = {c: model.addVar(vtype=grb.GRB.BINARY) for c in clients}
        
        # add constraints to the model that ensures that the total energy used by all clients in a particular zone
        # at each time step does not exceed the available energy in that zone
        for zone in set(client.zone for client in clients):  # iterate over all zones
            # create list of clients that are in the current zone
            clients_in_zone = [client for client in clients if client.zone == zone]
            # iterate over the forecasted available energy in the current zone at each time step
            for i, value in enumerate(power_domain_api.forecast(now, duration_in_timesteps=d, zone=zone)):
                # add constraint that ensures that the total energy used by all clients in the zone does not exceed
                # the forecasted available energy in the zone at that time step
                model.addConstr(
                    _sum(m_alloc[client, i] * client.energy_per_batch for client in clients_in_zone) <= value)

        # only allow a certain number of clients to be selected
        model.addConstr(_sum(b[c] for c in clients) == self.clients_per_round)

        # add constraints to the model to ensure that the number of batches processed by each client is within a
        # specified range
        for client in clients:
            min_batches = client.batches_per_epoch * self.min_epochs
            max_batches = client.batches_per_epoch * self.max_epochs
            model.addGenConstrIndicator(b[client], True, min_batches <= _sum(m_alloc[client, t] for t in range(d)))
            model.addGenConstrIndicator(b[client], True, max_batches >= _sum(m_alloc[client, t] for t in range(d)))
            model.addGenConstrIndicator(b[client], False, 0 >= _sum(m_alloc[client, t] for t in range(d)))

        model.ModelSense = grb.GRB.MAXIMIZE
        model.setObjective(_sum(b[c] * utility[c] * m_alloc[c, t] for c in clients for t in range(d)))
        model.optimize()

        if model.Status == grb.GRB.INFEASIBLE:
            return None

        df = pd.DataFrame([var.X for var in m_alloc.values()], index=pd.MultiIndex.from_tuples(m_alloc.keys()))
        df = df.unstack(level=1)
        selected_clients = pd.Series([var.X for var in b.values()], index=b.keys()).sort_index()
        df = df[np.isclose(selected_clients, 1)]

        df.columns = pd.date_range(start=now + pd.DateOffset(minutes=TIMESTEP_IN_MIN), periods=d, freq=f"{TIMESTEP_IN_MIN}min")
        return df.sort_index()


class OortSelectionStrategy(SelectionStrategy):
    def __init__(self, clients_per_round: int, use_forecasts: bool = False, seed: Optional[int] = None):
        super().__init__(clients_per_round)
        self.oort_selector = OortSelector(sample_seed=seed)
        self.use_forecasts = use_forecasts

    def __repr__(self):
        return f"oort{'_fc' if self.use_forecasts else ''}"

    def select(self, power_domain_api: PowerDomainApi, client_load_api: ClientLoadApi,
               round_number: int, now: datetime) -> Optional[pd.DataFrame]:
        # register clients
        if len(self.oort_selector.totalArms) == 0:
            for client in client_load_api.get_clients():
                timesteps_per_epoch = client.batches_per_epoch / client.batches_per_timestep
                self.oort_selector.register_client(clientId=client.name, size=client.num_samples,
                                                   duration=timesteps_per_epoch)

        clients = _filterby_current_capacity_and_energy(power_domain_api, client_load_api, now)
        if len(clients) < self.clients_per_round:
            return None

        for client in clients:
            # Estimate and update duration based on current capacity and energy
            required_batches = client.batches_per_epoch * MIN_LOCAL_EPOCHS
            if self.use_forecasts:
                fc_duration = int(MAX_ROUND_IN_MIN / TIMESTEP_IN_MIN)
                expected_duration_based_on_capacity = _estimate_duration_based_on_forecast(
                    required_batches, client_load_api.forecast(now, fc_duration, client.name)
                )
                expected_duration_based_on_energy = _estimate_duration_based_on_forecast(
                    required_batches, power_domain_api.forecast(now, fc_duration, client.zone) / client.energy_per_batch
                )
            else:
                expected_duration_based_on_capacity = required_batches / client_load_api.actual(now, client.name)
                expected_duration_based_on_energy = required_batches * client.energy_per_batch / power_domain_api.actual(
                    now, client.zone)
            expected_duration = max(expected_duration_based_on_capacity, expected_duration_based_on_energy)

            if client.participated_in_last_round(round_number):
                self.oort_selector.update_client_util(clientId=client.name, reward=client.statistical_utility(),
                                                      time_stamp=round_number, duration=expected_duration)
            else:
                self.oort_selector.update_duration(clientId=client.name, duration=expected_duration)

        selected_client_names = self.oort_selector.select_participant(self.clients_per_round,
                                                                      feasible_clients=[client.name for client in
                                                                                        clients])
        index = [c for c in client_load_api.get_clients() if c.name in selected_client_names]
        return pd.DataFrame(1, index=index, columns=[now + timedelta(minutes=TIMESTEP_IN_MIN)])


def _estimate_duration_based_on_forecast(required_batches, fc):
    remaining_batches = required_batches
    for i, batches_in_timestep in enumerate(fc, start=1):
        remaining_batches -= batches_in_timestep
        if remaining_batches <= 0:
            return i + remaining_batches / required_batches
    return required_batches / fc[0]


def _filterby_current_capacity_and_energy(power_domain_api: PowerDomainApi,
                                          client_load_api: ClientLoadApi,
                                          now: datetime) -> List[Client]:
    zones_with_energy = [zone for zone in power_domain_api.zones if power_domain_api.actual(now, zone) > 0.0]
    clients = [client for client in client_load_api.get_clients(zones_with_energy) if
               client_load_api.actual(now, client.name) > 0.0]
    print(f"There are {len(clients)} clients available across {len(zones_with_energy)} power domains.")
    return clients


def _filterby_current_capacity(client_load_api: ClientLoadApi,
                                clients: List[Client],
                                now: datetime,
                                d: int,
                                min_epochs: float) -> List[Client]:
    # # Fetch all clients without filtering by energy availability
    # clients = [client for client in client_load_api.get_clients() if client_load_api.actual(now, client.name) > 0.0]
    # print(f"There are {len(clients)} clients available based on current capacity.")
    # return clients

    filtered_clients: List[Client] = []
    for client in clients:
        possible_batches = client_load_api.forecast(now, duration_in_timesteps=d, client_name=client.name)
        # Significantly faster than pandas
        total_max_batches = possible_batches.values.sum()
        if total_max_batches >= client.batches_per_epoch * min_epochs:
            filtered_clients.append(client)
    return filtered_clients


def _filterby_forecasted_capacity_and_energy(power_domain_api: PowerDomainApi,
                                             client_load_api: ClientLoadApi,
                                             clients: List[Client],
                                             now: datetime,
                                             d: int,
                                             min_epochs: float) -> List[Client]:
    filtered_clients: List[Client] = []
    for client in clients:
        possible_batches = client_load_api.forecast(now, duration_in_timesteps=d, client_name=client.name)
        ree_powered_batches = power_domain_api.forecast(now, duration_in_timesteps=d,
                                                        zone=client.zone) / client.energy_per_batch
        # Significantly faster than pandas
        total_max_batches = np.minimum(possible_batches.values, ree_powered_batches.values).sum()
        if total_max_batches >= client.batches_per_epoch * min_epochs:
            filtered_clients.append(client)
    return filtered_clients
