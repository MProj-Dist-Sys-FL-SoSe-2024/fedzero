from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List

from fedzero.entities import Client


class UtilityJudge(ABC):
    """Generates a client utility weighting for each training round to mitigate biases introduced by FedZero.

    All weightings are between 0 and 1. Clients weighted with 0 cannot be selected for participation.
    """

    def __init__(self, clients: List[Client]):
        self.clients = clients

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def utility(self) -> Dict[Client, float]:
        pass


class StaticJudge(UtilityJudge):
    """Always assigns the same utility to each client."""

    def __repr__(self):
        return "static"

    def utility(self) -> Dict[Client, float]:
        return {client: 1 for client in self.clients}


class ParticipationJudge(UtilityJudge):
    """Weights clients based on their past participation.

    Weights each client by its inverse participation to the power of `weighting_exponent`.
    The result is normalized to return a weighting between 0 and 1 for each client.

    Args:
        clients: The clients to be weighted as Dictionary with client name as key and client as value.
        weighting_exponent: The client's inverse participation is taken to the power of this value. Higher
            values will result in underrepresented clients to retrieve a relatively higher weighting.
            A weighting_exponent of 0 will give the same weight to all clients.
    """

    def __init__(self, clients: List[Client], weighting_exponent: float):
        self.weighting_exponent = weighting_exponent
        super().__init__(clients)

    def __repr__(self):
        return "part"
        
    def utility(self) -> Dict[Client, float]:
        """
        Calculate and return dictionary of clients and their respective utility.
        Utility for each client is calculated as:
            (min_participation / past_participation) ** self.weighting_exponent

        Returns:
            Dict[Client, float]: Dictionary of clients and their respective utility.
        """
        participation = self._calculate_participation()             # dict of clients and their respective participation
        min_participation = max(1, min(participation.values()))     # minimum participation among all clients or 1
        weighting = {}

        # calculate the utility for each client
        for client, past_participation in participation.items():
            try:
                # calculate the utility for the client
                weighting[client] = (min_participation / past_participation) ** self.weighting_exponent
            except ZeroDivisionError:
                # if the client has not participated in any round, set the utility to 1
                weighting[client] = 1
        return weighting

    def _calculate_participation(self) -> Dict[Client, int]:
        """
        Calculate and return dictionary of clients and their respective participation.
        Keys are clients and values are the number of rounds they have participated in.

        Returns:
            Dict[Client, int]: Dictionary of clients and their respective participation.
        """
        participation_dict: Dict[Client, int] = defaultdict(int)
        # iterate over all clients and sum up their participation
        for client in self.clients:
            # add the number of rounds the client has participated in to the participation_dict
            participation_dict[client] += client.participated_rounds
        return participation_dict


class StatUtilityJudge(UtilityJudge):
    """
    Weights clients based on their statistical utility.

    Statistical utility is a measure for the quality of the data of a client. It can be interpreted as the
    overall importance of a client for the learning process.
    Clients with a higher statistical utility will receive a higher weighting and are given more importance in the
    learning process.

    The statistical utility is normalized to return a weighting between 0 and 1 for each client.
    It is calculated as the difference between the statistical utility of a client and the minimum statistical utility
    divided by the range of statistical utilities.
    """

    def __repr__(self):
        return "stat"

    def utility(self) -> Dict[Client, float]:
        # create dictionary of clients and their respective statistical utility.
        # statistical utility is obtained from the client's statistical_utility() method
        statistical_utility_dict = {client: client.statistical_utility() for client in self.clients}

        # min and max statistical utility among all clients
        min_utility = min(statistical_utility_dict.values())
        max_utility = max(statistical_utility_dict.values())

        weighting = {}
        # calculate the utility for each client
        for client, util in statistical_utility_dict.items():
            try:
                # normalize the statistical utility to return a weighting between 0 and 1
                weighting[client] = (statistical_utility_dict[client] - min_utility) / (max_utility - min_utility)
            except ZeroDivisionError:
                weighting[client] = 1
        return weighting
