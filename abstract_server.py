
'''
# filename: abstract_server.py
# author: Nick Johnson
# created: 3/15/21
# last modified: 3/15/21
# skeleton of an abstract factory 
#  for organizing different models in different servers
'''

from __future__ import annotations
from abc import ABC, abstractmethod


class SkyNet(ABC):
    '''
    interface for abstract model and server products
    '''

    @abstractmethod
    def create_server(self) -> AbstractServer:
        pass

    @abstractmethod
    def create_model(self) -> AbstractModel:
        pass

    
class GenerativeFactory(SkyNet):
    '''
    concrete factory for producing generative systems
    '''

    def create_server(self) -> AbstractServer:
        return GenerativeServer()

    def create_model(self) -> AbstractModel:
        return GenerativeModel()

    
class PredictiveFactory(SkyNet):
    '''
    concrete factory for producing predictive systems
    '''

    def create_server(self) -> AbstractServer:
        return VisionServer()

    def create_model(self) -> AbstractModel:
        return VisionModel()


class AbstractServer(ABC):
    '''
    base server interface
    '''

    @abstractmethod
    def flask_server(self) -> str:
        pass

    def redis_server(self) -> str:
        pass
    
    @abstractmethod
    def open_queue(self, collaborator: AbstractServer) -> str:
        pass


class GenerativeServer(AbstractServer):
    def flask_server(self) -> str:
        return "result"

    def redis_server(self) -> str:
        return "datastore"
    
    def open_queue(self, collaborator: AbstractServer) -> str:
        return"queue"


class PredictiveServer(AbstractServer):
    def flask_server(self) -> stf:
        return "results"

    def redis_server(self) -> str:
        return "datastore"
    
    def open_queue(self, collaborator: AbstractServer) -> str:
        return"queue"

    
class AbstractModel(ABC):
    '''
    base model interface
    '''
    
    @abstractmethod
    def load_model(self) -> str:
        pass

    @abstractmethod
    def run_model(self) -> str:
        pass
    
    
class GenerativeModel(AbstractModel):
    def load_model(self) -> str:
        # train model / specific layers
        return "model"

    def run_model(self) -> str:
        return "results"

    
class PredictiveModel(AbstractModel):
    def load_model(self) -> str:
        return "model"

    def run_model(self) -> str:
        return "results"
