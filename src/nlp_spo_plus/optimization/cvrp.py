from __future__ import annotations

from typing import NoReturn
from itertools import combinations
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import math
import torch
from pyepo.model.opt import optModel


class cvrpMTZModel(optModel):
    """
    Miller-Tucker-Zemlin (MTZ) formulation for the CVRP.
    """

    def __init__(
        self,
        num_nodes: int,
        demands: np.ndarray | list,
        capacity: float,
        num_vehicles: int
    ) -> None:
        self.num_nodes = num_nodes
        self.demands = demands
        self.capacity = capacity
        self.num_vehicles = num_vehicles
        # initialize edges
        self.edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
        super().__init__()

    @property
    def num_cost(self) -> int:
        return len(self.edges)

    def _getModel(self) -> tuple:
        m = gp.Model("cvrp_mtz")
        m.setParam("OutputFlag", 0)
        
        # variables
        self.x = m.addVars(self.edges, name="x", vtype=GRB.BINARY)
        clients = range(1, self.num_nodes)
        self.u = m.addVars(
            clients, name="u", vtype=GRB.CONTINUOUS, lb=self.demands[1:], ub=self.capacity
        )
        
        m.modelSense = GRB.MINIMIZE
        
        # degree constraints
        m.addConstrs((self.x.sum("*", j) == 1 for j in clients), name="in_degree")
        m.addConstrs((self.x.sum(i, "*") == 1 for i in clients), name="out_degree")
        m.addConstr(self.x.sum(0, "*") <= self.num_vehicles, name="max_vehicles")

        # MTZ subtour elimination
        for i in clients:
            for j in clients:
                if i != j:
                    m.addConstr(
                        self.u[i] - self.u[j] + self.capacity * self.x[i, j] <= self.capacity - self.demands[j],
                        name=f"MTZ_{i}_{j}"
                    )
        return m, self.x

    def setObj(self, c: np.ndarray | torch.Tensor | list) -> None:
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector cannot match number of edges.")
        obj = gp.quicksum(c[idx] * self.x[i, j] for idx, (i, j) in enumerate(self.edges))
        self._model.setObjective(obj)

    def solve(self) -> tuple[np.ndarray, float]:
        self._model.update()
        self._model.optimize()
        sol = np.zeros(self.num_cost)
        if self._model.status == GRB.OPTIMAL:
            for idx, (i, j) in enumerate(self.edges):
                sol[idx] = round(self.x[i, j].X)
        else:
            raise RuntimeError(f"Model has no optimal solution. Status: {self._model.status}")
        return sol, self._model.objVal

    def relax(self) -> cvrpMTZModelRel:
        return cvrpMTZModelRel(self.num_nodes, self.demands, self.capacity, self.num_vehicles)


class cvrpMTZModelRel(cvrpMTZModel):
    """
    LP relaxation of the MTZ formulation.
    """

    def _getModel(self) -> tuple:
        m = gp.Model("cvrp_mtz_rel")
        m.setParam("OutputFlag", 0)
        
        # continuous relaxed variables
        self.x = m.addVars(self.edges, name="x", vtype=GRB.CONTINUOUS, lb=0, ub=1)
        clients = range(1, self.num_nodes)
        self.u = m.addVars(
            clients, name="u", vtype=GRB.CONTINUOUS, lb=self.demands[1:], ub=self.capacity
        )
        
        m.modelSense = GRB.MINIMIZE
        
        m.addConstrs((self.x.sum("*", j) == 1 for j in clients), name="in_degree")
        m.addConstrs((self.x.sum(i, "*") == 1 for i in clients), name="out_degree")
        m.addConstr(self.x.sum(0, "*") <= self.num_vehicles, name="max_vehicles")

        for i in clients:
            for j in clients:
                if i != j:
                    m.addConstr(
                        self.u[i] - self.u[j] + self.capacity * self.x[i, j] <= self.capacity - self.demands[j],
                        name=f"MTZ_{i}_{j}"
                    )
        return m, self.x

    def solve(self) -> tuple[np.ndarray, float]:
        self._model.update()
        self._model.optimize()
        sol = np.zeros(self.num_cost)
        if self._model.status == GRB.OPTIMAL:
            for idx, (i, j) in enumerate(self.edges):
                sol[idx] = self.x[i, j].X
        else:
            raise RuntimeError(f"Model has no optimal solution. Status: {self._model.status}")
        return sol, self._model.objVal

    def relax(self) -> NoReturn:
        raise RuntimeError("Model has already been relaxed.")


class cvrpDFJModel(optModel):
    """
    Dantzig-Fulkerson-Johnson (DFJ) formulation with capacity cuts.
    Note: Exponential number of constraints, use only for small instances.
    """

    def __init__(
        self,
        num_nodes: int,
        demands: np.ndarray | list,
        capacity: float,
        num_vehicles: int
    ) -> None:
        self.num_nodes = num_nodes
        self.demands = demands
        self.capacity = capacity
        self.num_vehicles = num_vehicles
        self.edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
        super().__init__()

    @property
    def num_cost(self) -> int:
        return len(self.edges)

    def _getModel(self) -> tuple:
        m = gp.Model("cvrp_dfj")
        m.setParam("OutputFlag", 0)
        
        self.x = m.addVars(self.edges, name="x", vtype=GRB.BINARY)
        clients = list(range(1, self.num_nodes))
        
        m.modelSense = GRB.MINIMIZE
        
        m.addConstrs((self.x.sum("*", j) == 1 for j in clients), name="in_degree")
        m.addConstrs((self.x.sum(i, "*") == 1 for i in clients), name="out_degree")
        m.addConstr(self.x.sum(0, "*") <= self.num_vehicles, name="max_vehicles")

        # DFJ capacity cuts
        for r in range(1, len(clients) + 1):
            for subset in combinations(clients, r):
                sub_demands = sum(self.demands[i] for i in subset)
                k_S = math.ceil(sub_demands / self.capacity)
                if k_S > 0:
                    m.addConstr(
                        gp.quicksum(self.x[i, j] for i in subset for j in subset if i != j)
                        <= len(subset) - k_S,
                        name=f"DFJ_{subset}"
                    )
        return m, self.x

    def setObj(self, c: np.ndarray | torch.Tensor | list) -> None:
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector cannot match number of edges.")
        obj = gp.quicksum(c[idx] * self.x[i, j] for idx, (i, j) in enumerate(self.edges))
        self._model.setObjective(obj)

    def solve(self) -> tuple[np.ndarray, float]:
        self._model.update()
        self._model.optimize()
        sol = np.zeros(self.num_cost)
        if self._model.status == GRB.OPTIMAL:
            for idx, (i, j) in enumerate(self.edges):
                sol[idx] = round(self.x[i, j].X)
        else:
            raise RuntimeError(f"Model has no optimal solution. Status: {self._model.status}")
        return sol, self._model.objVal

    def relax(self) -> cvrpDFJModelRel:
        return cvrpDFJModelRel(self.num_nodes, self.demands, self.capacity, self.num_vehicles)


class cvrpDFJModelRel(cvrpDFJModel):
    """
    LP relaxation of the DFJ formulation.
    """

    def _getModel(self) -> tuple:
        m = gp.Model("cvrp_dfj_rel")
        m.setParam("OutputFlag", 0)
        
        self.x = m.addVars(self.edges, name="x", vtype=GRB.CONTINUOUS, lb=0, ub=1)
        clients = list(range(1, self.num_nodes))
        
        m.modelSense = GRB.MINIMIZE
        
        m.addConstrs((self.x.sum("*", j) == 1 for j in clients), name="in_degree")
        m.addConstrs((self.x.sum(i, "*") == 1 for i in clients), name="out_degree")
        m.addConstr(self.x.sum(0, "*") <= self.num_vehicles, name="max_vehicles")

        for r in range(1, len(clients) + 1):
            for subset in combinations(clients, r):
                sub_demands = sum(self.demands[i] for i in subset)
                k_S = math.ceil(sub_demands / self.capacity)
                if k_S > 0:
                    m.addConstr(
                        gp.quicksum(self.x[i, j] for i in subset for j in subset if i != j)
                        <= len(subset) - k_S,
                        name=f"DFJ_{subset}"
                    )
        return m, self.x

    def solve(self) -> tuple[np.ndarray, float]:
        self._model.update()
        self._model.optimize()
        sol = np.zeros(self.num_cost)
        if self._model.status == GRB.OPTIMAL:
            for idx, (i, j) in enumerate(self.edges):
                sol[idx] = self.x[i, j].X
        else:
            raise RuntimeError(f"Model has no optimal solution. Status: {self._model.status}")
        return sol, self._model.objVal

    def relax(self) -> NoReturn:
        raise RuntimeError("Model has already been relaxed.")


class cvrpFlowModel(optModel):
    """
    Single-commodity flow formulation for the CVRP.
    Polynomial number of constraints, tighter than MTZ.
    """

    def __init__(
        self,
        num_nodes: int,
        demands: np.ndarray | list,
        capacity: float,
        num_vehicles: int
    ) -> None:
        self.num_nodes = num_nodes
        self.demands = demands
        self.capacity = capacity
        self.num_vehicles = num_vehicles
        self.edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
        super().__init__()

    @property
    def num_cost(self) -> int:
        return len(self.edges)

    def _getModel(self) -> tuple:
        m = gp.Model("cvrp_flow")
        m.setParam("OutputFlag", 0)
        
        # routing variables
        self.x = m.addVars(self.edges, name="x", vtype=GRB.BINARY)
        # flow variables
        self.f = m.addVars(self.edges, name="f", vtype=GRB.CONTINUOUS, lb=0)
        clients = range(1, self.num_nodes)
        
        m.modelSense = GRB.MINIMIZE
        
        m.addConstrs((self.x.sum("*", j) == 1 for j in clients), name="in_degree")
        m.addConstrs((self.x.sum(i, "*") == 1 for i in clients), name="out_degree")
        m.addConstr(self.x.sum(0, "*") <= self.num_vehicles, name="max_vehicles")

        # flow conservation
        for i in clients:
            m.addConstr(
                self.f.sum("*", i) - self.f.sum(i, "*") == self.demands[i],
                name=f"flow_cons_{i}"
            )
            
        # capacity coupling
        for i, j in self.edges:
            m.addConstr(self.f[i, j] <= self.capacity * self.x[i, j], name=f"cap_{i}_{j}")

        return m, self.x

    def setObj(self, c: np.ndarray | torch.Tensor | list) -> None:
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector cannot match number of edges.")
        obj = gp.quicksum(c[idx] * self.x[i, j] for idx, (i, j) in enumerate(self.edges))
        self._model.setObjective(obj)

    def solve(self) -> tuple[np.ndarray, float]:
        self._model.update()
        self._model.optimize()
        sol = np.zeros(self.num_cost)
        if self._model.status == GRB.OPTIMAL:
            for idx, (i, j) in enumerate(self.edges):
                sol[idx] = round(self.x[i, j].X)
        else:
            raise RuntimeError(f"Model has no optimal solution. Status: {self._model.status}")
        return sol, self._model.objVal

    def relax(self) -> cvrpFlowModelRel:
        return cvrpFlowModelRel(self.num_nodes, self.demands, self.capacity, self.num_vehicles)


class cvrpFlowModelRel(cvrpFlowModel):
    """
    LP relaxation of the Single-commodity flow formulation.
    """

    def _getModel(self) -> tuple:
        m = gp.Model("cvrp_flow_rel")
        m.setParam("OutputFlag", 0)
        
        self.x = m.addVars(self.edges, name="x", vtype=GRB.CONTINUOUS, lb=0, ub=1)
        self.f = m.addVars(self.edges, name="f", vtype=GRB.CONTINUOUS, lb=0)
        clients = range(1, self.num_nodes)
        
        m.modelSense = GRB.MINIMIZE
        
        m.addConstrs((self.x.sum("*", j) == 1 for j in clients), name="in_degree")
        m.addConstrs((self.x.sum(i, "*") == 1 for i in clients), name="out_degree")
        m.addConstr(self.x.sum(0, "*") <= self.num_vehicles, name="max_vehicles")

        for i in clients:
            m.addConstr(
                self.f.sum("*", i) - self.f.sum(i, "*") == self.demands[i],
                name=f"flow_cons_{i}"
            )
            
        for i, j in self.edges:
            m.addConstr(self.f[i, j] <= self.capacity * self.x[i, j], name=f"cap_{i}_{j}")

        return m, self.x

    def solve(self) -> tuple[np.ndarray, float]:
        self._model.update()
        self._model.optimize()
        sol = np.zeros(self.num_cost)
        if self._model.status == GRB.OPTIMAL:
            for idx, (i, j) in enumerate(self.edges):
                sol[idx] = self.x[i, j].X
        else:
            raise RuntimeError(f"Model has no optimal solution. Status: {self._model.status}")
        return sol, self._model.objVal

    def relax(self) -> NoReturn:
        raise RuntimeError("Model has already been relaxed.")


class cvrpAPModel(optModel):
    """
    Assignment Problem (AP) relaxation for the CVRP.
    Ignores subtours and capacity. Extremely loose but fast.
    """

    def __init__(
        self,
        num_nodes: int,
        demands: np.ndarray | list,
        capacity: float,
        num_vehicles: int
    ) -> None:
        self.num_nodes = num_nodes
        self.demands = demands
        self.capacity = capacity
        self.num_vehicles = num_vehicles
        self.edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
        super().__init__()

    @property
    def num_cost(self) -> int:
        return len(self.edges)

    def _getModel(self) -> tuple:
        m = gp.Model("cvrp_ap")
        m.setParam("OutputFlag", 0)
        
        self.x = m.addVars(self.edges, name="x", vtype=GRB.BINARY)
        clients = range(1, self.num_nodes)
        
        m.modelSense = GRB.MINIMIZE
        
        # only match in/out degrees
        m.addConstrs((self.x.sum("*", j) == 1 for j in clients), name="in_degree")
        m.addConstrs((self.x.sum(i, "*") == 1 for i in clients), name="out_degree")
        m.addConstr(self.x.sum(0, "*") <= self.num_vehicles, name="max_vehicles")

        return m, self.x

    def setObj(self, c: np.ndarray | torch.Tensor | list) -> None:
        if len(c) != self.num_cost:
            raise ValueError("Size of cost vector cannot match number of edges.")
        obj = gp.quicksum(c[idx] * self.x[i, j] for idx, (i, j) in enumerate(self.edges))
        self._model.setObjective(obj)

    def solve(self) -> tuple[np.ndarray, float]:
        self._model.update()
        self._model.optimize()
        sol = np.zeros(self.num_cost)
        if self._model.status == GRB.OPTIMAL:
            for idx, (i, j) in enumerate(self.edges):
                sol[idx] = round(self.x[i, j].X)
        else:
            raise RuntimeError(f"Model has no optimal solution. Status: {self._model.status}")
        return sol, self._model.objVal

    def relax(self) -> cvrpAPModelRel:
        return cvrpAPModelRel(self.num_nodes, self.demands, self.capacity, self.num_vehicles)


class cvrpAPModelRel(cvrpAPModel):
    """
    LP relaxation of the Assignment Problem (AP) formulation.
    """

    def _getModel(self) -> tuple:
        m = gp.Model("cvrp_ap_rel")
        m.setParam("OutputFlag", 0)
        
        self.x = m.addVars(self.edges, name="x", vtype=GRB.CONTINUOUS, lb=0, ub=1)
        clients = range(1, self.num_nodes)
        
        m.modelSense = GRB.MINIMIZE
        
        m.addConstrs((self.x.sum("*", j) == 1 for j in clients), name="in_degree")
        m.addConstrs((self.x.sum(i, "*") == 1 for i in clients), name="out_degree")
        m.addConstr(self.x.sum(0, "*") <= self.num_vehicles, name="max_vehicles")

        return m, self.x

    def solve(self) -> tuple[np.ndarray, float]:
        self._model.update()
        self._model.optimize()
        sol = np.zeros(self.num_cost)
        if self._model.status == GRB.OPTIMAL:
            for idx, (i, j) in enumerate(self.edges):
                sol[idx] = self.x[i, j].X
        else:
            raise RuntimeError(f"Model has no optimal solution. Status: {self._model.status}")
        return sol, self._model.objVal

    def relax(self) -> NoReturn:
        raise RuntimeError("Model has already been relaxed.")