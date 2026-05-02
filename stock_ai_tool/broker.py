from __future__ import annotations

from typing import Dict, List

from .models import OrderRequest, Position, TradeExecution


class PaperBroker:
    def __init__(self, starting_cash: float) -> None:
        if starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        self.cash = float(starting_cash)
        self.positions: Dict[str, Position] = {}
        self.executions: List[TradeExecution] = []

    def execute(self, order: OrderRequest) -> TradeExecution:
        if order.quantity <= 0:
            raise ValueError("Order quantity must be positive")
        symbol = order.symbol.upper()
        total_cost = order.quantity * order.price

        if order.side == "BUY":
            if total_cost > self.cash:
                raise ValueError(
                    f"Insufficient cash for {symbol}: need {total_cost:.2f}, have {self.cash:.2f}"
                )
            self.cash -= total_cost
            pos = self.positions.get(symbol)
            if pos is None:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=order.quantity,
                    average_price=order.price,
                )
            else:
                new_qty = pos.quantity + order.quantity
                new_avg = ((pos.quantity * pos.average_price) + total_cost) / new_qty
                pos.quantity = new_qty
                pos.average_price = new_avg
            note = "Opened/increased long position"

        elif order.side == "SELL":
            pos = self.positions.get(symbol)
            if pos is None or pos.quantity < order.quantity:
                owned = 0 if pos is None else pos.quantity
                raise ValueError(
                    f"Insufficient position for {symbol}: trying to sell {order.quantity}, own {owned}"
                )
            self.cash += total_cost
            pos.quantity -= order.quantity
            if pos.quantity == 0:
                del self.positions[symbol]
            note = "Reduced/closed long position"
        else:
            raise ValueError(f"Unsupported side: {order.side}")

        execution = TradeExecution(
            symbol=symbol,
            side=order.side,
            quantity=order.quantity,
            fill_price=order.price,
            cash_after=round(self.cash, 2),
            note=note,
        )
        self.executions.append(execution)
        return execution

    def snapshot(self) -> dict:
        return {
            "cash": round(self.cash, 2),
            "positions": [
                {
                    "symbol": p.symbol,
                    "quantity": p.quantity,
                    "average_price": round(p.average_price, 4),
                }
                for p in self.positions.values()
            ],
            "executions": [
                {
                    "symbol": e.symbol,
                    "side": e.side,
                    "quantity": e.quantity,
                    "fill_price": e.fill_price,
                    "cash_after": e.cash_after,
                    "note": e.note,
                }
                for e in self.executions
            ],
        }

