import pandas as pd
import numpy as np

from collections import defaultdict, deque
from dataclasses import dataclass, field

# =========================================================
# RECENT TRANSACTION STATE
# =========================================================

WINDOW_1H = pd.Timedelta(hours=1)
WINDOW_24H = pd.Timedelta(hours=24)

@dataclass
class WindowState:
    """
    Stores transactions for one entity
    for 1-hour and 24-hour windows.
    """

    transactions_1h: deque = field(default_factory=deque)
    transactions_24h: deque = field(default_factory=deque)

    total_1h: float = 0.0
    total_24h: float = 0.0


class RecentTransactionState:
    """
    Stateful store for sliding-window features.

    Separate state is maintained for:
        - sender
        - receiver

    Each entity stores only recent transactions.
    """

    def __init__(self):

        self.sender = defaultdict(WindowState)
        self.receiver = defaultdict(WindowState)

        self.current_time = None

    # =====================================================
    # INTERNAL CLEANUP
    # =====================================================

    def _cleanup_entity(
        self,
        state: WindowState,
        current_time: pd.Timestamp
    ):
        """
        Remove transactions that are outside 1h / 24h windows.
        """

        cutoff_1h = current_time - WINDOW_1H
        cutoff_24h = current_time - WINDOW_24H

        # -------------------------
        # Remove expired 1h
        # -------------------------

        while (
            state.transactions_1h
            and state.transactions_1h[0][0] <= cutoff_1h
        ):
            _, amount = state.transactions_1h.popleft()

            state.total_1h -= amount

        # -------------------------
        # Remove expired 24h
        # -------------------------

        while (
            state.transactions_24h
            and state.transactions_24h[0][0] <= cutoff_24h
        ):
            _, amount = state.transactions_24h.popleft()

            state.total_24h -= amount

    # =====================================================
    # ADD TRANSACTION
    # =====================================================

    def add_transaction(
        self,
        entity_type: str,
        entity_id,
        dt: pd.Timestamp,
        amount: float
    ):
        """
        Add one transaction AFTER prediction.

        entity_type:
            'sender'
            'receiver'
        """

        if entity_type == "sender":
            state = self.sender[entity_id]

        elif entity_type == "receiver":
            state = self.receiver[entity_id]

        else:
            raise ValueError(
                "entity_type must be 'sender' or 'receiver'"
            )

        # Clean expired transactions first
        self._cleanup_entity(
            state,
            dt
        )

        # --------------------------------
        # Add to 24h window
        # --------------------------------

        state.transactions_24h.append(
            (dt, float(amount))
        )

        state.total_24h += float(amount)

        # --------------------------------
        # Add to 1h window
        # --------------------------------

        state.transactions_1h.append(
            (dt, float(amount))
        )

        state.total_1h += float(amount)

    # =====================================================
    # GET WINDOW FEATURES
    # =====================================================

    def get_features(
        self,
        entity_type: str,
        entity_id,
        current_time: pd.Timestamp
    ):
        """
        Get 1h / 24h window aggregates BEFORE
        processing the current transaction.
        """

        if entity_type == "sender":
            state = self.sender[entity_id]

        elif entity_type == "receiver":
            state = self.receiver[entity_id]

        else:
            raise ValueError(
                "entity_type must be 'sender' or 'receiver'"
            )

        # Remove expired transactions
        self._cleanup_entity(
            state,
            current_time
        )

        # --------------------------------
        # 1 hour
        # --------------------------------

        count_1h = len(
            state.transactions_1h
        )

        total_1h = state.total_1h

        avg_1h = (
            total_1h / count_1h
            if count_1h > 0
            else 0.0
        )

        # --------------------------------
        # 24 hour
        # --------------------------------

        count_24h = len(
            state.transactions_24h
        )

        total_24h = state.total_24h

        avg_24h = (
            total_24h / count_24h
            if count_24h > 0
            else 0.0
        )

        return {
            "count_1h": count_1h,
            "total_amount_1h": total_1h,
            "avg_amount_1h": avg_1h,

            "count_24h": count_24h,
            "total_amount_24h": total_24h,
            "avg_amount_24h": avg_24h,
        }