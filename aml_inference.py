import numpy as np
import pandas as pd

from utils.recent_state import RecentTransactionState, WindowState

class AMLStatefulInference:

    def __init__(
        self,
        bundle,
        sender_state,
        receiver_state,
        sender_seen_receivers,
        receiver_seen_senders,
        recent_state,
        pair_state=None,
        categorical_features_dict=None,  # Opsional: jika ingin menyimpan dictionary validasi kategori
    ):
        # =====================================================
        # MODEL & METADATA LOADING
        # =====================================================

        
        self.pipeline = bundle["pipeline"]
        self.threshold = bundle.get("threshold", 0.5)

        self.numeric_features = bundle.get("numeric_features", [])
        self.categorical_features = bundle.get("categorical_features", [
            "is_cross_border",
            "is_currency_conversion",
            "Payment_currency",
            "Received_currency",
            "Sender_bank_location",
            "Receiver_bank_location",
            "Payment_type",
        ])
        self.feature_cols = bundle["feature_cols"]

        self.target_col = bundle.get("target_col")
        self.model_type = bundle.get("model_type")
        self.random_state = bundle.get("random_state")

        # Dictionary kategori valid (jika disediakan)
        self.categorical_features_dict = categorical_features_dict or bundle.get("categorical_features_dict", {})

        # =====================================================
        # STATE MANAGEMENT
        # =====================================================
        self.sender_state = sender_state
        self.receiver_state = receiver_state

        self.sender_seen_receivers = sender_seen_receivers
        self.receiver_seen_senders = receiver_seen_senders

        self.recent_state = recent_state
        self.pair_state = pair_state if pair_state is not None else {}
        # Posisi transaksi berikutnya yang akan diprediksi
        self.inference_position = 0

    def info(self):
        print(f"model threshold : {self.threshold}")
        print(f"model type : {self.model_type}")
        
    def reset_inference(self):
        self.inference_position = 0
        print("Inference cursor reset ke 0.")

    
    # =====================================================
    # CUMULATIVE FEATURES
    # =====================================================

    def _get_sender_cumulative_features(
        self,
        sender,
        current_time,
        amount,
    ):
        if sender in self.sender_state.index:
            state = self.sender_state.loc[sender]

            tx_count = state["sender_tx_count_history"]
            total_amount = state["sender_total_amount_history"]
            avg_amount = state["sender_avg_amount_history"]
            unique_receivers = state["sender_unique_receivers_history"]
            last_tx = state["sender_time_since_last_tx"]
        else:
            tx_count = 0
            total_amount = 0.0
            avg_amount = 0.0
            unique_receivers = 0
            last_tx = pd.NaT

        amount_vs_history_ratio = (
            amount / avg_amount if avg_amount > 0 else 0.0
        )

        if pd.isna(last_tx):
            time_since_last_tx = 0.0
        else:
            time_since_last_tx = (current_time - last_tx).total_seconds()

        return {
            "sender_tx_count_history": tx_count,
            "sender_total_amount_history": total_amount,
            "sender_avg_amount_history": avg_amount,
            "sender_amount_vs_history_ratio": amount_vs_history_ratio,
            "sender_unique_receivers_history": unique_receivers,
            "sender_time_since_last_tx": time_since_last_tx,
        }

    def _get_receiver_cumulative_features(
        self,
        receiver,
        current_time,
        amount,
    ):
        if receiver in self.receiver_state.index:
            state = self.receiver_state.loc[receiver]

            tx_count = state["receiver_tx_count_history"]
            total_amount = state["receiver_total_amount_history"]
            avg_amount = state["receiver_avg_amount_history"]
            unique_senders = state["receiver_unique_senders_history"]
            last_tx = state["receiver_time_since_last_tx"]
        else:
            tx_count = 0
            total_amount = 0.0
            avg_amount = 0.0
            unique_senders = 0
            last_tx = pd.NaT

        amount_vs_history_ratio = (
            amount / avg_amount if avg_amount > 0 else 0.0
        )

        if pd.isna(last_tx):
            time_since_last_tx = 0.0
        else:
            time_since_last_tx = (current_time - last_tx).total_seconds()

        return {
            "receiver_tx_count_history": tx_count,
            "receiver_total_amount_history": total_amount,
            "receiver_avg_amount_history": avg_amount,
            "receiver_amount_vs_history_ratio": amount_vs_history_ratio,
            "receiver_unique_senders_history": unique_senders,
            "receiver_time_since_last_tx": time_since_last_tx,
        }

    def _get_pair_cumulative_features(self, sender, receiver):
        pair_count = self.pair_state.get((sender, receiver), 0)
        return {
            "pair_tx_count_history": pair_count,
        }

    # =====================================================
    # WINDOWED FEATURES
    # =====================================================

    def _get_window_features(
        self,
        sender,
        receiver,
        current_time,
        amount,
    ):
        sender_window = self.recent_state.get_features(
            entity_type="sender",
            entity_id=sender,
            current_time=current_time,
        )

        receiver_window = self.recent_state.get_features(
            entity_type="receiver",
            entity_id=receiver,
            current_time=current_time,
        )

        s_avg_1h = sender_window["avg_amount_1h"]
        s_avg_24h = sender_window["avg_amount_24h"]
        r_avg_1h = receiver_window["avg_amount_1h"]
        r_avg_24h = receiver_window["avg_amount_24h"]

        sender_amount_vs_1h_avg = amount / s_avg_1h if s_avg_1h > 0 else 0.0
        sender_amount_vs_24h_avg = amount / s_avg_24h if s_avg_24h > 0 else 0.0
        receiver_amount_vs_1h_avg = amount / r_avg_1h if r_avg_1h > 0 else 0.0
        receiver_amount_vs_24h_avg = (
            amount / r_avg_24h if r_avg_24h > 0 else 0.0
        )

        return {
            "sender_tx_count_1h": sender_window["count_1h"],
            "sender_total_amount_1h": sender_window["total_amount_1h"],
            "sender_avg_amount_1h": s_avg_1h,
            "sender_tx_count_24h": sender_window["count_24h"],
            "sender_total_amount_24h": sender_window["total_amount_24h"],
            "sender_avg_amount_24h": s_avg_24h,
            "sender_amount_vs_1h_avg": sender_amount_vs_1h_avg,
            "sender_amount_vs_24h_avg": sender_amount_vs_24h_avg,
            "receiver_tx_count_1h": receiver_window["count_1h"],
            "receiver_total_amount_1h": receiver_window["total_amount_1h"],
            "receiver_avg_amount_1h": r_avg_1h,
            "receiver_tx_count_24h": receiver_window["count_24h"],
            "receiver_total_amount_24h": receiver_window["total_amount_24h"],
            "receiver_avg_amount_24h": r_avg_24h,
            "receiver_amount_vs_1h_avg": receiver_amount_vs_1h_avg,
            "receiver_amount_vs_24h_avg": receiver_amount_vs_24h_avg,
        }

    # =====================================================
    # BUILD ONE TRANSACTION'S FEATURES
    # =====================================================

    def build_features_for_transaction(self, row):
        dt = pd.to_datetime(row["Datetime"])
        sender = row["Sender_account"]
        receiver = row["Receiver_account"]
        amount = float(row["Amount"])

        # 1. CATEGORICAL FEATURES (Ekstraksi dari transaksi saat ini)
        cat_features = {}
        for col in self.categorical_features:
            if col in row:
                cat_features[col] = row[col]
            else:
                cat_features[col] = None

        # 2. CUMULATIVE FEATURES
        sender_features = self._get_sender_cumulative_features(
            sender=sender,
            current_time=dt,
            amount=amount,
        )

        receiver_features = self._get_receiver_cumulative_features(
            receiver=receiver,
            current_time=dt,
            amount=amount,
        )

        pair_features = self._get_pair_cumulative_features(
            sender=sender,
            receiver=receiver,
        )

        # 3. WINDOW FEATURES
        window_features = self._get_window_features(
            sender=sender,
            receiver=receiver,
            current_time=dt,
            amount=amount,
        )

        # 4. NUMERICAL TRANSACTION & DATETIME FEATURES
        transaction_features = {
            "Amount": amount,
            "log_amount": np.log1p(amount),
            "hour_sin": np.sin(2 * np.pi * dt.hour / 24),
            "hour_cos": np.cos(2 * np.pi * dt.hour / 24),
            "dow_sin": np.sin(2 * np.pi * dt.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * dt.dayofweek / 7),
        }

        # COMBINE ALL FEATURES
        features = {
            **cat_features,
            **transaction_features,
            **sender_features,
            **receiver_features,
            **pair_features,
            **window_features,
        }

        return features

    # =====================================================
    # STATE UPDATES
    # =====================================================

    def update_cumulative_state(self, sender, receiver, amount, dt):
        # Sender Update
        if sender in self.sender_state.index:
            s_count = (
                self.sender_state.loc[sender, "sender_tx_count_history"] + 1
            )
            s_total = (
                self.sender_state.loc[sender, "sender_total_amount_history"]
                + amount
            )
            s_unique = self.sender_state.loc[
                sender, "sender_unique_receivers_history"
            ]
        else:
            s_count = 1
            s_total = amount
            s_unique = 0

        if sender not in self.sender_seen_receivers:
            self.sender_seen_receivers[sender] = set()

        if receiver not in self.sender_seen_receivers[sender]:
            self.sender_seen_receivers[sender].add(receiver)
            s_unique += 1

        self.sender_state.loc[sender] = {
            "sender_tx_count_history": s_count,
            "sender_total_amount_history": s_total,
            "sender_avg_amount_history": s_total / s_count,
            "sender_unique_receivers_history": s_unique,
            "sender_time_since_last_tx": dt,
        }

        # Receiver Update
        if receiver in self.receiver_state.index:
            r_count = (
                self.receiver_state.loc[receiver, "receiver_tx_count_history"]
                + 1
            )
            r_total = (
                self.receiver_state.loc[
                    receiver, "receiver_total_amount_history"
                ]
                + amount
            )
            r_unique = self.receiver_state.loc[
                receiver, "receiver_unique_senders_history"
            ]
        else:
            r_count = 1
            r_total = amount
            r_unique = 0

        if receiver not in self.receiver_seen_senders:
            self.receiver_seen_senders[receiver] = set()

        if sender not in self.receiver_seen_senders[receiver]:
            self.receiver_seen_senders[receiver].add(sender)
            r_unique += 1

        self.receiver_state.loc[receiver] = {
            "receiver_tx_count_history": r_count,
            "receiver_total_amount_history": r_total,
            "receiver_avg_amount_history": r_total / r_count,
            "receiver_unique_senders_history": r_unique,
            "receiver_time_since_last_tx": dt,
        }

        # Pair State Update
        pair_key = (sender, receiver)
        self.pair_state[pair_key] = self.pair_state.get(pair_key, 0) + 1

    def update_state(self, row):
        dt = pd.to_datetime(row["Datetime"])
        sender = row["Sender_account"]
        receiver = row["Receiver_account"]
        amount = float(row["Amount"])

        # Cumulative
        self.update_cumulative_state(
            sender=sender,
            receiver=receiver,
            amount=amount,
            dt=dt,
        )

        # Recent
        self.recent_state.add_transaction(
            entity_type="sender",
            entity_id=sender,
            dt=dt,
            amount=amount,
        )

        self.recent_state.add_transaction(
            entity_type="receiver",
            entity_id=receiver,
            dt=dt,
            amount=amount,
        )

    # =====================================================
    # PREDICTION METHOD
    # =====================================================

    def predict(self, new_transactions):
        df = new_transactions.copy()
        df["Datetime"] = pd.to_datetime(df["Datetime"])

        # Urutkan berdasarkan waktu transaksi (kronologis)
        df = df.sort_values("Datetime").reset_index(drop=True)

        predictions = []

        for _, row in df.iterrows():
            # 1. READ CURRENT STATE & BUILD FEATURES
            features = self.build_features_for_transaction(row)

            # 2. MODEL INPUT (Pastikan urutan kolom sesuai self.feature_cols)
            X = pd.DataFrame([features])
            if self.feature_cols:
                X = X[self.feature_cols]

            # 3. PREDICT (Gunakan self.pipeline dan self.threshold dari bundle)
            probability = self.pipeline.predict_proba(X)[0, 1]
            prediction = int(probability >= self.threshold)

            # 4. STORE RESULT
            result = row.to_dict()
            result["aml_probability"] = probability
            result["prediction"] = prediction

            # Untuk debugging / melacak fitur saat transaksi terjadi
            result.update(features)

            predictions.append(result)

            # 5. UPDATE STATE SETELAH PREDIKSI (Stateful Stream Update)
            self.update_state(row)

        return pd.DataFrame(predictions)

    def predict_batch(self, new_transactions, n_rows=10):
        """
        Stateful batched inference.
    
        Parameters
        ----------
        new_transactions : pd.DataFrame
            Dataset transaksi yang akan digunakan untuk inference.
            Biasanya test_df.
    
        n_rows : int
            Berapa banyak transaksi berikutnya yang akan diprediksi
            pada call ini.
    
        Behavior
        --------
        Call 1, n_rows=10
            -> row 1-10
    
        Call 2, n_rows=15
            -> row 11-25
    
        Call 3, n_rows=20
            -> row 26-45
    
        State akan tetap dipertahankan antar call.
        """
    
        # =====================================================
        # VALIDATION
        # =====================================================
    
        if n_rows <= 0:
            raise ValueError("n_rows harus lebih besar dari 0.")
    
        df = new_transactions.copy()
    
        df["Datetime"] = pd.to_datetime(df["Datetime"])
    
        # Pastikan kronologis
        df = (
            df
            .sort_values("Datetime")
            .reset_index(drop=True)
        )
    
        total_rows = len(df)
    
        # =====================================================
        # CHECK WHETHER DATASET IS FINISHED
        # =====================================================
    
        if self.inference_position >= total_rows:
            print("Semua transaksi pada test set sudah diprediksi.")
            return pd.DataFrame()
    
        # =====================================================
        # DETERMINE CURRENT BATCH
        # =====================================================
    
        start_idx = self.inference_position
        end_idx = min(
            start_idx + n_rows,
            total_rows
        )
    
        batch_df = df.iloc[start_idx:end_idx].copy()
    
        print(
            f"Predicting rows "
            f"{start_idx + 1} - {end_idx} "
            f"of {total_rows}"
        )
    
        # =====================================================
        # BUILD FEATURES SEQUENTIALLY
        # =====================================================
    
        feature_rows = []
        original_rows = []
    
        for _, row in batch_df.iterrows():
    
            # ---------------------------------------------
            # 1. READ CURRENT STATE
            # ---------------------------------------------
    
            features = self.build_features_for_transaction(row)
    
            feature_rows.append(features)
            original_rows.append(row.to_dict())
    
            # ---------------------------------------------
            # 2. UPDATE STATE
            # ---------------------------------------------
            #
            # Penting:
            # Row berikutnya harus bisa melihat transaksi ini.
            #
            # Jadi meskipun model prediction dilakukan secara
            # batch, feature generation tetap sequential.
            #
    
            self.update_state(row)
    
        # =====================================================
        # CREATE BATCH MODEL INPUT
        # =====================================================
    
        X = pd.DataFrame(feature_rows)
    
        if self.feature_cols:
            X = X[self.feature_cols]
    
        # =====================================================
        # BATCH PREDICTION
        # =====================================================
    
        probabilities = self.pipeline.predict_proba(X)[:, 1]
    
        predictions = (
            probabilities >= self.threshold
        ).astype(int)
    
        # =====================================================
        # BUILD OUTPUT
        # =====================================================
    
        results = []
    
        for i in range(len(batch_df)):
    
            result = original_rows[i]
    
            result["aml_probability"] = probabilities[i]
            result["prediction"] = predictions[i]
    
            # Untuk debugging / explainability
            result.update(feature_rows[i])
    
            results.append(result)
    
        results_df = pd.DataFrame(results)
    
        # =====================================================
        # MOVE CURSOR
        # =====================================================
    
        self.inference_position = end_idx
    
        # =====================================================
        # INFO
        # =====================================================
    
        print(
            f"Cursor sekarang: "
            f"{self.inference_position}/{total_rows}"
        )
    
        print(
            f"Sisa transaksi: "
            f"{total_rows - self.inference_position}"
        )
    
        return results_df

    def remaining_rows(self, new_transactions):
        df = new_transactions.copy()
    
        return max(
            len(df) - self.inference_position,
            0
        )
        