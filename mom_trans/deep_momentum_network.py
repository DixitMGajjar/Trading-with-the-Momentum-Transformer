import os
import json
import pathlib
import shutil
import copy

from keras_tuner.tuners import RandomSearch
from abc import ABC, abstractmethod

from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import collections

import keras_tuner as kt

from settings.hp_grid import (
    HP_HIDDEN_LAYER_SIZE,
    HP_DROPOUT_RATE,
    HP_MAX_GRADIENT_NORM,
    HP_LEARNING_RATE,
    HP_MINIBATCH_SIZE,
)

from settings.fixed_params import MODLE_PARAMS

from mom_trans.model_inputs import ModelFeatures
from empyrical import sharpe_ratio


class SharpeLoss(tf.keras.losses.Loss):
    def __init__(self, output_size: int = 1):
        self.output_size = output_size
        super().__init__()

    def call(self, y_true, weights):
        captured_returns = weights * y_true
        
        # More robust NaN prevention
        epsilon = 1e-6
        
        # Replace any NaN or Inf values with zeros
        captured_returns = tf.where(
            tf.math.is_finite(captured_returns), 
            captured_returns, 
            tf.zeros_like(captured_returns)
        )
        
        mean_returns = tf.reduce_mean(captured_returns)
        variance = tf.reduce_mean(tf.square(captured_returns)) - tf.square(mean_returns)
        
        # Ensure variance is positive and add epsilon
        variance = tf.maximum(variance, epsilon)
        
        sharpe = mean_returns / tf.sqrt(variance) * tf.sqrt(252.0)
        
        # Return negative sharpe (for minimization) with bounds
        return -tf.clip_by_value(sharpe, -10.0, 10.0)


class SharpeValidationLoss(keras.callbacks.Callback):
    def __init__(
        self,
        inputs,
        returns,
        time_indices,
        num_time,
        early_stopping_patience,
        n_multiprocessing_workers,
        weights_save_location="tmp/checkpoint",
        min_delta=1e-4,
    ):
        super(keras.callbacks.Callback, self).__init__()
        self.inputs = inputs
        self.returns = returns
        self.time_indices = time_indices
        self.n_multiprocessing_workers = n_multiprocessing_workers
        self.early_stopping_patience = early_stopping_patience
        self.num_time = num_time
        self.min_delta = min_delta

        self.best_sharpe = -np.inf 
        self.weights_save_location = weights_save_location
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(weights_save_location)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def set_weights_save_loc(self, weights_save_location):
        self.weights_save_location = weights_save_location
        # Create directory for the new location too
        directory = os.path.dirname(weights_save_location)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def on_train_begin(self, logs=None):
        self.patience_counter = 0
        self.stopped_epoch = 0
        self.best_sharpe = -np.inf 

    def on_epoch_end(self, epoch, logs=None):
        # Removed workers/multiprocessing args for Keras 3
        positions = self.model.predict(self.inputs)
        
        captured_returns = tf.math.unsorted_segment_mean(
            positions * self.returns, self.time_indices, self.num_time
        )[1:]

        sharpe = (
            tf.reduce_mean(captured_returns)
            / tf.sqrt(
                tf.math.reduce_variance(captured_returns)
                + tf.constant(1e-9, dtype=tf.float64)
            )
            * tf.sqrt(tf.constant(252.0, dtype=tf.float64))
        ).numpy()
        
        # Handle NaN values
        if np.isnan(sharpe) or np.isinf(sharpe):
            sharpe = -10.0 
            print(f"Warning: Invalid Sharpe value detected, setting to {sharpe}")
        
        if sharpe > self.best_sharpe + self.min_delta:
            self.best_sharpe = sharpe
            self.patience_counter = 0 
            self.model.save_weights(self.weights_save_location)
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if os.path.exists(self.weights_save_location):
                    self.model.load_weights(self.weights_save_location)
                else:
                    print(f"Warning: Weights file {self.weights_save_location} not found. Using current weights.")
        
        logs["sharpe"] = sharpe 
        print(f"\nval_sharpe {logs['sharpe']}")


class TunerValidationLoss(kt.tuners.RandomSearch):
    def __init__(
        self,
        hypermodel,
        objective,
        max_trials,
        hp_minibatch_size,
        seed=None,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        max_consecutive_failed_trials=3,
        **kwargs,
    ):
        self.hp_minibatch_size = hp_minibatch_size
        kwargs['max_consecutive_failed_trials'] = max_consecutive_failed_trials
        super().__init__(
            hypermodel,
            objective,
            max_trials,
            seed,
            hyperparameters,
            tune_new_entries,
            allow_new_entries,
            **kwargs,
        )

    def run_trial(self, trial, *args, **kwargs):
        kwargs["batch_size"] = trial.hyperparameters.Choice(
            "batch_size", values=self.hp_minibatch_size
        )
        super(TunerValidationLoss, self).run_trial(trial, *args, **kwargs)


class TunerDiversifiedSharpe(kt.tuners.RandomSearch):
    def __init__(
        self,
        hypermodel,
        objective,
        max_trials,
        hp_minibatch_size,
        seed=None,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        max_consecutive_failed_trials=3,
        **kwargs,
    ):
        self.hp_minibatch_size = hp_minibatch_size
        kwargs['max_consecutive_failed_trials'] = max_consecutive_failed_trials
        super().__init__(
            hypermodel,
            objective,
            max_trials,
            seed,
            hyperparameters,
            tune_new_entries,
            allow_new_entries,
            **kwargs,
        )

    def _get_checkpoint_fname(self, trial_id):
        """Return a path to save checkpoint weights for a trial."""
        os.makedirs(self.project_name, exist_ok=True)
        try:
            trial_id_int = int(trial_id)
            return os.path.join(self.project_name, f"trial_{trial_id_int:02d}.weights.h5")
        except ValueError:
            return os.path.join(self.project_name, f"trial_{trial_id}.weights.h5")

    def run_trial(self, trial, *args, **kwargs):
        kwargs["batch_size"] = trial.hyperparameters.Choice(
            "batch_size", values=self.hp_minibatch_size
        )

        original_callbacks = kwargs.pop("callbacks", [])

        for callback in original_callbacks:
            if isinstance(callback, SharpeValidationLoss):
                # FIXED: Removed self._reported_step argument to prevent AttributeError
                fname = self._get_checkpoint_fname(trial.trial_id)
                
                if not fname.endswith(".weights.h5"):
                    fname += ".weights.h5"
                callback.set_weights_save_loc(fname)
        
        metrics = collections.defaultdict(list)
        for execution in range(self.executions_per_trial):
            copied_fit_kwargs = copy.copy(kwargs)
            callbacks = self._deepcopy_callbacks(original_callbacks)
            self._configure_tensorboard_dir(callbacks, trial, execution)
            copied_fit_kwargs["callbacks"] = callbacks

            history = self._build_and_fit_model(trial, *args, **copied_fit_kwargs)
            for metric, epoch_values in history.history.items():
                if self.oracle.objective.direction == "min":
                    best_value = np.min(epoch_values)
                else:
                    best_value = np.max(epoch_values)
                metrics[metric].append(best_value)

        averaged_metrics = {}
        for metric, execution_values in metrics.items():
            averaged_metrics[metric] = np.mean(execution_values)
        self.oracle.update_trial(
            trial.trial_id, metrics=averaged_metrics
        )


class DeepMomentumNetworkModel(ABC):
    def __init__(self, project_name, hp_directory, hp_minibatch_size, **params):
        params = params.copy()

        self.time_steps = int(params["total_time_steps"])
        self.input_size = int(params["input_size"])
        self.output_size = int(params["output_size"])
        self.n_multiprocessing_workers = int(params["multiprocessing_workers"])
        self.num_epochs = int(params["num_epochs"])
        self.early_stopping_patience = int(params["early_stopping_patience"])
        self.random_search_iterations = params["random_search_iterations"]
        self.evaluate_diversified_val_sharpe = params["evaluate_diversified_val_sharpe"]
        self.force_output_sharpe_length = params["force_output_sharpe_length"]

        print("Deep Momentum Network params:")
        for k in params:
            print(f"{k} = {params[k]}")

        def model_builder(hp):
            return self.model_builder(hp)

        if self.evaluate_diversified_val_sharpe:
            self.tuner = TunerDiversifiedSharpe(
                model_builder,
                objective=kt.Objective("sharpe", "max"),
                hp_minibatch_size=hp_minibatch_size,
                max_trials=self.random_search_iterations,
                directory=hp_directory,
                project_name=project_name,
            )
        else:
            self.tuner = TunerValidationLoss(
                model_builder,
                objective="val_loss",
                hp_minibatch_size=hp_minibatch_size,
                max_trials=self.random_search_iterations,
                directory=hp_directory,
                project_name=project_name,
            )

    @abstractmethod
    def model_builder(self, hp):
        return

    @staticmethod
    def _index_times(val_time):
        val_time_unique = np.sort(np.unique(val_time))
        if val_time_unique[0]:  # check if ""
            val_time_unique = np.insert(val_time_unique, 0, "")
        mapping = dict(zip(val_time_unique, range(len(val_time_unique))))

        @np.vectorize
        def get_indices(t):
            return mapping[t]

        return get_indices(val_time), len(mapping)

    def hyperparameter_search(self, train_data, valid_data):
        try:
            data = train_data
            validation_data = valid_data

            if self.evaluate_diversified_val_sharpe:
                val_data, val_labels, val_flags, _, val_time = ModelFeatures._unpack(valid_data)
                val_time_indices, num_val_time = self._index_times(val_time)
                
                callbacks = [
                    SharpeValidationLoss(
                        val_data,
                        val_labels,
                        val_time_indices,
                        num_val_time,
                        self.early_stopping_patience,
                        self.n_multiprocessing_workers,
                    ),
                    tf.keras.callbacks.TerminateOnNaN(),
                ]
                
                self.tuner.search(
                    x=data["inputs"],
                    y=data["outputs"],
                    sample_weight=data.get("active_entries", None),
                    epochs=self.num_epochs,
                    callbacks=callbacks,
                    shuffle=True,
                    # Removed workers/multiprocessing for Keras 3 tuner compatibility
                )
            else:
                val_data, val_labels, val_flags, _, _ = ModelFeatures._unpack(valid_data)
                
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=self.early_stopping_patience,
                        min_delta=1e-4,
                    ),
                    tf.keras.callbacks.TerminateOnNaN(),
                ]
                
                self.tuner.search(
                    x=data["inputs"],
                    y=data["outputs"], 
                    sample_weight=data.get("active_entries", None),
                    epochs=self.num_epochs,
                    validation_data=(
                        val_data,
                        val_labels,
                        val_flags,
                    ),
                    callbacks=callbacks,
                    shuffle=True,
                    # Removed workers/multiprocessing for Keras 3 tuner compatibility
                )

            best_hp = self.tuner.get_best_hyperparameters(num_trials=1)[0].values
            best_model = self.tuner.get_best_models(num_models=1)[0]

            if best_hp is None or best_model is None:
                raise ValueError("Hyperparameter search failed to find valid parameters")

            return best_hp, best_model
        
        except Exception as e:
            print(f"Error during hyperparameter search: {e}")
            raise

    def load_model(
        self,
        hyperparameters,
    ) -> tf.keras.Model:
        hyp = kt.engine.hyperparameters.HyperParameters()
        hyp.values = hyperparameters
        return self.tuner.hypermodel.build(hyp)

    def fit(
        self,
        train_data: np.array,
        valid_data: np.array,
        hyperparameters,
        temp_folder: str,
    ):
        data, labels, active_flags, _, _ = ModelFeatures._unpack(train_data)
        val_data, val_labels, val_flags, _, val_time = ModelFeatures._unpack(valid_data)

        model = self.load_model(hyperparameters)

        if not temp_folder.endswith(".weights.h5"):
             temp_folder += ".weights.h5"

        if self.evaluate_diversified_val_sharpe:
            val_time_indices, num_val_time = self._index_times(val_time)
            callbacks = [
                SharpeValidationLoss(
                    val_data,
                    val_labels,
                    val_time_indices,
                    num_val_time,
                    self.early_stopping_patience,
                    self.n_multiprocessing_workers,
                    weights_save_location=temp_folder,
                ),
                tf.keras.callbacks.TerminateOnNaN(),
            ]
            
            # FIXED: Removed workers/use_multiprocessing args which cause TypeError in Keras 3
            model.fit(
                x=data,
                y=labels,
                sample_weight=active_flags,
                epochs=self.num_epochs,
                batch_size=hyperparameters["batch_size"],
                callbacks=callbacks,
                shuffle=True,
            )
            
            if os.path.exists(temp_folder):
                 model.load_weights(temp_folder)
        else:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.early_stopping_patience,
                    min_delta=1e-4,
                    restore_best_weights=True,
                ),
                tf.keras.callbacks.TerminateOnNaN(),
            ]
            
            # FIXED: Removed workers/use_multiprocessing args
            model.fit(
                x=data,
                y=labels,
                sample_weight=active_flags,
                epochs=self.num_epochs,
                batch_size=hyperparameters["batch_size"],
                validation_data=(
                    val_data,
                    val_labels,
                    val_flags,
                ),
                callbacks=callbacks,
                shuffle=True,
            )
        return model

    def evaluate(self, data, model):
        inputs, outputs, active_entries, _, _ = ModelFeatures._unpack(data)

        if self.evaluate_diversified_val_sharpe:
            _, performance = self.get_positions(data, model, False)
            return performance

        else:
            # FIXED: Removed workers/use_multiprocessing args
            metric_values = model.evaluate(
                x=inputs,
                y=outputs,
                sample_weight=active_entries,
            )

            metrics = pd.Series(metric_values, model.metrics_names)
            return metrics["loss"]

    def get_positions(
        self,
        data,
        model,
        sliding_window=True,
        years_geq=np.iinfo(np.int32).min,
        years_lt=np.iinfo(np.int32).max,
    ):
        inputs, outputs, _, identifier, time = ModelFeatures._unpack(data)
        if sliding_window:
            time = pd.to_datetime(
                time[:, -1, 0].flatten()
            ) 
            years = time.map(lambda t: t.year)
            identifier = identifier[:, -1, 0].flatten()
            returns = outputs[:, -1, 0].flatten()
        else:
            time = pd.to_datetime(time.flatten())
            years = time.map(lambda t: t.year)
            identifier = identifier.flatten()
            returns = outputs.flatten()
        mask = (years >= years_geq) & (years < years_lt)

        # FIXED: Removed workers/use_multiprocessing args
        positions = model.predict(inputs)
        
        if sliding_window:
            positions = positions[:, -1, 0].flatten()
        else:
            positions = positions.flatten()

        captured_returns = returns * positions
        results = pd.DataFrame(
            {
                "identifier": identifier[mask],
                "time": time[mask],
                "returns": returns[mask],
                "position": positions[mask],
                "captured_returns": captured_returns[mask],
            }
        )

        performance = sharpe_ratio(results.groupby("time")["captured_returns"].sum())

        return results, performance


class LstmDeepMomentumNetworkModel(DeepMomentumNetworkModel):
    def __init__(
        self, project_name, hp_directory, hp_minibatch_size=HP_MINIBATCH_SIZE, **params
    ):
        super().__init__(project_name, hp_directory, hp_minibatch_size, **params)

    def model_builder(self, hp):
        hidden_layer_size = hp.Choice("hidden_layer_size", values=HP_HIDDEN_LAYER_SIZE)
        dropout_rate = hp.Choice("dropout_rate", values=HP_DROPOUT_RATE)
        max_gradient_norm = hp.Choice("max_gradient_norm", values=HP_MAX_GRADIENT_NORM)
        learning_rate = hp.Choice("learning_rate", values=HP_LEARNING_RATE)

        input = keras.Input((self.time_steps, self.input_size))
        lstm = tf.keras.layers.LSTM(
            hidden_layer_size,
            return_sequences=True,
            dropout=dropout_rate,
            stateful=False,
            activation="tanh",
            recurrent_activation="sigmoid",
            recurrent_dropout=0,
            unroll=False,
            use_bias=True,
        )(input)
        dropout = keras.layers.Dropout(dropout_rate)(lstm)

        output = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(
                self.output_size,
                activation=tf.nn.tanh,
                kernel_constraint=keras.constraints.max_norm(3),
            )
        )(dropout[..., :, :])

        model = keras.Model(inputs=input, outputs=output)

        adam = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=max_gradient_norm)

        sharpe_loss = SharpeLoss(self.output_size).call

        model.compile(
            loss=sharpe_loss,
            optimizer=adam,
        )
        return model