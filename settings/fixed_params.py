# # MODLE_PARAMS = {
# #     "architecture": "TFT",
# #     "total_time_steps": 252,
# #     "early_stopping_patience": 50,
# #     "multiprocessing_workers": 32,
# #     "num_epochs": 10,
# #     "early_stopping_patience": 32,
# #     "fill_blank_dates": False,
# #     "split_tickers_individually": True,
# #     "random_search_iterations": 1 ,
# #     "evaluate_diversified_val_sharpe": True,
# #     "train_valid_ratio": 0.90,
# #     "time_features": False,
# #     "force_output_sharpe_length": 0,
# # }

# # # MODLE_PARAMS = {
# # #     "total_time_steps": 252,
# # #     "input_size": 9,
# # #     "output_size": 1,
# # #     "num_epochs": 50,  # Changed from 300 to 50
# # #     "early_stopping_patience": 25,
# # #     "multiprocessing_workers": 32,
# # #     "fill_blank_dates": False,
# # #     "split_tickers_individually": True,
# # #     "random_search_iterations": 50,
# # #     "evaluate_diversified_val_sharpe": True,
# # #     "train_valid_ratio": 0.90,
# # #     "time_features": False,
# # #     "force_output_sharpe_length": None,
# # # }

# MODLE_PARAMS = {
#     "architecture": "TFT",
#     "total_time_steps": 252,
#     "multiprocessing_workers": 32,
#     "num_epochs": 300,                 # Reduced for 3-hour test run
#     "early_stopping_patience": 32,    # Stops if no improvement after 32 epochs
#     "fill_blank_dates": False,
#     "split_tickers_individually": True,
#     "random_search_iterations": 50,    # Set to 1 to skip tuning loops
#     "evaluate_diversified_val_sharpe": True,
#     "train_valid_ratio": 0.90,
#     "time_features": False,
#     "force_output_sharpe_length": 0,
# }

MODLE_PARAMS = {
    "architecture": "TFT",
    "total_time_steps": 252,
    "multiprocessing_workers": 16,    # Reduced from 32 to match your i7 threads (safer)
    "num_epochs": 60,                 # Enough for convergence, stops waste
    "early_stopping_patience": 10,    # Kills bad trials fast
    "fill_blank_dates": False,
    "split_tickers_individually": True,
    "random_search_iterations": 5,    # The "Student Strategy" compromise
    "evaluate_diversified_val_sharpe": True,
    "train_valid_ratio": 0.90,
    "time_features": False,
    "force_output_sharpe_length": 0,
}