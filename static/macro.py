#files for macro

PROFANITY = ['fuck',  'asshole', "bitch", "bullshit",
             "cunt", "motherfucker", "hell", "holy shit", 
             "nigga", "crap", "shit", "prick", "goddamn", "fucking", "slut"]


greetings_1 = "hello"
greetings_2 = "Hi there"
greetings_3 = "Hi"
greetings_4 = "Hello"
greetings_5 = "hi"
greetings_6 = "hey"
greetings_7 = "Hey"

GREETINGS = [greetings_1, greetings_2, greetings_3, greetings_4, greetings_5, greetings_6, greetings_7]

farewell_1 = 'bye'
farewell_2 = "okay bye"
farewell_3 = 'see you'
farewell_4 = 'thanks bye'
farewell_5 = "Bye"

FAREWELL = [farewell_1, farewell_2, farewell_3, farewell_4, farewell_5]

MODEL_ARGS = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 35,
    "train_batch_size": 4,
    "num_train_epochs": 5,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    "evaluate_generated_text": True,
    "evaluate_during_training": True,
    "evaluate_during_training_verbose": True,
    "save_best_model": True,
    "max_length": 35,
    'gradient_accumulation_steps': 2,
    'eval_batch_size': 4,
    "save_steps": 8000,
    "evaluate_during_training_steps": 8000,
    'use_multiprocessing': False,
    'fp16': True,
    'no_save': False
}
