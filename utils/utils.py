import os
from datetime import datetime


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_time_stamp() -> str:
    current_time = datetime.now()
    return current_time.strftime('%m%d%H%M%S')


def generate_descriptor(train_with_supervision, use_negative_samples, use_projector_mlp, train_end_to_end, **kwargs):
    """
        Function to generate the output folder name for logging with tensorboard
    """

    if train_with_supervision:
        descriptor = 'supervised'
    elif use_negative_samples:
        descriptor = 'negative_samples'
    else:
        descriptor = 'lpl'

    if use_projector_mlp:
        descriptor += '_mlp-projection'

    if not train_with_supervision and not use_negative_samples:
        if kwargs['no_pooling']:
            descriptor += '_no-pooling'

    if train_end_to_end:
        descriptor += '_end-to-end'

    return descriptor


def get_checkpoint_path_from_args(**kwargs):
    experiment_name = kwargs['experiment_name']
    model_descriptor = generate_descriptor(**kwargs)

    full_path_to_checkpoint = get_project_root()
    full_path_to_checkpoint = os.path.join(full_path_to_checkpoint, 'logs', kwargs['dataset'],
                                           experiment_name, model_descriptor, 'checkpoints')
    file_name = os.listdir(full_path_to_checkpoint)[0]
    full_path_to_checkpoint = os.path.join(full_path_to_checkpoint, file_name)

    return full_path_to_checkpoint
