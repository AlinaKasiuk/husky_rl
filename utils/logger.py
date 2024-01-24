import os
import torch



class BasicLogger:

    def __init__(self, project, name, exist_ok=False):
        exp_n = get_exp_n(project, name=name)
        if not exist_ok:
            exp_n += 1
        log_dir = os.path.join(project, "{0}{1}".format(name, exp_n))
        os.makedirs(log_dir, exist_ok=exist_ok)

    def save_model(self, model, model_name):
        torch.save(model, os.path.join(self.sw.log_dir, model_name + ".pt"))


def get_exp_n(project, name='exp'):
    if not os.path.exists(project):
        return 0
    ns = [
        int(f[len(name):]) for f in sorted(os.listdir(project)) if f.startswith(name) and str.isdigit(f[len(name):])
    ]
    return max(ns) if len(ns) else 0
