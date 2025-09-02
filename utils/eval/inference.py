import torch
import torch.distributed as dist
from ..smp import *
from ..model import Qwen2VLChat



def infer_data(model_name, work_dir, model_path, dataset, out_file, verbose=False):
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    # rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(len(dataset)))
    lt = len(sheet_indices)
    data = dataset.data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)
    model = Qwen2VLChat(model_path, min_pixels=4*28*28, max_pixels=768*28*28) if isinstance(model_path, str) else model_name
    model.set_dump_image(dataset.dump_image)

    for i in tqdm(range(lt)):
        idx = data.iloc[i]['index']
        if idx in res:
            continue
        struct = dataset.build_prompt(data.iloc[i])
        response = model.generate(message=struct, dataset=dataset_name)
        torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 20 == 0:
            dump(res, out_file)

    res = {k: res[k] for k in data_indices}
    dump(res, out_file)
    return model


# A wrapper for infer_data, do the pre & post processing
def infer_data_job(work_dir, model_name, model_path, dataset,result_file, verbose=False, ignore_failed=False):
    dataset_name = dataset.dataset_name

    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    out_file = osp.join(work_dir, f'{dataset_name}.pkl')
    
    model = infer_data(model_name=model_name, work_dir=work_dir, model_path=model_path, dataset=dataset, out_file=out_file, verbose=verbose)
    data_all = {}
    data_all.update(load(out_file))

    data = dataset.data
    for x in data['index']:
        if x not in data_all:
            print(f'Error data key {x} not in data all')
            return None
    data['prediction'] = [str(data_all[x]) for x in data['index']]

    dump(data, result_file)
    os.remove(out_file)

    return model
