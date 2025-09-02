import torch
from utils.dataset import build_dataset, DATASET_TYPE
from utils.eval import infer_data_job, convert_xlsx_to_jsonl, convert_to_template, evaluate
from utils.smp import *

def parse_args():
    parser = argparse.ArgumentParser()
    # Essential Args
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    # Work Dir
    parser.add_argument('--work-dir', type=str, default='./outputs', help='select the output directory')
    # Infer + Eval or Infer Only
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'infer'])
    # Logging Utils
    parser.add_argument('--verbose', action='store_true')
    # Eval_model_path
    parser.add_argument('--eval_model_path', type=str, default=None, help='Path to eval model (required if data is ScreenQA-short)')
    # Ignore: will not rerun failed VLM inference
    parser.add_argument('--ignore', action='store_true', help='Ignore failed indices. ')

    args = parser.parse_args()
    return args

def main():
    logger = get_logger('RUN')
    args = parse_args()
    if len(args.data)<=0:
        logger.error('Eval data should be a list of data files')
        return
    model_path = args.model[0]
    model_name = os.path.basename(os.path.normpath(model_path))
    eval_model_path = args.eval_model_path
    eval_id = timencommit()
    pred_root = osp.join(args.work_dir, model_name, eval_id)
    pred_root_meta = osp.join(args.work_dir, model_name)
    os.makedirs(pred_root_meta, exist_ok=True)
    prev_pred_roots = ls(osp.join(args.work_dir, model_name), mode='dir')

    if len(prev_pred_roots):
            prev_pred_roots.sort()
    
    if not osp.exists(pred_root):
            os.makedirs(pred_root, exist_ok=True)

    # Load dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_name = args.data[0]
    if dataset_name == 'ScreenQA-short' and eval_model_path is None and args.mode == 'all':
        logger.error(f'Eval_model_path is required when data is ScreenQA-short')

    dataset_path = os.path.join(f'{current_dir}/datasets',dataset_name)
    try:
        dataset = build_dataset(dataset_name, dataset_path)
        if dataset is None:
            logger.error(f'Dataset {dataset_name} is not valid. ')
            return

        result_file_base = f'{dataset_name}.xlsx'
        result_file = osp.join(pred_root, result_file_base)

        
        model = infer_data_job(
            work_dir=pred_root,
            model_name=model_name,
            model_path=model_path,
            dataset=dataset,
            verbose=args.verbose,
            ignore_failed=args.ignore,
            result_file=result_file)
        
        judge_kwargs = {
            'nproc': 4,
            'verbose': args.verbose,
            'retry': 3,
            'eval_model_path':eval_model_path
        }

        if args.mode == 'all':
            if dataset_name in ['AC-Low', 'AC-High', 'GUI-Odyssey']:
                model_infer_output_path = convert_xlsx_to_jsonl(result_file)
                template_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'utils','eval',f'{dataset_name}_template.jsonl')
                model_infer_output_path_modified = os.path.join(os.path.dirname(model_infer_output_path),f'{dataset_name}_modified.jsonl')
                convert_to_template(template_file_path, model_infer_output_path, model_infer_output_path_modified)
                final_result_path = os.path.join(os.path.dirname(model_infer_output_path),f'final_result_{dataset_name}')
                if dataset_name == 'AC-Low':
                    data_name='android_control_low_test'
                elif dataset_name == 'AC-High':
                    data_name='android_control_high_test'
                else:
                    data_name='gui_odyssey_test'
                final_result_args = argparse.Namespace(
                    seed=2000,
                    input_path=model_infer_output_path_modified,
                    output_dir=final_result_path,
                    data_name=data_name,
                    eval_android_control=True
                )
                evaluate(final_result_args)
                logger.info(f"Evaluation completed. Results saved to {final_result_path}")
            else:
                eval_result_file = dataset.evaluate(result_file, **judge_kwargs)
                logger.info(f"Evaluation completed. Results saved to {eval_result_file}")
        else:
            logger.info(f"Inference completed. Results saved to {result_file}")

    except Exception as e:
        logger.exception(f'Model {model_name} x Dataset {dataset_name} combination failed: {e}, '
                        'skipping this combination.')
    

if __name__ == '__main__':
    load_env()
    main()