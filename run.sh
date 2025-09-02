python run_eval.py --data ScreenSpot_v2_mobile --model /opt/nas/p/mm/ie_env/cmx/HONOR_TEST4/models/CPT --mode all
python run_eval.py --data Os-Atlas-mobile --model /opt/nas/p/mm/ie_env/cmx/HONOR_TEST4/models/RFT --mode all
# rich
python run_eval.py --data Complex --model /opt/nas/p/mm/ie_env/cmx/HONOR_TEST4/models/RFT --mode all
python run_eval.py --data Handling_Exception --model /opt/nas/p/mm/ie_env/cmx/HONOR_TEST4/models/RFT --mode all
python run_eval.py --data Instruction --model /opt/nas/p/mm/ie_env/cmx/HONOR_TEST4/models/RFT --mode all
python run_eval.py --data Routine --model /opt/nas/p/mm/ie_env/cmx/HONOR_TEST4/models/RFT --mode all
# 多步
python run_eval.py --data AC-Low --model /opt/nas/p/mm/ie_env/cmx/HONOR_TEST4/models/RFT --mode all
python run_eval.py --data AC-High --model /opt/nas/p/mm/ie_env/cmx/HONOR_TEST4/models/RFT --mode all
python run_eval.py --data GUI-Odyssey --model /opt/nas/p/mm/ie_env/cmx/HONOR_TEST4/models/RFT --mode all

python run_eval.py --data ScreenQA-short --model /opt/nas/p/mm/ie_env/cmx/HONOR_TEST5/models/RFT  --mode all --eval_model_path /opt/nas/p/mm/ie_env/cmx/HONOR_TEST5/models/Qwen2.5-7B-Instruct