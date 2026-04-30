# Cloudcast
paper_plots="/data1/pantea/Glia/scripts/Camera_ready_Engram"
python plot_methods_paper.py -a /data1/pantea/Glia/scripts/paper_json/cloudcast_my_prompt.json -o $paper_plots/cloudcast_my_prompt --max-num-sims 100 --problem-name cloudcast
python plot_methods_paper.py -a /data1/pantea/Glia/scripts/paper_json/cloudcast_simp_prompt.json -o $paper_plots/cloudcast_simp_prompt --max-num-sims 100 --problem-name cloudcast
python plot_methods_paper.py -a /data1/pantea/Glia/scripts/paper_json/cloudcast_simp_prompt_5.2.json -o $paper_plots/cloudcast_simp_prompt_5.2 --max-num-sims 100 --problem-name cloudcast
python plot_methods_paper.py -a /data1/pantea/Glia/scripts/paper_json/cloudcast_baseline_my_o3.json -o $paper_plots/cloudcast_baseline_my_o3 --max-num-sims 100 --problem-name cloudcast
python plot_methods_paper.py -a /data1/pantea/Glia/scripts/paper_json/vidur.json -o $paper_plots/vidur_baseline --max-num-sims 100 --problem-name vidur
python plot_methods_paper.py -a /data1/pantea/Glia/scripts/paper_json/cloudcast_motivation.json -o $paper_plots/cloudcast_motivation --max-num-sims 100 --problem-name cloudcast
python plot_methods_paper.py -a /data1/pantea/Glia/scripts/paper_json/cloudcast_ablation.json -o $paper_plots/cloudcast_ablation --max-num-sims 100 --problem-name cloudcast
python plot_methods_paper.py -a /data1/pantea/Glia/scripts/paper_json/cloudcast_simp_o3.json -o $paper_plots/cloudcast_simp_o3 --max-num-sims 100 --problem-name cloudcast
python plot_methods_paper.py -a /data1/pantea/Glia/scripts/paper_json/cloudcast_simp_gpt5.2.json -o $paper_plots/cloudcast_simp_gpt5.2 --max-num-sims 100 --problem-name cloudcast
python plot_methods_paper.py -a /data1/pantea/Glia/scripts/paper_json/cloudcast_my_prompt_5.2.json -o $paper_plots/cloudcast_my_prompt_5.2 --max-num-sims 100 --problem-name cloudcast
python plot_methods_paper.py -a paper_json/vidur_ablation.json -o $paper_plots/vidur_ablation --max-num-sims 100 --problem-name vidur
python plot_methods_paper.py -a /data1/pantea/Glia/scripts/paper_json/cloudcast_system_prompt_ablation.json -o $paper_plots/cloudcast_system_prompt_ablation --max-num-sims 100 --problem-name cloudcast
python plot_methods_paper.py -a /data1/pantea/Glia/scripts/paper_json/cloudcast_simp_gpt5.2_all.json -o $paper_plots/cloudcast_simp_gpt5.2_all --max-num-sims 100 --problem-name cloudcast