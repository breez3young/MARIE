marie_path="xx.pth"
# marie_path="/mnt/data/optimal/zhangyang/Projects/reprod_w_old/0503_repro_results/starcraft/3s_vs_3z_vq/run1/ckpt/model_final.pth"
tokenizer="vq"

mamba_path="xx.pth"

# 2m_vs_1z
map_name="3s_vs_5z"
env="starcraft"

python visualize_errors.py --env ${env} --map_name ${map_name} --tokenizer ${tokenizer} \
                           --marie_model_path ${marie_path} --mamba_model_path $mamba_path \
                           --eval_episodes 10 \
                           --ce_for_av --temperature 0.5 --horizon 25