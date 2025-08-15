# python dem/train.py experiment=gmm_rebuttal_32_2 task_name=snis_32_2_ratio_0.1 model.explore_ratio=0.1
# python dem/train.py experiment=gmm_rebuttal_32_2 task_name=snis_32_2_ratio_0.5 model.explore_ratio=0.5
# python dem/train.py experiment=gmm_rebuttal_32_2 task_name=snis_32_2_ratio_0.8 model.explore_ratio=0.8
# python dem/train.py experiment=gmm_rebuttal_32_2 task_name=snis_32_2_K_500 model.num_estimator_mc_samples=500
# python dem/train.py experiment=gmm_rebuttal_32_2 task_name=snis_32_2_K_1000 model.num_estimator_mc_samples=1000
# python dem/train.py experiment=gmm_rebuttal_32_2 task_name=snis_32_2_K_1500 model.num_estimator_mc_samples=1500
# python dem/train.py experiment=gmm_rebuttal_32_2 task_name=snis_32_2_L_2 model.num_samples_to_snis=2
# python dem/train.py experiment=gmm_rebuttal_32_2 task_name=snis_32_2_L_5 model.num_samples_to_snis=5
# python dem/train.py experiment=gmm_rebuttal_32_2 task_name=snis_32_2_L_10 model.num_samples_to_snis=10

# python dem/train.py experiment=gmm_rebuttal_32_2 task_name=idem_32_2 model.init_from_prior=false model.use_snis=false model.explore_ratio=0 model.num_init_samples=1024 model.num_samples_to_generate_per_epoch=1024
# python dem/train.py experiment=gmm_rebuttal_64_2 task_name=idem_64_2 model.init_from_prior=false model.use_snis=false model.explore_ratio=0 model.num_init_samples=1024 model.num_samples_to_generate_per_epoch=1024
# python dem/train.py experiment=gmm_rebuttal_64_8 task_name=idem_64_8 model.init_from_prior=false model.use_snis=false model.explore_ratio=0 model.num_init_samples=1024 model.num_samples_to_generate_per_epoch=1024


python dem/train.py experiment=gmm_rebuttal_64_8 task_name=snis_64_8_L_5 model.num_samples_to_snis=5 model.num_samples_to_sample_from_buffer=1280
python dem/train.py experiment=gmm_rebuttal_64_8 task_name=snis_64_8_L_10 model.num_samples_to_snis=10 model.num_samples_to_sample_from_buffer=512