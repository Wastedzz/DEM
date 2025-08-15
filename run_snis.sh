#!/bin/bash

# GMM实验配置：(dimensionality, n_mixes)
# 运行配置: (32, 4) (32, 8) (64, 8)
# SNIS配置: 2, 5, 10

# 定义实验参数
experiments=(
    # "32 4"
    # "32 8" 
    "64 8"
)

# 定义SNIS参数
snis_values=(2 5 10)

echo "开始运行GMM SNIS多配置实验..."

# 循环运行所有实验
for exp in "${experiments[@]}"; do
    # 解析参数
    read -r dim n_mixes <<< "$exp"
    
    echo "========================================"
    echo "运行实验组: 维度=${dim}, 混合数=${n_mixes}"
    echo "========================================"
    
    # 使用Hydra multirun同时运行所有SNIS配置
    python dem/train.py \
        --multirun \
        experiment=gmm_rebuttal \
        paths=rebuttal \
        energy.dimensionality=${dim} \
        energy.n_mixes=${n_mixes} \
        model.num_samples_to_snis=2,5,10 \
        task_name="snis_${dim}_${n_mixes}"
    
    # 检查上一个命令的退出状态
    if [ $? -eq 0 ]; then
        echo "✅ 实验组 (${dim}, ${n_mixes}) 所有SNIS配置完成成功"
    else
        echo "❌ 实验组 (${dim}, ${n_mixes}) 部分或全部SNIS配置失败"
        # 可选：在失败时继续下一个实验，或者退出
        # exit 1  # 取消注释这行如果希望在失败时停止所有实验
    fi
    
    echo ""
done

echo "🎉 所有GMM SNIS多配置实验完成！"