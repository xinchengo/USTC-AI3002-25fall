import argparse
import yaml
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA as PCA_sklearn
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_from_disk
from submission import PCA, GMM


def main():
    parser = argparse.ArgumentParser(description="MNIST聚类实验davies_bouldin_score测评脚本")
    
    parser.add_argument("--results_path", type=str, default=None, required=True,
                       help="结果路径")
    
    args = parser.parse_args()
    
    print("="*70)
    print("MNIST聚类实验 - davies_bouldin_score测评")
    print("="*70)
    
    # 读取配置
    config_path = Path(args.results_path) / "config.yaml"
    if not config_path.exists():
        print(f"错误：未找到配置文件 {config_path}")
        return
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    print(f"\n配置信息：")
    for key, value in cfg.items():
        print(f"  {key}: {value}")
    
    # 加载数据集
    print("\n加载数据集...")
    try:
        dataset = load_from_disk("./data/mnist_encoded")
    except:
        print("错误：无法加载数据集，请确保数据文件存在")
        return
    
    trainset = dataset["train"].to_pandas()
    testset = dataset["test"].to_pandas()
    
    traindata_raw1d = np.vstack(trainset["image1D"].to_numpy())
    testdata_raw1d = np.vstack(testset["image1D"].to_numpy())
    
    print(f"训练集大小: {len(traindata_raw1d)}")
    print(f"测试集大小: {len(testdata_raw1d)}")
    
    # ====================== 模型测试 ======================
    print("\n" + "="*70)
    print("测试你的模型")
    print("="*70)
    
    try:
        print("\n使用PCA降维...")
        pca = PCA.from_pretrained(Path(args.results_path) / "pca.npz")
        testdata = pca.transform(testdata_raw1d)
        print(f"降维后维度: {testdata.shape[1]}")
        
        print("\n加载GMM模型并预测...")
        gmm = GMM.from_pretrained(Path(args.results_path) / "gmm")
        cluster_labels = gmm.predict(testdata)
        
        # 计算Davies-Bouldin Score
        db_score_student = davies_bouldin_score(testdata_raw1d, cluster_labels)
        print(f"\n✓ 你的模型 Davies-Bouldin Score: {db_score_student:.4f}")
        
    except Exception as e:
        print(f"\n✗ 错误：无法运行你的模型")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # ====================== 测试sklearn基准模型 ======================
    print("\n" + "="*70)
    print("测试sklearn基准模型")
    print("="*70)
    
    try:
        print("\n使用PCA降维...")
        pca_components = cfg.get("pca_components", 100)
        random_state = cfg.get("seed", 42)
        
        
        pca_sklearn = PCA_sklearn(n_components=pca_components, random_state=random_state)
        pca_sklearn.fit(traindata_raw1d)
        
        traindata_sklearn = pca_sklearn.transform(traindata_raw1d)
        testdata_sklearn = pca_sklearn.transform(testdata_raw1d)
        
        # GMM参数
        n_components = cfg.get("gmm_components", 10)
        max_iter = cfg.get("gmm_max_iter", 300)
        gmm_tol=cfg.get("gmm_tol", 1e-5)
        
        print(f"\n训练sklearn GMM (n_components={n_components}, max_iter={max_iter})...")
        gmm_sklearn = GaussianMixture(
            n_components=n_components,
            max_iter=max_iter,
            random_state=random_state,
            tol=gmm_tol
        )
        gmm_sklearn.fit(traindata_sklearn)
        cluster_labels_sklearn = gmm_sklearn.predict(testdata_sklearn)
        
        # 计算Davies-Bouldin Score
        db_score_sklearn = davies_bouldin_score(testdata_raw1d, cluster_labels_sklearn)
        print(f"\n✓ sklearn模型 Davies-Bouldin Score: {db_score_sklearn:.4f}")
        
    except Exception as e:
        print(f"\n✗ 错误：无法运行sklearn模型")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # ====================== 测评结果 ======================
    print("\n" + "="*70)
    print("测评结果")
    print("="*70)

    print(f"\nDavies-Bouldin Score 对比：")
    print(f"  学生模型: {db_score_student:.4f}")
    print(f"  sklearn:  {db_score_sklearn:.4f}")
    
    print("="*70)
    print("测评完成")
    print("="*70)


if __name__ == "__main__":
    main()