import streamlit as st
import pandas as pd
import numpy as np
import torch
from PIL import Image
import os
from pathlib import Path
import pickle
from fastai.vision.all import *
from fastai.collab import *
from fastai.learner import load_learner
import matplotlib.pyplot as plt
from datetime import datetime
from fastai.vision.all import vision_learner, resnet34, DataLoaders,DataBlock, ImageBlock, CategoryBlock, get_image_files,GrandparentSplitter, parent_label, Resize, aug_transforms,vision_learner, accuracy, ClassificationInterpretation

# 设置页面配置
st.set_page_config(
    page_title="饮食健康分析系统", 
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)
nutrition_data = pd.read_excel('food_nutrition.xlsx')
# 缓存资源加载
@st.cache_resource
def load_image_model():
    """加载图像识别模型"""
    try:
        # 假设模型是基于FastAI的CNN模型
        model_arch = resnet34
        item_tfms = Resize(224)
        affine_tfms = aug_transforms(do_flip=True, flip_vert=False, max_rotate=10.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2)
        dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=GrandparentSplitter(train_name='train', valid_name='valid'),
        get_y=parent_label,
        item_tfms=item_tfms,
        batch_tfms=affine_tfms
)
        dest_base_path = Path('UECFOOD100_splitted')
        dls = dblock.dataloaders(dest_base_path, bs=64) 
        model1 = vision_learner(dls, model_arch)
        model1.load_state_dict(torch.load('shibie_model.pth'))
        st.success("✅ 图像识别模型加载成功")
        return model1
        #return learn
    except Exception as e:
        st.error(f"图像识别模型加载失败: {e}")
        return None

@st.cache_resource
def load_recommendation_model():
    """加载推荐模型权重"""
    try:
        # 加载推荐模型
        #model_path = Path("models/best_nutrition.pth")
        #learn = load_learner(model_path)
        collab_data = pd.read_excel('user_food_data.xlsx')
        dls = CollabDataLoaders.from_df(
        collab_data,
        user_name="user_id",
        item_name="food_id",
        rating_name="rating",
        valid_pct=0.2,
        seed=42,
        )
        model2 = collab_learner(dls, n_factors=30, y_range=(0, 5))
        model2.load_state_dict(torch.load('models/best.pth'))
        st.success("✅ 推荐模型加载成功")
        return model2
        #return learn
    except Exception as e:
        st.error(f"推荐模型加载失败: {e}")
        return None

@st.cache_data
def load_nutrition_data():
    """加载食物营养成分表"""
    try:
        nutrition_data = pd.read_excel('food_nutrition.xlsx')
        # 确保food_id唯一
        nutrition_data['food_id'] = range(1, len(nutrition_data) + 1)
        st.success("✅ 营养成分表加载成功")
        return nutrition_data
    except Exception as e:
        st.error(f"营养数据加载失败: {e}")
        return None

# 图像识别功能
def identify_foods(image, model,nutrition_data):
    """使用图像识别模型分析食物"""
    if model is None:
        st.error("图像识别模型未加载，无法进行食物识别")
        return []
    
    try:
        # 进行预测
        pred_class, pred_idx, probs = model.predict(image)
        confidence = probs[pred_idx].item()
        food_info = nutrition_data[nutrition_data['Food Name'] == pred_class]
        if not food_info.empty:
            food_id = food_info.iloc[0]['food_id']
            return [{"food_id": food_id, "food_name": pred_class, "confidence": confidence}]
        else:
            st.warning(f"未找到食物 '{pred_class}' 的营养信息")
            return []
        #return [{"food_name": pred_class, "confidence": confidence}]
    except Exception as e:
        st.error(f"食物识别失败: {e}")
        return []

# 营养评估功能
def calculate_nutrition(gaps, nutrition_data,food_weight):
    """计算营养缺口"""
    daily_nutrition_goals = {
        "Protein (g)": 55,     # 蛋白质
        "Fat (g)": 65,         # 脂肪
        "Carbohydrates (g)": 300,  # 碳水化合物
        "Dietary Fiber (g)": 25,   # 膳食纤维
        "Calcium (mg)": 800,      # 钙
        "Iron (mg)": 15,          # 铁
        "Vitamin C (mg)": 100,    # 维生素C
        "Calories (kcal)": 2000   # 卡路里
    }
    
    # 初始化摄入为0
    intake = {nutrient: 0 for nutrient in daily_nutrition_goals}
    
    # 根据识别的食物计算摄入
    for food in gaps:
        food_info = nutrition_data[nutrition_data['Food Name'] == food["food_name"]]
        if not food_info.empty:
            food_info = food_info.iloc[0]
            weight_factor = food_weight / 100.0
            for nutrient in daily_nutrition_goals:
                if nutrient in food_info:
                    intake[nutrient] += food_info[nutrient] * weight_factor
    
    # 计算缺口
    nutrition_gaps = {}
    for nutrient, goal in daily_nutrition_goals.items():
        gap = goal - intake.get(nutrient, 0)
        # 只保留有缺口的营养素（正值）
        if gap > 0:
            nutrition_gaps[nutrient] = gap
    
    return nutrition_gaps, intake


# 推荐功能
def get_recommendations(identified_foods, nutrition_gaps,recommendation_model, nutrition_data,daily_nutrition_goals):
    """生成个性化营养推荐"""
    if recommendation_model is None:
        st.error("推荐模型未加载，无法生成推荐")
        return []
    # 检查nutrition_gaps类型（新增调试代码）
    if not isinstance(nutrition_gaps, dict):
        st.error(f"营养缺口数据类型错误: {type(nutrition_gaps).__name__}")
        return []
    try:
        # 准备协同过滤输入数据
        user_ratings = {}
        for food in identified_foods:
            # 确保food_id存在
            if 'food_id' not in food:
                continue
            # 假设评分基于识别置信度和营养贡献
            rating = food["confidence"] * 5  # 转换为1-5分
            user_ratings[food["food_id"]] = rating
        
        # 冷启动推荐
        dls = recommendation_model.dls
        device = recommendation_model.dls.device
        
        item_embs = recommendation_model.model.i_weight.weight.to(device)
        item_bias = recommendation_model.model.i_bias.weight.to(device)
        n_factors = item_embs.shape[1]
        y_range = recommendation_model.y_range
        
        # 构建已评分食物映射
        item2idx = {item: i for i, item in enumerate(dls.classes['food_id']) if item in user_ratings}
        if not item2idx:
            st.warning("未识别到已知食物，无法生成个性化推荐")
            # 推荐高营养密度食物作为默认
            top_nutrition_foods = nutrition_data.nlargest(5, 'Protein (g)')
            return [{"food_id": int(fid), "food_name": name} 
                   for fid, name in zip(top_nutrition_foods['food_id'], top_nutrition_foods['Food Name'])]
        
        idx_tensor = torch.tensor(list(item2idx.values()), device=device)
        rating_tensor = torch.tensor([user_ratings[item] for item in item2idx], device=device, dtype=torch.float32)
        
        # 初始化用户嵌入向量
        usr_emb = torch.randn(1, n_factors, device=device, requires_grad=True)
        usr_bias = torch.randn(1, 1, device=device, requires_grad=True)
        opt = torch.optim.Adam([usr_emb, usr_bias], lr=0.02)
        
        # 优化用户嵌入向量
        n_iter = 30
        for ep in range(n_iter):
            opt.zero_grad()
            pred_raw = (usr_emb * item_embs[idx_tensor]).sum(dim=1) + usr_bias.squeeze() + item_bias[idx_tensor]
            pred = torch.sigmoid(pred_raw) * (y_range[1] - y_range[0]) + y_range[0]
            loss = torch.mean((pred - rating_tensor) ** 2)
            loss.backward()
            opt.step()
        
        usr_emb, usr_bias = usr_emb.detach(), usr_bias.detach()
        all_items = dls.classes['food_id']
        rated_items = set(user_ratings.keys())
        unrated_mask = ~torch.tensor([item in rated_items for item in all_items])
        cands_idx = unrated_mask.nonzero().squeeze()
        
        if cands_idx.ndim == 0:
            cands_idx = cands_idx.unsqueeze(0)
        if len(cands_idx) == 0:
            st.warning("所有食物均已评分，无法生成新推荐")
            return []
        
        pred_raw = (usr_emb * item_embs[cands_idx]).sum(dim=1, keepdim=True) + usr_bias + item_bias[cands_idx]
        pred_scores = (torch.sigmoid(pred_raw.squeeze()) * (y_range[1] - y_range[0]) + y_range[0]).cpu()
        
        recommended_items = []
        for i, idx in enumerate(cands_idx):
            food_id = all_items[idx]
            food_info = nutrition_data[nutrition_data['food_id'] == food_id]
            if food_info.empty:
                continue  # 跳过无营养数据的食物
            
            food_name = food_info['Food Name'].values[0]
            nutrition_contribution = {}
            for nutrient, gap in nutrition_gaps.items():
                if gap > 0 and nutrient in nutrition_data.columns:
                    nutrition_contribution[nutrient] = nutrition_data[nutrition_data['food_id'] == food_id][nutrient].values[0]
            
            nutrition_boost = sum(nutrition_contribution.values()) / sum(daily_nutrition_goals.values()) * 10
            combined_score = pred_scores[i].item() + nutrition_boost
            
            recommended_items.append({
                'food_id': food_id,
                'food_name': food_name,
                'pred_score': pred_scores[i].item(),
                'nutrition_boost': nutrition_boost,
                'combined_score': combined_score,
                'nutrition_contribution': nutrition_contribution
            })
        
        return sorted(recommended_items, key=lambda x: x['combined_score'], reverse=True)[:5]
    
    except Exception as e:
        st.error(f"推荐生成失败: {e}")
        return []

# 显示营养缺口可视化
def visualize_nutrition_gaps(gaps):
    """可视化营养缺口"""
    nutrients = [nutrient for nutrient in gaps if gaps[nutrient] > 0]
    if not nutrients:
        st.info("所有营养摄入充足，无需补充")
        return
    
    gap_values = [gaps[nutrient] for nutrient in nutrients]
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.figure(figsize=(10, 6))
    plt.bar(nutrients, gap_values, color='skyblue')
    plt.title('营养缺口分析')
    plt.xlabel('营养素')
    plt.ylabel('缺口量')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(plt)

# 主界面函数
def main():
    # 初始化session state
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = True
        st.session_state.current_step = 1
        st.session_state.identified_foods = []
        st.session_state.nutrition_gaps = {}
        st.session_state.recommendations = []
        st.session_state.food_weight = 100  # 默认100克
        st.session_state.food_ratings = {} 
        st.session_state.start_rating = False  # 新增状态变量 
     # 显式检查food_ratings是否存在
    if 'food_ratings' not in st.session_state:
        st.session_state.food_ratings = {}
    st.title("🍽️ 快来寻找你的健康美食！")
    st.markdown("···>基于图像识别和个性化推荐的营养分析平台<···")
    
    # 加载模型和数据
    with st.sidebar:
        st.header("💡 小贴士:")
        st.markdown("保持均衡饮食对健康至关重要。多摄入蔬菜、水果和全谷物，控制盐、糖和脂肪的摄入。")
        st.header("📌 系统信息")
        
        # 加载模型
        image_model = load_image_model()
        recommendation_model = load_recommendation_model()
        nutrition_data = load_nutrition_data()
        
        if image_model and recommendation_model and nutrition_data is not None:
            st.info(f"**图像识别模型**: CNN在UEC FOOD 100上微调")
            st.info(f"**推荐模型**: FastAI协同过滤+营养规则引擎")
            st.info(f"**食物种类**: {len(nutrition_data)}种")
            
            st.markdown("---")
            st.header("📊 当前进度")
            progress = st.session_state.current_step / 3
            st.progress(progress)
            st.write(f"步骤 {st.session_state.current_step} / 3")
            
            st.markdown("---")
            st.success("✅ 系统初始化完成")
        else:
            st.error("❌ 模型或数据加载失败，请检查文件路径")
            return
    # 随机食物展示和评分收集
    st.subheader("小测试：🍔 随机食物评分")
    st.markdown("请为以下随机选择的食物进行评分（1-5分）")
    # 初始化临时评分存储
    if 'temp_ratings' not in st.session_state:
        st.session_state.temp_ratings = {}

# 从食物数据集中随机选择3个食物名称
    random_foods = nutrition_data['Food Name'].sample(3).tolist()

# 使用st.columns()创建3列布局
    cols = st.columns(3)
    for i, food in enumerate(random_foods):
        with cols[i]:
            st.markdown(f"**{food}**")
        # 检查是否已有临时评分，没有则使用默认值
            current_rating = st.session_state.temp_ratings.get(food, 3)
        # 创建评分滑块，使用key参数确保唯一性
            rating = st.slider(
                f"为 {food} 评分", 
                1, 5, 
                key=f"temp_rating_{food}", 
                value=current_rating
        )
        # 保存临时评分
            st.session_state.temp_ratings[food] = rating

# 确定评分按钮
    if st.button("确定评分", type="primary"):
        # 将临时评分保存到正式评分存储中
        for food, rating in st.session_state.temp_ratings.items():
            st.session_state.food_ratings[food] = rating
    
    # 显示成功消息
        st.success("评分已保存！")
    
    # 显示用户评分
        st.subheader("你对随机食物的评分：")
        for food, rating in st.session_state.food_ratings.items():
            st.markdown(f"- {food}: {rating} 分")
        st.subheader("干得漂亮!已收集到你的个性化口味~")
    st.markdown("🍔🍔🍔🍔🍔🍔🍔🍔🍔🍔🍔🍔🍔🍔🍔🍔")
    st.subheader("前往下方开始你的个性化健康饮食之旅吧！")

    # 步骤1: 上传餐盘照片
    if st.session_state.current_step >= 1:
        st.header("📸 步骤1: 上传餐盘照片")
        st.markdown("请上传包含食物的餐盘照片，系统将自动识别食物并分析营养成分")
        
        uploaded_file = st.file_uploader("选择照片", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # 显示上传的照片
            image = Image.open(uploaded_file)
            st.image(image, caption="上传的餐盘照片", use_container_width=True)
            
            # 进行食物识别
            if st.button("开始识别", type="primary"):
                with st.spinner("正在识别食物..."):
                    identified_foods = identify_foods(image, image_model,nutrition_data)
                    if identified_foods:
                        st.session_state.identified_foods = identified_foods
                        st.session_state.current_step = 2
                    else:
                        st.session_state.identified_foods = []
            
            if st.session_state.get('identified_foods'):
                st.success("食物识别成功！")
                st.write(f"已识别到食物！它是{st.session_state.identified_foods[0]['food_name']},概率为{st.session_state.identified_foods[0]['confidence']*100:.1f}%")
            elif 'identified_foods' in st.session_state and not st.session_state.identified_foods:
                st.warning("未识别到食物，请尝试上传清晰的餐盘照片")
      
    
    # 步骤2: 营养分析
    if st.session_state.current_step >= 2 and st.session_state.identified_foods:
        st.header("📊 步骤2: 营养成分分析")
        
        st.subheader("识别到的食物")
        for i, food in enumerate(st.session_state.identified_foods):
            food_info = nutrition_data[nutrition_data['Food Name'] == food["food_name"]]
            if not food_info.empty:
                food_info = food_info.iloc[0]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**食物 {i+1}:** {food_info['Food Name']}")
                with col2:
                    st.markdown(f"**置信度:** {food['confidence']*100:.1f}%")
                with col3:
                    st.markdown(f"**蛋白质:** {food_info['Protein (g)']:.1f}g")
        # 食物重量选择组件
        st.subheader("选择食用重量")
        st.markdown("请选择你大约食用了多少克该食物：")
        
        food_weight_slider = st.slider(
            "食用重量 (克)",
            min_value=50, 
            max_value=1000, 
            value='food_weight_slider' in st.session_state,
            step=50,
            key="food_weight_slider"
        )
        st.markdown("请点击下面的“分析营养”来计算你的营养摄入与缺口！")
        # 计算营养缺口
        if st.button("分析营养", type="primary"):
            st.session_state.food_weight = food_weight_slider
            with st.spinner("正在分析营养成分..."):
                nutrition_gaps, intake = calculate_nutrition(st.session_state.identified_foods, nutrition_data,st.session_state.food_weight)
                st.session_state.nutrition_gaps = nutrition_gaps
                st.session_state.intake = intake  # 新增intake存储
                st.session_state.show_nutrition = True
        
            if st.session_state.get('show_nutrition', False):
                st.subheader("营养摄入与缺口（基于{0}克食用量）".format(st.session_state.food_weight_slider))
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**营养摄入:**")
                    for nutrient, amount in st.session_state.intake.items():  # 改用session_state
                        st.markdown(f"- {nutrient}: {amount:.1f}")
            
                with col2:
                    st.markdown("**营养缺口:**")
                    for nutrient, gap in st.session_state.nutrition_gaps.items():
                        if gap > 0:
                            st.markdown(f"- {nutrient}: {gap:.1f} (缺乏)")
                        else:
                            st.markdown(f"- {nutrient}: 0 (充足)")
            
        visualize_nutrition_gaps(st.session_state.nutrition_gaps)
        st.session_state.current_step = 3
    
    # 步骤3: 个性化推荐
    if st.session_state.current_step >= 3 and st.session_state.nutrition_gaps:
        st.header("🍎 步骤3: 个性化营养推荐")
        st.markdown("根据你的营养缺口，系统为你推荐以下食物")
        
        if st.button("生成推荐", type="primary"):
            with st.spinner("正在生成个性化推荐..."):
                recommendations = get_recommendations(
                    st.session_state.identified_foods,
                    st.session_state.nutrition_gaps,
                    recommendation_model,
                    nutrition_data,daily_nutrition_goals={
                        "Protein (g)": 55,     # 蛋白质
                        "Fat (g)": 65,         # 脂肪
                        "Carbohydrates (g)": 300,  # 碳水化合物
                        "Dietary Fiber (g)": 25,   # 膳食纤维}
                    }
                )
                
                if recommendations:
                    st.session_state.recommendations = recommendations
                    st.success("推荐生成成功！以下是为你推荐的食物")
                    
                    for i, rec in enumerate(recommendations):
                        with st.expander(f"推荐 {i+1}: {rec['food_name']}"):
                            st.markdown(f"**预测满意度:** {rec['pred_score']:.2f}")
                            st.markdown(f"**营养提升:** {rec['nutrition_boost']:.2f}")
                            st.markdown("**每100克营养贡献:**")
                            for nutrient, amount in rec['nutrition_contribution'].items():
                                st.markdown(f"- {nutrient}: {amount:.1f}g")
                else:
                    st.warning("无法生成推荐，请重试")
    
    # 重新开始按钮
    if st.sidebar.button("🔄 重新开始"):
        for key in ['identified_foods', 'nutrition_gaps', 'recommendations','food_weight', 'food_ratings']:
            if key == 'identified_foods':
                st.session_state[key] = []
            elif key == 'food_weight':
                st.session_state[key] = 100
            elif key == 'food_ratings':
                st.session_state[key] = {}
            else:
                st.session_state[key] = {}
        st.session_state.current_step = 1
        st.rerun()

if __name__ == "__main__":
    main()