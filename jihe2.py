import os
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.plot import reshape_as_raster
import math
from scipy import ndimage
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from rasterio.windows import Window
from skimage.transform import resize
import cv2


def create_folder(folder_path):
    """创建文件夹"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def process_prediction_masks(tif_files, base_folder):
    """
    处理预测掩膜，计算最大值和平均值

    Args:
        tif_files: 包含多个mask文件夹的列表
        base_folder: 基础文件夹路径
    """
    items = os.listdir(base_folder)

    for item in items:
        print(item)
        try:
            root_dir = os.path.join(base_folder, item)

            # 获取第一个文件夹中的文件作为参考
            folder_path = os.path.join(root_dir, tif_files[0])
            if not os.path.exists(folder_path):
                continue

            # 遍历文件夹中的文件
            for filename in os.listdir(folder_path):
                root_dir2 = os.path.join(folder_path, filename)

                # 使用GDAL打开图像获取基本信息
                from osgeo import gdal
                first_image = gdal.Open(root_dir2)
                cols = first_image.RasterXSize
                rows = first_image.RasterYSize
                bands = first_image.RasterCount

                # 创建数组存储最大值和总和
                max_values = np.zeros((rows, cols, bands))
                sum_values = np.zeros((rows, cols, bands))

                # 遍历所有mask文件夹计算最大值和平均值
                for tif_file in tif_files:
                    root_dir3 = os.path.join(root_dir, tif_file, filename)
                    ds = gdal.Open(root_dir3)
                    for band in range(1, bands + 1):
                        data = ds.GetRasterBand(band).ReadAsArray()
                        max_values[:, :, band - 1] = np.maximum(max_values[:, :, band - 1], data)
                        sum_values[:, :, band - 1] += data

                average_values = sum_values / len(tif_files)

                # 创建输出文件夹
                output_max_folder = os.path.join(root_dir, "output_max")
                output_average_folder = os.path.join(root_dir, "output_average")
                create_folder(output_max_folder)
                create_folder(output_average_folder)

                # 输出文件路径
                output_max_path = os.path.join(output_max_folder, filename)
                output_average_path = os.path.join(output_average_folder, filename)

                # 创建输出TIFF文件
                driver = gdal.GetDriverByName('GTiff')
                output_tiff_max = driver.Create(output_max_path, cols, rows, bands, gdal.GDT_Float32)
                output_tiff_avg = driver.Create(output_average_path, cols, rows, bands, gdal.GDT_Float32)

                # 写入数据
                for band in range(1, bands + 1):
                    output_tiff_max.GetRasterBand(band).WriteArray(max_values[:, :, band - 1])
                    output_tiff_avg.GetRasterBand(band).WriteArray(average_values[:, :, band - 1])

                # 设置地理参考信息
                output_tiff_max.SetGeoTransform(first_image.GetGeoTransform())
                output_tiff_max.SetProjection(first_image.GetProjection())
                output_tiff_avg.SetGeoTransform(first_image.GetGeoTransform())
                output_tiff_avg.SetProjection(first_image.GetProjection())

                # 关闭数据集
                output_tiff_max = None
                output_tiff_avg = None

        except Exception as e:
            print(f"处理 {item} 时出错: {e}")
            continue


def merge_tiff_georeference(input_folder, output_file):
    """
    使用rasterio.merge的自动拼接方法（保留地理坐标）
    """
    # 获取所有TIFF文件
    tiff_files = []
    for file in os.listdir(input_folder):
        if file.lower().endswith(('.tif', '.tiff')):
            tiff_files.append(os.path.join(input_folder, file))

    if not tiff_files:
        print("未找到TIFF文件")
        return

    tiff_files.sort()

    # 读取所有数据集
    src_files_to_mosaic = []
    for tiff_file in tiff_files:
        src = rasterio.open(tiff_file)
        src_files_to_mosaic.append(src)

    # 使用rasterio的merge函数
    mosaic, out_trans = merge(src_files_to_mosaic)

    # 获取输出元数据
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "crs": src_files_to_mosaic[0].crs
    })

    # 写入输出文件
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)

    print(f"地理拼接完成，输出文件: {output_file}")
    print(f"输出尺寸: {mosaic.shape[1]} × {mosaic.shape[2]}")
    print(f"坐标系统: {out_meta['crs']}")

    # 关闭所有数据集
    for src in src_files_to_mosaic:
        src.close()


def zhang_suen_thinning(img):
    """
    使用OpenCV形态学操作的Zhang-Suen细化算法
    输入: 二值图像 (0表示背景，1表示前景)
    输出: 细化后的二值图像
    """
    # 确保图像是二值的，并将1转换为255
    binary_img = (img > 0).astype(np.uint8) * 255

    size = np.size(binary_img)
    skel = np.zeros(binary_img.shape, np.uint8)

    # 应用阈值处理
    ret, img_thresh = cv2.threshold(binary_img, 0, 255, cv2.THRESH_BINARY)

    # 创建十字形结构元素
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        # 腐蚀图像
        eroded = cv2.erode(img_thresh, element)
        # 膨胀腐蚀后的图像
        temp = cv2.dilate(eroded, element)
        # 计算差异
        temp = cv2.subtract(img_thresh, temp)
        # 将差异添加到骨架
        skel = cv2.bitwise_or(skel, temp)
        # 更新图像为腐蚀后的版本
        img_thresh = eroded.copy()

        # 检查是否完成
        zeros = size - cv2.countNonZero(img_thresh)
        if zeros == size:
            done = True

    # 将结果转换回0和1
    skel_binary = (skel > 0).astype(np.uint8)

    return skel_binary


def process_classification_results(max_tiff_path, average_tiff_path, output_dir):
    """
    处理分类结果
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取max.tiff文件
    with rasterio.open(max_tiff_path) as src_max:
        max_data = src_max.read(1)  # 读取第一个波段
        max_profile = src_max.profile.copy()  # 复制元数据
        max_transform = src_max.transform  # 获取坐标变换信息
        max_crs = src_max.crs  # 获取坐标系

    # 读取average.tiff文件
    with rasterio.open(average_tiff_path) as src_avg:
        avg_data = src_avg.read(1)  # 读取第一个波段
        avg_profile = src_avg.profile.copy()  # 复制元数据
        avg_transform = src_avg.transform  # 获取坐标变换信息
        avg_crs = src_avg.crs  # 获取坐标系

    # 检查坐标系是否一致
    if max_crs != avg_crs:
        print("警告: 两个TIFF文件的坐标系不一致!")

    # 第一步: 将max.tiff按照50的值分成1,2两类
    max_classified = np.where(max_data >= 50, 2, 1)

    # 第二步: 将average.tiff按照50的值分成1,2两类
    avg_classified = np.where(avg_data >= 50, 2, 1)

    # 更新元数据以适应分类数据
    max_profile.update({
        'dtype': rasterio.uint8
    })

    avg_profile.update({
        'dtype': rasterio.uint8
    })

    # 保存分类后的max.tiff
    max_classified_path = os.path.join(output_dir, "max_classified.tif")
    with rasterio.open(max_classified_path, 'w', **max_profile) as dst:
        dst.write(max_classified.astype(rasterio.uint8), 1)

    # 保存分类后的average.tiff
    avg_classified_path = os.path.join(output_dir, "average_classified.tif")
    with rasterio.open(avg_classified_path, 'w', **avg_profile) as dst:
        dst.write(avg_classified.astype(rasterio.uint8), 1)

    print("已生成分类后的TIFF文件")

    # 第三步: 基于分类结果进行处理
    # 提取分类后的2类区域（>=50的区域）
    max_class2 = (max_classified == 2).astype(np.uint8)  # max分类中的2类
    avg_class2 = (avg_classified == 2).astype(np.uint8)  # average分类中的2类

    # 找到average分类中2类落在max分类中2类上的点
    avg_on_max_class2 = (avg_class2 == 1) & (max_class2 == 1)

    # 对max分类中的2类区域进行连通组件分析
    labeled_array, num_features = ndimage.label(max_class2)

    # 找到包含avg_on_max_class2点的连通组件
    valid_components = set()
    for y in range(labeled_array.shape[0]):
        for x in range(labeled_array.shape[1]):
            if avg_on_max_class2[y, x]:
                component_label = labeled_array[y, x]
                if component_label > 0:
                    valid_components.add(component_label)

    # 创建结果掩膜，只保留有效的连通组件
    result_mask = np.zeros_like(max_class2, dtype=bool)
    for label_num in valid_components:
        result_mask = result_mask | (labeled_array == label_num)

    # 创建最终结果，只保留有效连通组件中的值为1
    final_result = np.where(result_mask, 1, 0)  # 有效区域值为1，其他为0

    # 更新元数据
    result_profile = max_profile.copy()
    result_profile.update({
        'dtype': rasterio.uint8
    })

    # 保存最终结果
    result_path = os.path.join(output_dir, "final_result.tif")
    with rasterio.open(result_path, 'w', **result_profile) as dst:
        dst.write(final_result.astype(rasterio.uint8), 1)

    # 第四步: 应用Zhang-Suen细化算法
    print("应用Zhang-Suen细化算法...")

    # 对final_result进行细化
    final_result_thinned = zhang_suen_thinning(final_result)

    # 保存细化结果
    thinned_path = os.path.join(output_dir, "finalxfxt.tif")
    with rasterio.open(thinned_path, 'w', **result_profile) as dst:
        dst.write(final_result_thinned.astype(rasterio.uint8), 1)

    print("处理完成!")
    print(f"生成的分类文件保存在: {output_dir}")
    print(f"- max_classified.tif: max.tiff的分类结果 (1:<50, 2:≥50)")
    print(f"- average_classified.tif: average.tiff的分类结果 (1:<50, 2:≥50)")
    print(f"- final_result.tif: 最终处理结果 (有效区域值为1，其他为0)")
    print(f"- finalxf.tif: Zhang-Suen细化后的结果")

    # 输出统计信息
    total_avg_class2 = np.sum(avg_class2)
    total_max_class2 = np.sum(max_class2)
    overlap_points = np.sum(avg_on_max_class2)
    valid_pixels = np.sum(result_mask)
    thinned_pixels = np.sum(final_result_thinned)

    print(f"\n统计信息:")
    print(f"average分类中2类像元总数: {total_avg_class2}")
    print(f"max分类中2类像元总数: {total_max_class2}")
    print(f"average的2类落在max的2类上的像元数: {overlap_points}")
    print(f"最终保留的有效像元数: {valid_pixels}")
    print(f"细化后保留的像元数: {thinned_pixels}")
    print(f"有效连通组件数量: {len(valid_components)}")

    return {
        'max_classified': max_classified_path,
        'avg_classified': avg_classified_path,
        'final_result': result_path,
        'final_thinned': thinned_path  # 返回细化结果路径
    }


def calculate_metrics(y_true, y_pred, num_classes=2):
    """计算各种精度指标"""
    metrics = {}

    # 总体精度
    metrics['overall_accuracy'] = accuracy_score(y_true, y_pred)

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    metrics['confusion_matrix'] = cm

    # 各类别精度
    class_metrics = {}
    for class_id in range(num_classes):
        # 真正例、假正例、假负例
        tp = cm[class_id, class_id]
        fp = cm[:, class_id].sum() - tp
        fn = cm[class_id, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        # 精确率、召回率、F1分数、IoU
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

        class_metrics[class_id] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'iou': iou,
            'support': cm[class_id, :].sum()
        }

    metrics['class_metrics'] = class_metrics

    # 平均精度
    metrics['mean_precision'] = np.mean([class_metrics[cls]['precision'] for cls in range(num_classes)])
    metrics['mean_recall'] = np.mean([class_metrics[cls]['recall'] for cls in range(num_classes)])
    metrics['mean_f1'] = np.mean([class_metrics[cls]['f1_score'] for cls in range(num_classes)])
    metrics['mean_iou'] = np.mean([class_metrics[cls]['iou'] for cls in range(num_classes)])

    return metrics


def crop_to_same_extent(src1, src2):
    """裁剪两个栅格到相同的空间范围"""
    # 获取交集边界
    bounds1 = src1.bounds
    bounds2 = src2.bounds

    left = max(bounds1.left, bounds2.left)
    right = min(bounds1.right, bounds2.right)
    bottom = max(bounds1.bottom, bounds2.bottom)
    top = min(bounds1.top, bounds2.top)

    if left >= right or bottom >= top:
        return None, None

    # 转换到各自的行列号
    window1 = src1.window(left, bottom, right, top)
    window2 = src2.window(left, bottom, right, top)

    return window1, window2


def write_metrics_to_txt(metrics, result_name, txt_file_path):
    """将精度指标写入txt文件"""
    with open(txt_file_path, 'a', encoding='utf-8') as f:
        f.write(f"\n{result_name} 评估结果:\n")
        f.write("=" * 50 + "\n")
        f.write(f"总体精度: {metrics['overall_accuracy']:.4f}\n")
        f.write(f"平均精确率: {metrics['mean_precision']:.4f}\n")
        f.write(f"平均召回率: {metrics['mean_recall']:.4f}\n")
        f.write(f"平均F1分数: {metrics['mean_f1']:.4f}\n")
        f.write(f"平均IoU: {metrics['mean_iou']:.4f}\n")

        # 写入混淆矩阵
        f.write(f"\n混淆矩阵:\n")
        f.write(str(metrics['confusion_matrix']) + "\n")

        # 写入各类别指标
        f.write("\n各类别指标:\n")
        for class_id in range(2):  # 0,1两类
            class_metrics = metrics['class_metrics'][class_id]
            f.write(f"类别 {class_id}:\n")
            f.write(f"  精确率: {class_metrics['precision']:.4f}\n")
            f.write(f"  召回率: {class_metrics['recall']:.4f}\n")
            f.write(f"  F1分数: {class_metrics['f1_score']:.4f}\n")
            f.write(f"  IoU: {class_metrics['iou']:.4f}\n")
            f.write(f"  样本数: {class_metrics['support']}\n")

        f.write("\n" + "=" * 50 + "\n\n")


def evaluate_segmentation(result_tif_path, test_txt_path, label_folder, result_name="结果", txt_file_path=None):
    """评估语义分割结果 - 整体计算精度"""

    # 读取测试文件列表
    with open(test_txt_path, 'r') as f:
        lines = f.readlines()

    # 提取文件名
    test_files = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            label_path = parts[1]
            filename = os.path.basename(label_path).split('.')[0]  # 去除扩展名
            test_files.append(filename)

    print(f"找到 {len(test_files)} 个测试文件")

    # 收集所有真实标签和预测结果
    all_y_true = []
    all_y_pred = []

    # 打开结果文件
    with rasterio.open(result_tif_path) as result_src:
        result_profile = result_src.profile

        for filename in test_files:
            label_path = os.path.join(label_folder, f"{filename}.tiff")

            if not os.path.exists(label_path):
                print(f"警告: 标签文件 {label_path} 不存在，跳过")
                continue

            # 打开标签文件
            with rasterio.open(label_path) as label_src:
                # 裁剪到相同范围
                result_window, label_window = crop_to_same_extent(result_src, label_src)

                if result_window is None:
                    print(f"警告: {filename} 与结果图无重叠区域，跳过")
                    continue

                # 读取数据
                result_data = result_src.read(1, window=result_window)
                label_data = label_src.read(1, window=label_window)

                # 调整尺寸使其一致（如果行列数不同）
                if result_data.shape != label_data.shape:
                    print(f"调整尺寸: {result_data.shape} -> {label_data.shape}")
                    # 使用最近邻插值保持类别值
                    result_data = resize(result_data, label_data.shape, order=0,
                                         preserve_range=True, anti_aliasing=False).astype(result_data.dtype)

                # 收集所有像素
                all_y_true.extend(label_data.flatten())
                all_y_pred.extend(result_data.flatten())

    # 转换为numpy数组
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # 计算整体指标
    if len(all_y_true) > 0:
        print("\n" + "=" * 50)
        print(f"{result_name} 整体评估结果 (基于所有像素计算):")
        print("=" * 50)

        # 计算整体指标
        metrics = calculate_metrics(all_y_true, all_y_pred)

        print(f"总体精度: {metrics['overall_accuracy']:.4f}")
        print(f"平均精确率: {metrics['mean_precision']:.4f}")
        print(f"平均召回率: {metrics['mean_recall']:.4f}")
        print(f"平均F1分数: {metrics['mean_f1']:.4f}")
        print(f"平均IoU: {metrics['mean_iou']:.4f}")

        # 输出混淆矩阵
        print(f"\n混淆矩阵 (整体):")
        print(metrics['confusion_matrix'])

        # 各类别指标
        print("\n各类别指标 (整体):")
        for class_id in range(2):  # 0,1两类
            class_metrics = metrics['class_metrics'][class_id]
            print(f"类别 {class_id}:")
            print(f"  精确率: {class_metrics['precision']:.4f}")
            print(f"  召回率: {class_metrics['recall']:.4f}")
            print(f"  F1分数: {class_metrics['f1_score']:.4f}")
            print(f"  IoU: {class_metrics['iou']:.4f}")
            print(f"  样本数: {class_metrics['support']}")

        # 统计信息
        total_pixels = len(all_y_true)
        print(f"\n统计信息:")
        print(f"总像素数: {total_pixels}")
        print(f"类别0像素数: {np.sum(all_y_true == 0)}")
        print(f"类别1像素数: {np.sum(all_y_true == 1)}")

        # 如果提供了txt文件路径，将结果写入txt文件
        if txt_file_path:
            write_metrics_to_txt(metrics, result_name, txt_file_path)

        return metrics
    else:
        print("没有有效的评估结果")
        return None


def main():
    """主函数，执行完整的处理流程"""
    # 设置环境变量
    os.environ['PROJ_LIB'] = r'/home/cv/anaconda3/envs/test/share/proj'
    # base_folder = "/home/yons/ouyanggouzao/predict"  #开放并注释
    # items = os.listdir(base_folder) #开放并注释
    # 定义文件路径
    tif_files = ["mask0", "mask1", "mask2", "mask3", "mask4"]
    base_folder = "/home/yons/ouyanggouzao/predict"

    # 创建精度结果txt文件
    accuracy_txt_path = os.path.join(base_folder, "accuracy_results.txt")
    # 清空或创建文件
    with open(accuracy_txt_path, 'w', encoding='utf-8') as f:
        f.write("精度评估结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"评估时间: {np.datetime64('now')}\n")
        f.write(f"基础文件夹: {base_folder}\n")
        f.write("=" * 50 + "\n\n")

    print("步骤1: 处理预测掩膜...")
    process_prediction_masks(tif_files, base_folder)

    print("\n步骤2: 拼接TIFF文件...")
    # 遍历所有预测文件夹
    items = os.listdir(base_folder)
    for item in items:
        max_input_folder = os.path.join(base_folder, item, "output_max")
        avg_input_folder = os.path.join(base_folder, item, "output_average")

        if os.path.exists(max_input_folder):
            max_output_file = os.path.join(base_folder, item, "ymax_merged_geo.tiff")
            merge_tiff_georeference(max_input_folder, max_output_file)

        if os.path.exists(avg_input_folder):
            avg_output_file = os.path.join(base_folder, item, "yaverage_merged_geo.tiff")
            merge_tiff_georeference(avg_input_folder, avg_output_file)

    print("\n步骤3: 分类处理...")
    # 处理分类结果
    for item in items:
        max_tiff_path = os.path.join(base_folder, item, "ymax_merged_geo.tiff")
        average_tiff_path = os.path.join(base_folder, item, "yaverage_merged_geo.tiff")
        output_dir = os.path.join(base_folder, item, "result")

        if os.path.exists(max_tiff_path) and os.path.exists(average_tiff_path):
            process_classification_results(max_tiff_path, average_tiff_path, output_dir)

    print("\n步骤4: 评估结果并写入txt文件...")
    # 评估结果
    for item in items:
        if "test" in item:
            result_dir = os.path.join(base_folder, item, "result")
            final_result_path = os.path.join(result_dir, "final_result.tif")
            final_thinned_path = os.path.join(result_dir, "finalxfxt.tif")
            test_txt_path = "/home/yons/ouyanggouzao/newdata/bg200m_1_testtxt415.txt"  # 根据实际情况修改
            label_folder = "/home/yons/ouyanggouzao/bd1203/"  # 根据实际情况修改

            # 在txt文件中记录当前测试项
            with open(accuracy_txt_path, 'a', encoding='utf-8') as f:
                f.write(f"\n测试项: {item}\n")
                f.write("-" * 30 + "\n")

            # 评估原始结果
            if os.path.exists(final_result_path):
                print(f"\n评估原始结果: {item}")
                results_original = evaluate_segmentation(final_result_path, test_txt_path, label_folder,
                                                        f"{item} - 原始结果", accuracy_txt_path)

            # 评估细化结果
            if os.path.exists(final_thinned_path):
                print(f"\n评估细化结果: {item}")
                results_thinned = evaluate_segmentation(final_thinned_path, test_txt_path, label_folder,
                                                       f"{item} - 细化结果", accuracy_txt_path)

    print(f"\n所有精度结果已保存到: {accuracy_txt_path}")


if __name__ == "__main__":
    main()