import csv, math

def read_csv_file(file_path):
    # 讀取 CSV 檔案，返回資料
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if any(row.values()):
                data.append(row)
    return data

def read_labels(file_path):
    # 讀取標籤
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳過標題
        for row in reader:
            labels.append(row[0])
    return labels

def preprocess_data(data):
    # 預處理資料
    processed = []
    for i in range(len(data)):
        row = data[i]
        new_row = {}
        try:
            # 處理性別
            if row['gender'] == 'Male':
                new_row['gender'] = 1
            else:
                new_row['gender'] = 0

            # 處理 Yes/No 特徵
            yn_features = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
            for feature in yn_features:
                if row[feature] == 'Yes':
                    new_row[feature] = 1
                else:
                    new_row[feature] = 0

            # 處理 MultipleLines
            if row['MultipleLines'] == 'No phone service':
                new_row['MultipleLines'] = 0
            elif row['MultipleLines'] == 'No':
                new_row['MultipleLines'] = 1
            else:
                new_row['MultipleLines'] = 2

            # 處理 InternetService
            if row['InternetService'] == 'DSL':
                new_row['InternetService'] = 1
            elif row['InternetService'] == 'Fiber optic':
                new_row['InternetService'] = 2
            else:
                new_row['InternetService'] = 0

            # 處理依賴網路的特徵
            net_features = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
            for feature in net_features:
                if row[feature] == 'No internet service':
                    new_row[feature] = 0
                elif row[feature] == 'No':
                    new_row[feature] = 1
                else:
                    new_row[feature] = 2

            # 處理 Contract
            if row['Contract'] == 'Month-to-month':
                new_row['Contract'] = 0
            elif row['Contract'] == 'One year':
                new_row['Contract'] = 1
            else:
                new_row['Contract'] = 2

            # 處理 PaymentMethod
            pm = row['PaymentMethod']
            if pm == 'Electronic check':
                new_row['PaymentMethod'] = 0
            elif pm == 'Mailed check':
                new_row['PaymentMethod'] = 1
            elif pm == 'Bank transfer (automatic)':
                new_row['PaymentMethod'] = 2
            else:
                new_row['PaymentMethod'] = 3

            # 數值特徵
            new_row['tenure'] = int(row['tenure'])
            new_row['MonthlyCharges'] = float(row['MonthlyCharges'])
            if row['TotalCharges'].strip() == '':
                new_row['TotalCharges'] = 0.0
            else:
                new_row['TotalCharges'] = float(row['TotalCharges'])

            processed.append(new_row)
        except Exception as e:
            print(f"資料處理錯誤在第 {i} 行：{e}")
            continue
    return processed

def encode_labels(labels):
    # 編碼標籤
    encoded = []
    for l in labels:
        if l == 'Yes':
            encoded.append(1)
        else:
            encoded.append(0)
    return encoded

def standardize_features(features):
    # 標準化特徵
    means = {}
    stds = {}
    keys = features[0].keys()
    for key in keys:
        vals = [f[key] for f in features]
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum([(x - mean) ** 2 for x in vals]) / len(vals))
        means[key] = mean
        stds[key] = std

    standardized = []
    for f in features:
        s_f = {}
        for key in keys:
            if stds[key]:
                s_f[key] = (f[key] - means[key]) / stds[key]
            else:
                s_f[key] = 0
        standardized.append(s_f)
    return standardized, means, stds

def standardize_test_features(features, means, stds):
    # 標準化測試特徵
    standardized = []
    for f in features:
        s_f = {}
        for key in f:
            if stds[key]:
                s_f[key] = (f[key] - means[key]) / stds[key]
            else:
                s_f[key] = 0
        standardized.append(s_f)
    return standardized

def euclidean_distance(a, b):
    # 計算歐氏距離
    dist = 0
    for key in a:
        dist += (a[key] - b[key]) ** 2
    return math.sqrt(dist)

def get_k_nearest_neighbors(train_features, train_labels, test_instance, k):
    # 找到最近的 k 個鄰居
    distances = []
    for i in range(len(train_features)):
        dist = euclidean_distance(train_features[i], test_instance)
        distances.append((train_labels[i], dist))
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]
    return neighbors

def predict_classification(neighbors):
    # 預測分類
    class_votes = {}
    for label, distance in neighbors:
        if distance == 0:
            weight = 1e6
        else:
            weight = 1 / distance
        if label in class_votes:
            class_votes[label] += weight
        else:
            class_votes[label] = weight
    sorted_votes = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)
    return sorted_votes[0][0]

def knn_predict(train_features, train_labels, test_features, k):
    # 使用 KNN 進行預測
    predictions = []
    for i in range(len(test_features)):
        neighbors = get_k_nearest_neighbors(train_features, train_labels, test_features[i], k)
        result = predict_classification(neighbors)
        predictions.append(result)
    return predictions

def save_predictions(predictions, output_file):
    # 保存預測結果
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Churn'])
        for pred in predictions:
            if pred == 1:
                label = 'Yes'
            else:
                label = 'No'
            writer.writerow([label])

def calculate_accuracy(predicted_file, ground_truth_file, print_details=True):
    # 計算準確率
    with open(predicted_file, 'r', encoding='utf-8') as f_pred:
        pred_reader = csv.reader(f_pred)
        pred_values = [row[0] for row in pred_reader][1:]
    with open(ground_truth_file, 'r', encoding='utf-8') as f_gt:
        gt_reader = csv.reader(f_gt)
        gt_values = [row[0] for row in gt_reader][1:]

    correct = 0
    for p, g in zip(pred_values, gt_values):
        if p == g:
            correct += 1
    accuracy = correct / len(gt_values)
    if print_details:
        print(f"總共樣本數：{len(gt_values)}")
        print(f"正確預測數：{correct}")
        print(f"準確率：{accuracy}")
    else:
        print(accuracy)
    return accuracy

def main():
    # 主函數
    train_data = read_csv_file('train.csv')
    val_data = read_csv_file('val.csv')
    test_data = read_csv_file('test.csv')

    train_labels_raw = read_labels('train_gt.csv')
    val_labels_raw = read_labels('val_gt.csv')

    train_labels = encode_labels(train_labels_raw)
    val_labels = encode_labels(val_labels_raw)

    train_features = preprocess_data(train_data)
    val_features = preprocess_data(val_data)
    test_features = preprocess_data(test_data)

    train_features, means, stds = standardize_features(train_features)
    val_features = standardize_test_features(val_features, means, stds)
    test_features = standardize_test_features(test_features, means, stds)

    print(f"訓練資料量：{len(train_features)}")
    print(f"驗證資料量：{len(val_features)}")

    if len(train_features) != len(train_labels):
        print("資料和標籤數量不匹配！")
        return

    best_accuracy = 0
    best_k = 1
    # 使用 √N 決定 K 值的最大範圍
    max_k = int(math.sqrt(len(train_features))) +10  # 根據訓練資料的數量取平方根
    for k in range(1, max_k, 2):  # 測試奇數的 K 值
        print(f"正在測試 k = {k}")
        preds = knn_predict(train_features, train_labels, val_features, k)
        save_predictions(preds, 'val_pred.csv')
        acc = calculate_accuracy('val_pred.csv', 'val_gt.csv', print_details=False)
        print(f"k = {k}, 準確率 = {acc:.3f}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_k = k

    print(f"最佳的 k 值是 {best_k}，準確率是 {best_accuracy:.3f}")

    # 使用最佳 K 值進行測試
    val_preds = knn_predict(train_features, train_labels, val_features, best_k)
    test_preds = knn_predict(train_features, train_labels, test_features, best_k)

    save_predictions(val_preds, 'val_pred.csv')
    save_predictions(test_preds, 'test_pred.csv')

    print("最終驗證集結果：")
    calculate_accuracy('val_pred.csv', 'val_gt.csv')

if __name__ == '__main__':
    main()
