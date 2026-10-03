import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
from sklearn.metrics import accuracy_score, classification_report

### APPROCHE 1 & 2
# Programme permettant de remplir les listes im_add et labels
def init_lists_0():
    im_add = []  # va contenir les adresses des images
    labels = []  # va contenir les listes [n° cardinal, xmin, ymin, xmax, ymax]
    dict_card_en = {0: "E", 1: "N", 2: "S", 3: "W"}
    for i in range(4):
        for j in range(1, 100):
            # On assigne le numéro avec les 0 qui complètent à gauche
            num = "0" * (3 - len(str(j))) + str(j)
            base = f"../datasets/dataset1/images/{dict_card_en[i]}_{num}"
            add = None
            img = None
            for ext in [".jpg", ".png"]:
                candidate = base + ext
                tmp = cv2.imread(candidate)
                if tmp is not None:
                    add = candidate
                    img = tmp
                    break
            # Au cas où on ne peut pas lire le fichier (j > jmax) selon le cardinal
            if img is None:
                break
            h, w = img.shape[:2]
            # On ajoute à la liste
            im_add.append(add)
            # On cherche le fichier texte et on convertit les données pour les ajouter dans labels
            with open(f"../datasets/dataset1/labels/{dict_card_en[i]}_{num}.txt", "r", encoding="utf-8") as f:
                label = f.read().strip()
            gt, bb_xc, bb_yc, bb_w, bb_h = map(float, label.split())
            gt = int(gt)
            bb_xc, bb_yc, bb_w, bb_h = int(bb_xc * w), int(bb_yc * h), int(bb_w * w), int(bb_h * h)
            xmin, ymin = bb_xc - bb_w // 2, bb_yc - bb_h // 2
            xmax, ymax = bb_xc + bb_w // 2, bb_yc + bb_h // 2
            labels.append([gt, xmin, ymin, xmax, ymax])
    return im_add, labels

### APPROCHE 3
# Programme permettant de remplir les listes im_add et labels
def init_lists_1(train=True):
    DATASET_DIR = ("train" if train else "val")
    im_add = []  # va contenir les adresses des images
    labels = []  # va contenir les listes [n° cardinal, xmin, ymin, xmax, ymax]
    dict_card_en = {0: "E", 1: "N", 2: "S", 3: "W"}
    for i in range(4):
        for j in range(1, 100):
            # On assigne le numéro avec les 0 qui complètent à gauche
            num = "0" * (3 - len(str(j))) + str(j)
            base = f"../datasets/buoys0/images/{DATASET_DIR}/{dict_card_en[i]}_{num}"
            add = None
            img = None
            for ext in [".jpg", ".png"]:
                candidate = base + ext
                try:
                    tmp = cv2.imread(candidate)
                except FileNotFoundError:
                    continue
                if tmp is not None:
                    add = candidate
                    img = tmp
                    break
            # Au cas où on ne peut pas lire le fichier (j > jmax) selon le cardinal
            if img is not None:
                h, w = img.shape[:2]
                # On ajoute à la liste
                im_add.append(add)
                # On cherche le fichier texte et on convertit les données pour les ajouter dans labels
                with open(f"../datasets/buoys0/labels/{DATASET_DIR}/{dict_card_en[i]}_{num}.txt", "r", encoding="utf-8") as f:
                    label = f.read().strip()
                gt, bb_xc, bb_yc, bb_w, bb_h = map(float, label.split())
                gt = int(gt)
                bb_xc, bb_yc, bb_w, bb_h = int(bb_xc * w), int(bb_yc * h), int(bb_w * w), int(bb_h * h)
                xmin, ymin = bb_xc - bb_w // 2, bb_yc - bb_h // 2
                xmax, ymax = bb_xc + bb_w // 2, bb_yc + bb_h // 2
                labels.append([gt, xmin, ymin, xmax, ymax])
    return im_add, labels

### APPROCHE 1 & 2
# Calcul de l'intersection et de l'union des boîtes englobantes
def calculate_iou(boxA, boxB):
    xA, yA, xB, yB = max(boxA[0], boxB[0]), max(boxA[1], boxB[1]), min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    areaA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    areaB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(areaA + areaB - interArea) if (areaA + areaB - interArea) > 0 else 0

#### APPROCHE 1
# Détection robuste de la bouée dans une image donnée
def detect_buoy_robust(image_path, visualize=False, gt_box=None):
    img = cv2.imread(image_path)
    if img is None: return [0, 0, 0, 0]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Masques pour isoler les couleurs de la bouée
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask_y = cv2.inRange(hsv, lower_yellow, upper_yellow)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60])
    mask_b = cv2.inRange(hsv, lower_black, upper_black)

    # Combinaison et nettoyage morphologique
    combined = cv2.bitwise_or(mask_y, mask_b)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (w//140, h//130))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (w//9, h//5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)

    # Recherche des contours externes
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        pred_box = [0, 0, 0, 0]
    else:
        valid_boxes = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = bh / float(bw)
            if 1.5 < aspect_ratio < 5:
                valid_boxes.append((x, y, x + bw, y + bh, cv2.contourArea(cnt)))

        if valid_boxes:
            # Sélection de la meilleure boîte basée sur l'aire maximale
            best_box = max(valid_boxes, key=lambda x: x[4])
            pred_box = list(best_box[:4])
        else:
            # Solution de repli sur le plus grand contour
            c = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(c)
            pred_box = [x, y, x + bw, y + bh]

    if visualize:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1); plt.imshow(img_rgb); plt.title("Image Originale")
        plt.subplot(1, 3, 2); plt.imshow(combined, cmap='gray'); plt.title("Masque N&B des deux composantes nettoyé")
        res_img = img_rgb.copy()
        cv2.rectangle(res_img, (pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), (0, 255, 0), 5)
        if gt_box:
            cv2.rectangle(res_img, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (255, 0, 0), 5)
        plt.subplot(1, 3, 3); plt.imshow(res_img); plt.title("Prédiction : Vert | Vérité terrain : Rouge")
        plt.show()

    return pred_box

### APPROCHE 1
# Extraction des centres des contours détectés dans le masque binaire
def extraire_centres_par_couleur(mask, couleur, min_area=10):
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centres = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            centres.append((y + h / 2, couleur, area))
    return centres

### APPROCHE 1 & 2
# Fonctions pour calculer les métriques d'évaluation à partir du dictionnaire de statistiques
def calculer_metriques_classif(dico_stats):
    """
    Calcule l'Accuracy, la Précision, le Rappel et le F1-score 
    à partir d'un dictionnaire de type {'Vérité_Prédiction': occurrences}.
    """
    # 1. Extraire automatiquement les noms des classes uniques
    classes = set()
    for cle in dico_stats.keys():
        verite, prediction = cle.split('_')
        classes.add(verite)
        classes.add(prediction)
    
    # 2. Initialisation des variables globales
    resultats = {'Global': {}, 'Par_Classe': {}}
    total_echantillons = sum(dico_stats.values())
    
    # Trouver le nombre total de bonnes prédictions (diagonale)
    total_corrects = sum(dico_stats.get(f"{c}_{c}", 0) for c in classes)
    
    # Calcul de l'Accuracy (Exactitude globale)
    accuracy = total_corrects / total_echantillons if total_echantillons > 0 else 0
    resultats['Global']['Accuracy'] = accuracy
    
    # 3. Calcul des métriques pour chaque classe (One-vs-Rest)
    for c in classes:
        # Vrais Positifs : Vérité == c ET Prédiction == c
        tp = dico_stats.get(f"{c}_{c}", 0)
        
        # Faux Positifs : Prédiction == c MAIS Vérité != c
        fp = sum(valeur for cle, valeur in dico_stats.items() 
                 if cle.endswith(f"_{c}") and not cle.startswith(f"{c}_"))
        
        # Faux Négatifs : Vérité == c MAIS Prédiction != c
        fn = sum(valeur for cle, valeur in dico_stats.items() 
                 if cle.startswith(f"{c}_") and not cle.endswith(f"_{c}"))
        
        # Calculs avec protection contre la division par zéro
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rappel = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * rappel) / (precision + rappel) if (precision + rappel) > 0 else 0.0

        # Accuracy par classe (en considérant les vrais négatifs)
        tn = total_echantillons - tp - fp - fn
        accuracy_classe = (tp + tn) / total_echantillons if total_echantillons > 0 else 0.0
        
        # Stockage
        resultats['Par_Classe'][c] = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'Precision': precision,
            'Rappel': rappel,
            'F1_score': f1,
            'AccuracyClasse': accuracy_classe
        }
    return resultats

### APPROCHE 1 & 2
# Fonction pour afficher les métriques de manière formatée
def afficher_metriques_classif(resultats):
    """
    Affiche les métriques de manière formatée.
    """
    print(f"--- MÉTROLOGIE GLOBALE ---")
    print(f"Accuracy globale: {resultats['Global']['Accuracy']:.2%}\n")

    print(f"--- MÉTROLOGIE PAR CLASSE ---")
    for classe, stats in resultats['Par_Classe'].items():
        # On ignore la classe 'Inconnu' si elle est vide pour alléger l'affichage
        if stats['TP'] == 0 and stats['FP'] == 0 and stats['FN'] == 0:
            continue
            
        print(f"Classe {classe} :")
        print(f"  Précision : {stats['Precision']:.2%}")
        print(f"  Rappel    : {stats['Rappel']:.2%}")
        print(f"  F1-score  : {stats['F1_score']:.2%}")
        print(f"  Accuracy  : {stats['AccuracyClasse']:.2%}")
        print(f"  (Détails  : TP={stats['TP']}, FP={stats['FP']}, FN={stats['FN']})\n")

### APPROCHE 2
# On peut aussi faire une fonction pour fusionner les boîtes englobantes qui se chevauchent en une seule grande boîte qui englobe l'ensemble, au cas où il y aurait plusieurs détections proches les unes des autres
def fusionner_boites_englobantes(boites):
    """
    Regroupe toutes les boîtes qui se chevauchent en une seule grande boîte
    qui englobe l'ensemble.
    Format attendu : listes de [xmin, ymin, xmax, ymax]
    """
    if not len(boites): 
        return []
    
    # On copie la liste pour ne pas modifier l'originale
    boites_restantes = [list(b) for b in boites]
    boites_finales = []
    
    # Fonction interne pour vérifier le chevauchement simple
    def se_chevauchent(b1, b2):
        if b1[2] < b2[0] or b2[2] < b1[0]: return False # Ne se touchent pas en X
        if b1[3] < b2[1] or b2[3] < b1[1]: return False # Ne se touchent pas en Y
        return True

    while boites_restantes:
        # On prend la première boîte comme base
        boite_courante = boites_restantes.pop(0)
        fusion_en_cours = True
        
        while fusion_en_cours:
            fusion_en_cours = False
            i = 0
            while i < len(boites_restantes):
                # Si la boîte courante chevauche une autre boîte de la liste
                if se_chevauchent(boite_courante, boites_restantes[i]):
                    b_test = boites_restantes.pop(i)
                    
                    # On met à jour la boîte courante pour qu'elle englobe les deux
                    boite_courante = [
                        min(boite_courante[0], b_test[0]), # xmin
                        min(boite_courante[1], b_test[1]), # ymin
                        max(boite_courante[2], b_test[2]), # xmax
                        max(boite_courante[3], b_test[3])  # ymax
                    ]
                    # Comme la boîte a grandi, on doit revérifier depuis le début
                    fusion_en_cours = True 
                else:
                    i += 1
                    
        boites_finales.append(boite_courante)
        
    return boites_finales

### APPROCHE 2
# Fonction qui va garder la plus grande boîte parmi un groupe de boîtes 
def garder_plus_grande_boite(boites):
    """
    Parmi un groupe de boîtes, ne garder que la plus grande (en surface).
    Format attendu : listes de [xmin, ymin, xmax, ymax]
    """
    if not len(boites): 
        return []
    
    # On calcule la surface de chaque boîte
    surfaces = [(b[2] - b[0]) * (b[3] - b[1]) for b in boites]
    
    # On trouve l'indice de la boîte avec la plus grande surface
    indice_max = np.argmax(surfaces)
    
    # On retourne uniquement cette boîte
    return boites[indice_max]

### APPROCHE 2
# Post processing : on enlève les pixels dont le contour est plus petit que le seuil
def clean_noise(mask, threshold=100):
    # Trouver tous les amas (contours)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Effacer les amas trop petits
    for cnt in contours:
        if cv2.contourArea(cnt) < threshold:
            cv2.drawContours(mask, [cnt], -1, 0, -1) # Remplit de noir
    return mask

### APPROCHE 2
# Algorithme de clustering pour extraire les centres des amas
def get_clusters(image_bw, h_total, w_total, tolerance_ratio=0.2):
    """Fonction utilitaire pour extraire les centres validés d'un masque"""
    y_coords, x_coords = np.where(image_bw == 1)
    if len(y_coords) == 0: return np.array([])

    points = np.column_stack((y_coords, x_coords))
    init_centers = np.array([[np.min(y_coords), w_total // 2], [np.max(y_coords), w_total // 2]])

    kmeans = KMeans(n_clusters=2, init=init_centers, n_init=1)
    kmeans.fit(points)
    centers = kmeans.cluster_centers_

    # Fusion si trop proches verticalement
    diff_y = abs(centers[0, 0] - centers[1, 0])
    if diff_y < (tolerance_ratio * h_total):
        return np.array([np.mean(centers, axis=0)])
    return centers

### APPROCHE 3
# Entraînement du modèle ResNet avec un scheduler pour ajuster le learning rate
def train_resnet(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):    
    # Ajout d'un scheduler pour réduire le LR si la perte de validation stagne
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.float() / len(train_loader.dataset)

        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_running_corrects.float() / len(val_loader.dataset)
        
        # Mise à jour du scheduler basée sur la perte de validation
        scheduler.step(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')

### APPROCHE 3
# Fonction pour prédire et afficher une image aléatoire du dataset avec le modèle entraîné
def predict_and_display_image(model, dataset, classes, device):
    model.eval() # Set model to evaluation mode

    # Choose a random image from the dataset
    idx = np.random.randint(0, len(dataset))
    image, true_label_idx = dataset[idx]
    true_label_name = classes[true_label_idx]

    # Prepare image for model input (add batch dimension)
    input_image = image.unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(input_image)
        _, predicted_label_idx = torch.max(output, 1)
        predicted_label_name = classes[predicted_label_idx.item()]

    # Convert image back to numpy for displaying (denormalize if needed)
    # The transformation includes normalization, so we need to reverse it for proper display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_for_display = image.numpy().transpose((1, 2, 0)) # C, H, W to H, W, C
    img_for_display = std * img_for_display + mean
    img_for_display = np.clip(img_for_display, 0, 1)

    # For simplicity, we'll just display the image as is; it might look a bit off due to normalization
    # If the user asks for proper display, we can add denormalization.
    img_for_display = image.permute(1, 2, 0).cpu().numpy()
    # Normalize to 0-1 range for display if not already (or clamp to prevent issues)
    img_for_display = (img_for_display - img_for_display.min()) / (img_for_display.max() - img_for_display.min())

    # Display the image and prediction
    plt.imshow(img_for_display)
    plt.title(f"True Label: {true_label_name}\nPredicted Label: {predicted_label_name}")
    plt.axis('off')
    plt.show()

### APPROCHE 3
# Fonction pour évaluer le modèle sur l'ensemble de test et afficher les métriques
def evaluate_full_test_set(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calcul de l'accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy globale : {accuracy:.2%}\n")

    # Rapport détaillé par classe
    target_names = test_loader.dataset.classes
    print("Rapport de classification :")
    print(classification_report(all_labels, all_preds, target_names=target_names))

    return all_labels, all_preds