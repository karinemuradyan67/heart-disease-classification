# Սրտի Հիվանդության Դասակարգում (Real ML Classification Project)

Այս նախագիծը լիարժեք աշխատող մեքենայական ուսուցման (ML) համակարգ է, որը իրական տվյալներով սովորում է և կանխատեսում է սրտի հիվանդության առկայությունը (binary classification)։

## 1. Նախագծի նպատակը

Տրված են հիվանդի կլինիկական ցուցանիշներ (տարիք, ճնշում, խոլեստերին, կրծքավանդակի ցավի տեսակ և այլն)։
Մեր նպատակը է կանխատեսել թիրախային դասը.

- 0 -> առողջ (հիվանդություն չի հայտնաբերվում)
- 1 -> հիվանդություն (target > 0)

Սա դասական supervised classification խնդիր է, ոչ թե regression, քանի որ ելքը դաս է, ոչ թե անընդհատ թվային արժեք։

## 2. Օգտագործված տվյալները

- Աղբյուր: UCI Cleveland Heart Disease Dataset
- Ֆայլ: data\heart_disease.csv
- Մաքրման և վերափոխման արդյունքում ուսուցման համար օգտագործվել է 534 նմուշ

Նշում. train.py-ի մեջ տվյալների ուղին հիմա աշխատում է հարաբերական ձևով (default = data\heart_disease.csv), և history/log-ում նույնպես պահվում է հարաբերական ուղի։

## 3. Ինչ է իրականում արված (ոչ միայն տեսական մաս)

Նախագծում իրականացված է ամբողջ ML pipeline-ը.

1. Տվյալների բեռնում CSV-ից
2. Մաքրման փուլ.
	- target-ի սխալ արժեքներով տողերի հեռացում
	- բացակայող արժեքների մշակում (`ca`, `thal`)
3. Թիրախի binarization (`target > 0` -> 1)
4. Feature engineering.
	- age_group
	- high_bp
	- high_chol
	- stress_index
5. One-hot encoding categorical ֆիչերների համար
6. Train/test բաժանում (`stratify=y`, 80/20)
7. StandardScaler նորմավորում
8. Բազմաթիվ մոդելների ուսուցում և համեմատություն.
	- Logistic Regression
	- KNN (k=13)
	- Random Forest
	- Gradient Boosting
9. Գնահատում մի քանի չափանիշներով.
	- Accuracy
	- F1
	- ROC-AUC
	- 5-fold cross-validated F1
10. Լավագույն մոդելի ընտրություն CV F1-ով
11. Պահպանում սկավառակին.
	- models/best_model.joblib
	- models/scaler.joblib
12. Ուսուցման պատմության պահպանում.
	- models/training_history.json
13. Ինտերակտիվ prediction ռեժիմ CLI-ով (`--predict`)

Սա նշանակում է՝ նախագիծը վերջից վերջ աշխատող ML համակարգ է (training + evaluation + persistence + inference)։

## 4. Փաստացի արդյունքները (վերջին գործարկումից)

train.py-ի վերջին գործարկման արդյունքները.

- Լավագույն մոդել: Gradient Boosting
- Accuracy: 0.9626
- F1: 0.9574
- ROC-AUC: 0.9743
- CV F1: 0.8941 +- 0.0328

Classification report.

- Healthy: precision 0.94, recall 1.00, f1 0.97
- Disease: precision 1.00, recall 0.92, f1 0.96

Top feature importance (Gradient Boosting).

- thal
- cp
- ca
- stress_index
- oldpeak

Այս թվերը ցույց են տալիս, որ համակարգը իրականում սովորում է տվյալներից և ունի բարձր որակ դասակարգման խնդրում։

## 5. Ինչ լրացուցիչ աշխատանք է արվել


- Մի քանի մոդելի մրցակցային համեմատություն
- Cross-validation՝ կայունության գնահատման համար
- Feature engineering՝ բժշկական իմաստ ունեցող նոր ֆիչերներով
- History tracking՝ յուրաքանչյուր ուսուցման արդյունքների արխիվացում
- Interactive inference ռեժիմ՝ օգտագործելիության համար
- Notebook + Script երկակի ճարտարապետություն.
  - Notebook՝ ուսումնասիրություն/վիզուալիզացիա/բացատրություն
  - Script՝ production-style վերարտադրելի ուսուցում

## 6. Քայլ առ քայլ գործարկում (Windows)

1. Բացել terminal-ը նախագծի թղթապանակում
2. Ստեղծել վիրտուալ միջավայր.

```powershell
py -m venv .venv
```

3. Տեղադրել գրադարանները.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

4. Սովորեցնել մոդելները (relative data path-ով).

```powershell
.\.venv\Scripts\python.exe train.py --data data\heart_disease.csv
```

5. Դիտել training history.

```powershell
.\.venv\Scripts\python.exe train.py --history
```

6. Փորձել ինտերակտիվ կանխատեսում.

```powershell
.\.venv\Scripts\python.exe train.py --predict
```

7. (Ընտրովի) Գործարկել notebook-ը ամբողջությամբ.

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace heart_disease_classification.ipynb
```

## 7. Ինչպես ենք համոզվել, որ նախագիծը լիարժեք աշխատում է

Ապահովվել է հետևյալ ստուգումներով.

1. End-to-end գործարկում train.py-ով հաջող ավարտվել է
2. Բոլոր չափանիշները հաշվարկվում են առանց runtime error
3. Լավագույն մոդելն ավտոմատ ընտրվում է և պահպանվում է disk-ին
4. Scaler-ը նույնպես պահպանվում է և օգտագործվում է prediction-ի ժամանակ
5. prediction ռեժիմը աշխատում է օգտատիրոջ մուտքագրման հիման վրա
6. history ֆայլը թարմացվում է յուրաքանչյուր run-ից հետո
7. տվյալների ուղին աշխատում է հարաբերական ձևաչափով (data\heart_disease.csv)

## 8. Նախագծի կառուցվածք

```text
ML project/
|-- data/
|   `-- heart_disease.csv
|-- models/
|   |-- best_model.joblib
|   |-- scaler.joblib
|   `-- training_history.json
|-- heart_disease_classification.ipynb
|-- train.py
|-- requirements.txt
`-- README.md
```

## 9. Եզրակացություն

Այս նախագիծը իրական, վերարտադրելի և տեխնիկապես ամբողջական ML classification համակարգ է։
Այն ներառում է տվյալների նախապատրաստում, մոդելների ուսուցում, համեմատական գնահատում, լավագույն մոդելի ընտրություն, model persistence, inference և փորձարկելի արդյունքների պատմություն։

Այդ պատճառով այն կարելի է պաշտպանել ոչ թե որպես «սլայդային ներկայացում», այլ որպես ամբողջական աշխատող ինժեներական լուծում։

## 10. Ուղեցույցի պահանջների համապատասխանության պարզ checklist


1. Խնդրի տեսակի որոշում (Classification) -> արված է notebook-ում և README-ի սկզբում բացատրված է
2. Պարտադիր մոդելներ (Logistic Regression, KNN) -> արված է notebook-ում
3. Տողերի/սյունակների քանակ, տիպեր, describe -> արված է notebook-ում
4. Կորելացիա, missing values, preprocessing -> արված է notebook-ում
5. Categorical encoding, normalization, feature engineering -> արված է notebook-ում
6. Logistic Regression.
	- coefficients/intercept -> արված է notebook-ում
	- Accuracy/Precision/Recall -> արված է notebook-ում
	- confusion matrix -> արված է notebook-ում
7. KNN.
	- Accuracy/Precision/Recall -> արված է notebook-ում
	- confusion matrix -> արված է notebook-ում
8. README-ում ներկայացված է՝ ինչ ենք արել, ինչ ենք փորձել, վերջնական արդյունք, տվյալների հետևություններ, և մոդելների համեմատություն 
9. Հավելյալ աշխատանք (Cross-validation, ROC-AUC, multi-model benchmark, history tracking, interactive prediction) 


## 11. Ինչու որոշ մոդելներ ավելի թույլ կամ ավելի լավ արդյունք տվեցին (պարզ բառերով)

1. Logistic Regression լավ աշխատեց, որովհետև.
	- տվյալներում կան բավական պարզ գծային սահմանով բաժանվող օրինակներ
	- scaled տվյալների վրա LR-ը կայուն է և քիչ է overfit անում

2. KNN այս dataset-ում ավելի թույլ արդյունք տվեց, որովհետև.
	- տվյալներում կան կրկնվող/մոտիկ կետեր, որոնք տեղային մակարդակում «շփոթում» են հարևանների մեթոդը
	- KNN-ը շատ զգայուն է տեղային աղմուկին, հատկապես երբ սահմանի մոտ դասերը խառնված են

3. Gradient Boosting և Random Forest ավելի լավ աշխատեցին, որովհետև.
	- բռնում են ոչ գծային կապերը
	- լավ են աշխատում feature interaction-ների դեպքում
	- այս տվյալների վրա ունեն ավելի բարձր ընդհանրացման որակ



Վերջում ընտրվել է լավագույն մոդելը՝ Gradient Boosting, որը տվել է բարձր արդյունք (Accuracy 0.9626, F1 0.9574, ROC-AUC 0.9743)։
Նախագիծը լիարժեք է, որովհետև ունի training, evaluation, model saving, history tracking և interactive prediction ռեժիմ։
