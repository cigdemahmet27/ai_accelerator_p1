# LG481E - AI Accelerators - Project 1: Unsupervised Learning with AutoEncoders

## Yapilan Degisiklikler ve Iyilestirmeler

Bu belge, projenin ilk versiyonunda tespit edilen eksikleri ve bunlarin nasil giderildigini detaylica aciklamaktadir.

---

## 1. BatchNorm Eklenmesi (`autoencoder.py`)

### Onceki Durum
Encoder ve decoder katmanlari sadece `Linear` + `ReLU` iceriyordu:
```
784 -> 256 (ReLU) -> 64 (ReLU) -> latent_dim
```

### Yapilan Degisiklik
Her gizli katmandan sonra `BatchNorm1d` katmani eklendi:
```
784 -> 256 (BatchNorm -> ReLU) -> 64 (BatchNorm -> ReLU) -> latent_dim
```
Ayni yapi decoder icin de uygulandI.

### Neden?
- **BatchNorm**, her mini-batch icindeki aktivasyonlari normalize eder.
- Bu, egitim sirasinda "internal covariate shift" problemini azaltir.
- Sonuc olarak egitim daha hizli yakinsir ve daha kararli olur.
- Genellikle daha iyi genelleme (generalization) saglar, bu da latent temsilin kalitesini arttirir.
- Ozellikle dusuk latent boyutlarda (ornegin 16) encoder'in onemli ozellikleri ogrenmesi kolaylasir.

### Ayrica: `encode()` Metodu Eklendi
Modele ayri bir `encode()` metodu eklendi. Onceden `evaluate.py` icinde `model.encoder` dogrudan erisilebiliyordu ama bu, modelin ic yapisina bagimlilik olusturuyordu. Yeni `encode()` metodu, flat edilme islemini de kendi icinde yaparak daha temiz bir arayuz sunar.

---

## 2. Test Loss Hesaplanmasi (`main.py`)

### Onceki Durum
`testloader` parametresi `train_autoencoder()` fonksiyonuna veriliyordu ama **icerde hic kullanilmiyordu**. Sadece training loss hesaplaniyordu. Bu durumda modelin overfit yapip yapmadigini anlamak imkansizdi.

### Yapilan Degisiklik
Ayri bir `compute_loss()` fonksiyonu eklendi. Her epoch sonunda hem train loss hem de test loss hesaplanip kaydediliyor. Loss curve grafiginde artik iki egri goruntuleniyor:
- **Duz cizgi**: Training loss
- **Kesikli cizgi**: Test loss

### Neden?
- **Overfitting tespiti**: Eger test loss artarken train loss dusuyorsa, model ezberliyor demektir.
- **Model secimi**: Farkli latent boyutlari karsilastirirken hem train hem test performansini gormek gerekir.
- Odev raporu icin loss curve istendiginde, sadece train loss ile sunulan bir grafik eksik kalir.

---

## 3. Coklu Latent Boyut Destegi (`main.py`)

### Onceki Durum
Sadece `latent_dim=16` ile tek bir model egitiliyordu.

### Yapilan Degisiklik
Proje artik 3 farkli latent boyut icin otomatik olarak calisir: **16, 32, 64**. Her latent boyut icin ayri bir klasor olusturulur (`results/latent_16/`, `results/latent_32/`, `results/latent_64/`) ve tum sonuclar karsilastirmali olarak yazdirilan bir tabloda toplanir.

### Neden?
Farkli latent boyutlari farkli bir tradeoff sunar:

| Latent Boyut | Beklenen Reconstruction | Beklenen Clustering |
|---|---|---|
| 16 | Daha dusuk (cok sikiistirma) | Kompakt, K-Means icin avantajli olabilir |
| 32 | Dengeli | Genellikle en iyi denge |
| 64 | Daha yuksek (az sikiistirma) | Yuksek boyut, "curse of dimensionality" riski |

Bu karsilastirma raporda tartismak icin zengin bir analiz malzemesi saglar.

### Karsilastirma Tablosu
Tum deneylerin sonuclari `results/comparison.txt` dosyasina kaydedilir. Tablo sunlari icerir:
- Final Train Loss
- Final Test Loss
- PMS (%)
- AD
- AVC
- TD

---

## 4. Reconstruction Gorsellerine Satir Basliklari (`evaluate.py`)

### Onceki Durum
Reconstruction gorseli iki satirdan olusuyordu ama hangi satirin "Original", hangisinin "Reconstructed" oldugu belli degildi.

### Yapilan Degisiklik
Birinci satira **"Original"**, ikinci satira **"Reconstructed"** etiketi eklendi. Ayrica her gorselin basligina latent boyut bilgisi eklendi.

### Neden?
Gorsel bir degerlendirme araci olarak kullanilacaksa, neyin ne oldugu acik olmali. Odev raporu icin bu etiketler zorunlu sayilir.

---

## 5. Cluster Gorsellestirmesi (`evaluate.py`)

### Onceki Durum
Latent uzay sadece **gercek etiketlere** (true labels) gore renklendiriliyordu. K-Means'in buldugu cluster atamalarina gore bir gorsellestirme yoktu.

### Yapilan Degisiklik
Yeni `plot_cluster_space()` fonksiyonu eklendi. Artik her latent boyut icin 4 latent uzay gorseli uretiliyor:
1. **PCA - True Labels**: Gercek rakamlara gore renklendirilmis
2. **PCA - K-Means Clusters**: Cluster atamalarina gore renklendirilmis
3. **t-SNE - True Labels**: Gercek rakamlara gore renklendirilmis
4. **t-SNE - K-Means Clusters**: Cluster atamalarina gore renklendirilmis

### Neden?
Bu 4 gorseli yan yana koyarak sunlari analiz edebilirsiniz:
- Clustering gercek gruplari ne kadar yakalayabiliyor?
- Hangi rakamlar birbiriyle karistiriliyor? (ornegin 4 ve 9, 3 ve 8)
- Latent uzay yapisinin boyuta gore nasil degistigi

---

## 6. Confusion Matrix Etiketleri (`evaluate.py`)

### Onceki Durum
`sns.heatmap()` cagirisinda x ve y eksenleri icin acik etiketler yoktu. Hangi satirin/sutunun hangi rakama karsilik geldigi belli degildi.

### Yapilan Degisiklik
`xticklabels=range(10)` ve `yticklabels=range(10)` parametreleri eklendi. Artik her satir ve sutun 0-9 arasindaki rakamlari acikca gosteriyor.

---

## 7. Metriklerin Dosyaya Kaydedilmesi (`evaluate.py`)

### Onceki Durum
PMS, AD, AVC, TD metrikleri sadece console'a yazdiriliiyordu. Rapor yazarken bu degerleri tekrar bulmak icin kodu yeniden calistirmak gerekiyordu.

### Yapilan Degisiklik
Her latent boyut icin metrikler `results/latent_X/metrics.txt` dosyasina kaydediliyor. Ayrica tum latent boyutlarin karsilastirmasi `results/comparison.txt` dosyasina yaziliyor.

---

## 8. t-SNE Parametre Iyilestirmesi (`evaluate.py`)

### Onceki Durum
t-SNE varsayilan parametrelerle calistiriliyordu (`perplexity=30`, `n_iter=250`).

### Yapilan Degisiklik
- `perplexity=40`: MNIST boyutundaki veri setleri icin daha uygun. Perplexity, her noktanin etrafindaki komsu sayisini belirler; 40-50 arasi degerlerin 10K+ veri noktasinda daha anlamli yapilar ortaya cikardigi biliniyor.
- `n_iter=1000`: Varsayilan iterasyon sayisi dusuk olabilir. 1000 iterasyon, t-SNE'nin kararli bir yerlesime ulasmasi icin yeterli suredir.

### Neden?
Daha iyi parametreler = daha anlamli 2D goruntuleme = raporda daha acik cluster yapilari.

---

## 9. Epoch Sayisinin Artirilmasi (`main.py`)

### Onceki Durum
20 epoch.

### Yapilan Degisiklik
50 epoch.

### Neden?
20 epoch, ozellikle dusuk latent boyutlarda (16) model icin yeterli olmayabilir. 50 epoch ile modelin tam yakinsadigini gormek mumkun. Loss curve'de plato goruldugunde modelin ogrenmeyi tamamladigini anlayabiliriz. BatchNorm eklenmesiyle birlikte 50 epoch buyuk ihtimalle gereginden fazla olacak ama bu, loss curve'de net bir plato gormemizi saglar - ki bu raporda gostermek icin idealdir.

---

## 10. Dosya Adi Duzeltmesi

`requiriments.txt` -> `requirements.txt` olarak duzeltildi.

---

## Sonuc Dosya Yapisi

Proje calistirildiktan sonra olusacak dosya yapisi:

```
ai_accelerator_p1/
├── src/
│   ├── main.py              # Ana calistirma dosyasi
│   ├── autoencoder.py        # Model tanimI
│   └── evaluate.py           # Degerlendirme fonksiyonlari
├── requirements.txt
├── rapor.md
└── results/
    ├── comparison.txt        # Tum latent boyutlarin metrik karsilastirmasi
    ├── latent_16/
    │   ├── autoencoder_model.pth
    │   ├── loss_curve.png
    │   ├── reconstructions.png
    │   ├── latent_space_pca_true.png
    │   ├── latent_space_pca_clusters.png
    │   ├── latent_space_t-sne_true.png
    │   ├── latent_space_t-sne_clusters.png
    │   ├── confusion_matrix.png
    │   └── metrics.txt
    ├── latent_32/
    │   └── ... (ayni dosyalar)
    └── latent_64/
        └── ... (ayni dosyalar)
```

## Nasil Calistirilir

```bash
cd ai_accelerator_p1/src
pip install -r ../requirements.txt
python main.py
```

GPU mevcut ise otomatik olarak kullanilir (CUDA). Yoksa CPU uzerinde calisir.
