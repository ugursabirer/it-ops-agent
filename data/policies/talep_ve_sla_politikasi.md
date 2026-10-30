# BT Talep Yönetimi ve SLA Politikası

## Talep Kategorileri
BT talepleri dört kategoriye ayrılır:

- **ariza / incident:** Çalışmayan bir servis veya cihaz.
- **talep / request:** Yeni donanım, yazılım veya erişim talebi.
- **degisiklik / change:** Mevcut bir sistemde yapılandırma değişikliği.
- **bilgi / info:** Bilgi talebi, nasıl yapılır soruları.

## Öncelik Seviyeleri ve Çözüm Süreleri (SLA)
| Öncelik | Tanım | İlk yanıt | Çözüm |
|---|---|---|---|
| kritik | Üretim durdu, birden fazla kullanıcı etkilendi | 15 dakika | 4 saat |
| yuksek | Tek kullanıcı çalışamıyor, iş kaybı var | 1 saat | 8 saat |
| normal | İş devam ediyor, geçici çözüm mevcut | 4 saat | 3 iş günü |
| dusuk | Bilgi talebi veya iyileştirme | 1 iş günü | 10 iş günü |

Dizüstü bilgisayarın tamamen açılmaması **yuksek** öncelikli kabul edilir.
Yeni donanım talepleri **normal** öncelikle açılır.

## Atama Grupları
- Donanım konuları: **BT-Donanim**
- Ağ, VPN ve erişim konuları: **BT-Altyapi**
- Yazılım, lisans ve uygulama konuları: **BT-Uygulama**
- Bilgi güvenliği konuları: **Bilgi-Guvenligi**

## Talep Açma Kuralları
Aynı çalışan için aynı kategoride **açık bir talep varsa** ikinci bir talep açılmaz;
mevcut talep güncellenir. Talep açılırken çalışan kimliği, kategori, konu ve
öncelik zorunludur. Eksik bilgiyle talep açılamaz.

## Eskalasyon
SLA çözüm süresinin %80'i aşıldığında talep otomatik olarak birim yöneticisine
eskale edilir. Kritik talepler eskalasyon olmadan doğrudan nöbetçi ekibe düşer.
