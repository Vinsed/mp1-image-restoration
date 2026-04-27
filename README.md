# Mini Project 1: Image Restoration
### Mata Kuliah: Pengolahan Citra dan Video

**Nama &nbsp;:** Vinsen Dwi Putra  
**NRP &nbsp;&nbsp;&nbsp;:** 5024241094

---

## Daftar Isi
1. [Overview](#1-overview)
2. [Dependencies](#2-dependencies)
3. [Cara Build dan Run Project](#3-cara-build-dan-run-project)
4. [Restoration Pipeline](#4-restoration-pipeline)
5. [Hasil Dan Analisis Teknik](#5-hasil-dan-analisis-teknik)

---

## 1. Overview
Proyek ini bertujuan merestorasi citra (seperti foto Lena) yang terdegradasi secara kompleks oleh kombinasi salt-and-pepper noise, Gaussian noise, kontras rendah, dan kekaburan (blur). Berbeda dengan sekadar memanggil fungsi bawaan secara instan, proyek ini mendemonstrasikan implementasi low-level dari algoritma inti pengolahan citra. Konvolusi 2D, pemetaan histogram (CLAHE) dengan interpolasi bilinear, hingga Unsharp Masking dibangun secara logis menggunakan array dasar untuk memastikan kontrol penuh terhadap modifikasi piksel.

## 2. Dependencies
- Python 3.x(newer)
- Numpy
- OpenCV2
- Matplotlib

## 3. Cara Build dan Run Project
Untuk menjalankan proyek ini, tahap pertama adalah memastikan seluruh pustaka yang dibutuhkan telah terinstal dengan menjalankan perintah `pip install opencv-python numpy matplotlib` pada terminal. Selanjutnya, pastikan struktur direktori kerja Anda sudah benar, di mana skrip `restoration.py` berada di akar direktori, didampingi oleh folder `input/` yang berisi citra rusak seperti `lena_noisy.png`, serta sebuah folder `output/` untuk menampung hasil pemrosesan. Setelah persiapan selesai, eksekusi *pipeline* utama dengan menjalankan perintah `python restoration.py`. Sistem akan memproses citra melalui seluruh tahapan secara berurutan dan menyimpan citra hasil restorasi beserta panel komparatif visualnya ke dalam direktori keluaran.

## 4. Restoration Pipeline
Alur kerja restorasi pada proyek ini dirancang berjalan lurus dan sekuensial melalui tiga tahapan utama. Tahap pertama difokuskan pada *denoising* gabungan, di mana citra disapu menggunakan Median Filter untuk menghilangkan noise impulsif, lalu diperhalus menggunakan Gaussian Filter. Citra yang telah bersih dari noise kemudian masuk ke tahap kedua, yaitu *Contrast Limited Adaptive Histogram Equalization* (CLAHE). Pada tahap ini, kontras citra dinaikkan secara lokal khusus pada ruang warna Luminance (Y) agar akurasi warna asli tetap terjaga. Sebagai penutup, tahap ketiga menerapkan *Unsharp Masking* untuk memulihkan detail batas objek dan ketajaman frekuensi tinggi yang sempat tereduksi akibat proses penghalusan di tahap pertama.

---

## 5. Hasil Dan Analisis Teknik

### A. Tahap Denoising
Citra awal mengalami degradasi parah akibat dua jenis noise yang saling tumpang tindih: *salt-and-pepper noise* yang merusak struktur spasial dengan piksel ekstrem bernilai 0 atau 255, serta *Gaussian noise* yang menyebar sebagai butiran halus di seluruh permukaan. Untuk mengatasi masalah ganda ini, diterapkan pendekatan sekuensial yang dimulai dengan Median Filter berukuran kernel agresif 7x7, dilanjutkan dengan Gaussian Filter berukuran 5x5 dengan nilai sigma 1.2. Pemilihan metode ini dilandasi oleh fakta matematis bahwa filter linear seperti rata-rata (*mean*) akan gagal total dan justru menyebarkan nilai *outlier* ekstrem menjadi noda buram. Oleh karena itu, sifat non-linear Median Filter menjadi keharusan untuk mengeliminasi nilai impulsif. Setelah noise ekstrem ini hancur, barulah Gaussian filter dapat bekerja meratakan sisa fluktuasi acak. Walaupun kernel 7x7 terbukti ampuh menyapu bersih noise padat, pendekatan ini membawa cacat inheren karena sifatnya yang destruktif terhadap integritas geometri asli citra. Sebagai evaluasi, implementasi *Adaptive Median Filter*—di mana operasi spasial hanya dieksekusi secara reaktif pada piksel yang terdeteksi cacat (0/255) tanpa mengganggu piksel yang sehat—akan menjadi kerangka solusi yang jauh lebih ketat dan dapat dipertanggungjawabkan secara matematis.

### B. Tahap Equalization (CLAHE)
Pasca proses *denoising*, citra kehilangan ketajaman global dan memiliki rentang intensitas yang sempit sehingga tampak kusam (*low contrast*). Menerapkan pemerataan histogram global secara serampangan bukanlah solusi yang rasional, karena metode tersebut sering kali menghancurkan rasio warna (*color shifting*), memicu *washing out* pada area terang, dan secara tidak langsung meledakkan kembali sisa *noise floor* mikroskopis. Solusi yang dibangun dalam *pipeline* ini adalah mengisolasi informasi pencahayaan dengan mentransformasi citra ke ruang warna YCrCb, kemudian mengeksekusi algoritma CLAHE secara manual pada *tile* berukuran 8x8 secara khusus untuk kanal Luminance (Y). Batas pemotongan (*clip limit*) pada distribusi histogram menjamin kontras meningkat secara lokal tanpa over-amplifikasi, sementara keutuhan kanal krominansi (Cr, Cb) menjaga warna asli tidak bergeser. Untuk mencegah artefak visual berupa batas kotak-kotak (*blocky effect*) antar-*tile*, sistem mengandalkan interpolasi *bilinear* untuk menjahit peta ekualisasi secara mulus. Meskipun arsitektur logika ini sangat solid, penggunaan parameter `clip_limit=3.0` secara statis mengasumsikan bahwa distribusi degradasi bersifat homogen, sebuah asumsi dasar yang idealnya harus diuji dan disesuaikan secara dinamis berdasarkan perhitungan *variance* lokal citra input.

### C. Tahap Sharpening (Unsharp Masking)
Masalah krusial terakhir yang harus diselesaikan adalah kekaburan batas objek dan hilangnya tekstur mikro, yang murni merupakan efek samping dari penggunaan kernel spasial raksasa di tahap pertama. Sebagai solusi, diterapkan teknik *Unsharp Masking* yang bekerja dengan mengekstrak rentang frekuensi tinggi (dengan mengurangkan citra *denoised* dengan proyeksi blurnya sendiri), lalu menambahkan mask detail tersebut kembali ke citra dengan faktor penguatan sebesar 1.2. Pada awalnya, deteksi tepi murni menggunakan mask Laplacian sempat diuji untuk tujuan ini. Namun, Laplacian terbukti cacat secara operasional karena mask tersebut terlalu sensitif terhadap anomali butiran terkecil, sehingga menghasilkan *output* yang kotor dan berbintik. *Unsharp Masking* lebih masuk akal karena operasi penajamannya dilakukan secara relatif. Kendati demikian, terdapat kontradiksi logis yang nyata pada arsitektur linier ini: menyuntikkan kembali frekuensi tinggi tanpa diskriminasi berisiko secara langsung membatalkan fungsi Gaussian di tahap awal, karena algoritma tidak mampu membedakan antara "garis tepi asli objek" dan "sisa noise". Untuk perbaikan metodologis, kerangka kerja berbasis dekonvolusi (*Wiener Deconvolution* atau *Richardson-Lucy*) harus diutamakan di masa depan, karena pendekatan ini secara holistik memodelkan distorsi penyebab blur dan memulihkan data berdasarkan rasio *Signal-to-Noise*, bukan sekadar mempertebal kontras piksel secara artifisial.