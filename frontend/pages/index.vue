<template>
  <div class="main-container d-flex flex-column align-items-center py-5 px-3">
    <!-- Başlık -->
    <h1 class="text-center fw-bold text-white mb-4 display-4">
      Hayvan Sınıflandırma <span class="text-success">AI</span>
    </h1>
    <p class="text-center text-white mb-4 lead">
      Bir hayvan fotoğrafı yükleyin ve yapay zeka hangi hayvan olduğunu tahmin etsin!
    </p>

    <!-- Yükleme & Tahmin Kutusu -->
    <div class="bg-white rounded-4 shadow p-4 w-100 prediction-box">
      <div class="row g-4">
        <!-- Sol: Görsel Yükleme -->
        <div class="col-md-6">
          <label class="form-label w-100 border border-2 border-primary p-4 rounded-3 text-center d-block bg-light upload-label">
            <input type="file" @change="onFileChange" accept="image/*" class="d-none" />
            <span class="text-primary fw-medium">Bir hayvan fotoğrafı seçmek için tıkla</span>
          </label>

          <!-- Önizleme -->
          <div v-if="imageUrl" class="mt-3 border rounded-3 overflow-hidden shadow-sm d-flex justify-content-center align-items-center">
            <img :src="imageUrl" alt="Yüklenen görsel" class="img-fluid" />
          </div>
        </div>

        <!-- Sağ: Tahmin & Sonuç -->
        <div class="col-md-6 d-flex flex-column justify-content-between">
          <!-- Buton -->
          <button
            @click="uploadImage"
            :disabled="!selectedFile || loading"
            class="btn btn-success w-100 fw-semibold py-2 mb-3 predict-button"
          >
            <span v-if="loading">
              <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
              Tahmin Yapılıyor...
            </span>
            <span v-else>Tahmin Et</span>
          </button>

          <!-- Sonuç -->
          <div v-if="predictions.length > 0" class="mt-auto">
            <div class="alert alert-success mb-3">
              <h5 class="alert-heading mb-2">En İyi 3 Tahmin:</h5>
              <div v-for="(pred, index) in predictions" :key="index" class="d-flex justify-content-between align-items-center mb-2">
                <span class="fw-medium">{{ index + 1 }}. {{ pred.class }}</span>
                <span class="badge bg-primary">{{ pred.confidence }}</span>
              </div>
            </div>

          </div>
          <div v-else class="alert alert-secondary text-muted mt-auto">
            Henüz bir tahmin yapılmadı
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'

const selectedFile = ref(null)
const imageUrl = ref(null)
const predictions = ref([])
const loading = ref(false)
const inferenceTime = ref(0)

const onFileChange = (event) => {
  const file = event.target.files[0]
  selectedFile.value = file
  imageUrl.value = URL.createObjectURL(file)
  predictions.value = [] // Yeni dosya seçildiğinde tahminleri sıfırla
}

const uploadImage = async () => {
  if (!selectedFile.value) return

  loading.value = true
  predictions.value = []

  const formData = new FormData()
  formData.append('file', selectedFile.value)

  try {
    const response = await axios.post('http://127.0.0.1:8000/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    predictions.value = response.data.predictions
    inferenceTime.value = response.data.inference_time
  } catch (error) {
    console.error(error)
    predictions.value = [{
      class: 'Hata',
      confidence: 'Bir hata oluştu'
    }]
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.main-container {
  min-height: 100vh;
  position: relative;
  z-index: 2;
  background-image: url('resim.png');
  background-size: cover;
  background-position: center;
}

.prediction-box {
  max-width: 900px;
}

.upload-label {
  cursor: pointer;
  transition: all 0.3s ease;
  border-style: dashed !important;
}

.upload-label:hover {
  background-color: #f8f9fa;
  transform: translateY(-2px);
}

.predict-button {
  cursor: pointer;
  transition: all 0.3s ease;
}

.predict-button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.predict-button:disabled {
  cursor: not-allowed;
  opacity: 0.7;
}

.alert {
  border-radius: 0.5rem;
}

.badge {
  font-size: 0.875rem;
  padding: 0.5em 0.75em;
}
</style>
