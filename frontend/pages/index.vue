<template>
  <div class="min-vh-100 bg-light d-flex flex-column align-items-center py-5 px-3">
    <!-- Başlık -->
    <h1 class="text-center fw-bold text-primary mb-4 fs-3 fs-md-1">
      Hadi tişörtün <span class="text-danger">Kısa kollu mu</span> <span class="text-success">Uzun Kollu mu</span> olduğunu bulalım!
    </h1>

    <!-- Yükleme & Tahmin Kutusu -->
    <div class="bg-white rounded shadow p-4 w-100" style="max-width: 900px;">
      <div class="row g-4">
        <!-- Sol: Görsel Yükleme -->
        <div class="col-md-6">
          <label class="form-label w-100 border border-2 border-primary border-dashed p-4 rounded text-center d-block bg-light cursor-pointer">
            <input type="file" @change="onFileChange" accept="image/*" class="d-none" />
            <span class="text-primary fw-medium">Bir fotoğraf seçmek için tıkla</span>
          </label>

          <!-- Önizleme -->
          <div v-if="imageUrl" class="mt-3 border rounded overflow-hidden shadow-sm">
            <img :src="imageUrl" alt="Yüklenen görsel" class="img-fluid" />
          </div>
        </div>

        <!-- Sağ: Tahmin & Sonuç -->
        <div class="col-md-6 d-flex flex-column justify-content-between">
          <!-- Buton -->
          <button
            @click="uploadImage"
            :disabled="!selectedFile"
            class="btn btn-success w-100 fw-semibold py-2 mb-3"
          >
            Tahmin Et
          </button>

          <!-- Sonuç -->
          <div
            class="alert mt-auto"
            :class="result ? 'alert-success' : 'alert-secondary text-muted'"
          >
            <span v-if="result">Tahmin: {{ result }}</span>
            <span v-else>Henüz bir tahmin yapılmadı</span>
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
const result = ref(null)

const onFileChange = (event) => {
  const file = event.target.files[0]
  selectedFile.value = file
  imageUrl.value = URL.createObjectURL(file)
}

const uploadImage = async () => {
  if (!selectedFile.value) return

  const formData = new FormData()
  formData.append('file', selectedFile.value)

  try {
    const response = await axios.post('http://127.0.0.1:8000/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    result.value = response.data.prediction
  } catch (error) {
    console.error(error)
    result.value = 'Bir hata oluştu.'
  }
}
</script>
