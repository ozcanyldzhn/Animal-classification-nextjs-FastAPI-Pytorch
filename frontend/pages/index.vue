<template>
  <div class="min-h-screen bg-gradient-to-b from-blue-100 to-white flex flex-col items-center py-10 px-4">
    <!-- Başlık -->
    <h1 class="text-3xl md:text-4xl font-bold text-center text-blue-800 mb-10">
      Hadi tişörtün <span class="text-red-500">Kısa kollu mu</span> <span class="text-green-600">Uzun Kollu mu</span> olduğunu bulalım!
    </h1>

    <!-- Yükleme & Tahmin Kutusu -->
    <div class="bg-white rounded-2xl shadow-xl w-full max-w-4xl flex flex-col md:flex-row p-6 gap-6 items-start">
      <!-- Sol: Görsel Önizleme ve Dosya Yükleme -->
      <div class="flex-1">
        <!-- Dosya Yükleme Alanı -->
        <label class="block border-2 border-dashed border-blue-300 p-6 rounded-xl text-center cursor-pointer hover:border-blue-500 transition">
          <input type="file" @change="onFileChange" accept="image/*" class="hidden" />
          <span class="text-blue-600 font-medium">Bir fotoğraf seçmek için tıkla</span>
        </label>

        <!-- Görsel Önizleme -->
        <div v-if="imageUrl" class="mt-4 rounded-xl overflow-hidden shadow border">
          <img :src="imageUrl" alt="Yüklenen görsel" class="w-full h-auto object-contain" />
        </div>
      </div>

      <!-- Sağ: Tahmin & Sonuç -->
      <div class="flex flex-col justify-between flex-1 h-full w-full">
        <!-- Tahmin Butonu -->
        <button
          @click="uploadImage"
          :disabled="!selectedFile"
          class="bg-gradient-to-r from-green-400 to-blue-500 text-white font-semibold px-6 py-3 rounded-full shadow-lg hover:scale-105 transition disabled:opacity-50 self-end"
        >
          Tahmin Et
        </button>

        <!-- Sonuç -->
        <div
          class="mt-6 border border-green-300 bg-green-50 text-green-800 px-4 py-3 rounded-xl shadow min-h-[60px] text-center text-lg font-medium"
        >
          <span v-if="result">Tahmin: {{ result }}</span>
          <span v-else class="text-gray-400">Henüz bir tahmin yapılmadı</span>
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
