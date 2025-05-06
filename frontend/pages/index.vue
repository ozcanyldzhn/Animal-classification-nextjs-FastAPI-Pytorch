<!-- <template>
    <div class="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-6">
      <h1 class="text-3xl font-bold mb-4">T-Shirt Sınıflandırıcı</h1>
  
      <div
        class="w-full max-w-md p-6 bg-white rounded-2xl shadow-md flex flex-col items-center gap-4"
      >
        <input type="file" @change="handleFileUpload" class="block w-full text-sm text-gray-700" />
  
        <div v-if="imageUrl" class="mt-4 w-full">
          <img :src="imageUrl" alt="Yüklenen Görsel" class="rounded-xl shadow w-full" />
        </div>
  
        <button
          :disabled="!image"
          @click="sendToAPI"
          class="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
        >
          Tahmin Et
        </button>
  
        <div v-if="result" class="mt-4 text-center">
          <p class="text-lg font-semibold">
            Sonuç: <span class="text-blue-600">{{ result.label }}</span>
          </p>
          <p class="text-sm text-gray-500">Güven: {{ result.confidence }}%</p>
        </div>
      </div>
    </div>
  </template>
  
  <script setup>
  import { ref } from 'vue'
  import axios from 'axios'
  
  const image = ref(null)
  const imageUrl = ref(null)
  const result = ref(null)
  
  const handleFileUpload = (event) => {
    const file = event.target.files[0]
    image.value = file
    imageUrl.value = URL.createObjectURL(file)
  }
  
  const sendToAPI = async () => {
    const formData = new FormData()
    formData.append('file', image.value)
  
    const response = await axios.post('http://localhost:8000/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  
    result.value = response.data
  }
  </script>
   -->

   <template>
    <div class="p-4 max-w-md mx-auto">
      <h1 class="text-xl font-bold mb-4">Tişört Sınıflandırma</h1>
  
      <input type="file" @change="onFileChange" accept="image/*" class="mb-4" />
      
      <button @click="uploadImage" class="bg-blue-500 text-white px-4 py-2 rounded">
        Tahmin Et
      </button>
  
      <p v-if="result" class="mt-4">Tahmin: {{ result }}</p>
    </div>
  </template>
  
  <script setup>
  import { ref } from 'vue'
  import axios from 'axios'
  
  const selectedFile = ref(null)
  const result = ref(null)
  
  const onFileChange = (event) => {
    selectedFile.value = event.target.files[0]
  }
  
  const uploadImage = async () => {
    if (!selectedFile.value) return
  
    const formData = new FormData()
    formData.append('file', selectedFile.value)
  
    try {
      const response = await axios.post('http://127.0.0.1:8000/predict/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      result.value = response.data.prediction
    } catch (error) {
      result.value = 'Hata oluştu.'
      console.error(error)
    }
  }
  </script>
  