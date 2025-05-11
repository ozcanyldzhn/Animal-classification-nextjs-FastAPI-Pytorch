"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Upload, ArrowRight, Loader2, Camera, RefreshCw } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import axios from "axios"
import { getTurkishAnimalName } from "@/lib/animal-translations"

// Client-side only component for particles
const ClientParticles = () => {
  const [particles, setParticles] = useState<
    Array<{
      id: number
      width: number
      height: number
      left: string
      top: string
      duration: number
      distance: number
    }>
  >([])

  useEffect(() => {
    // Generate particles only on the client side
    const newParticles = Array.from({ length: 20 }).map((_, i) => ({
      id: i,
      width: Math.random() * 10 + 5,
      height: Math.random() * 10 + 5,
      left: `${Math.random() * 100}%`,
      top: `${Math.random() * 100}%`,
      duration: Math.random() * 10 + 10,
      distance: Math.random() * 100 + 50,
    }))
    setParticles(newParticles)
  }, [])

  return (
    <>
      {particles.map((particle) => (
        <motion.div
          key={particle.id}
          className="absolute rounded-full bg-white opacity-20"
          style={{
            width: particle.width,
            height: particle.height,
            left: particle.left,
            top: particle.top,
          }}
          animate={{
            y: [0, -particle.distance],
            opacity: [0.2, 0],
          }}
          transition={{
            duration: particle.duration,
            repeat: Number.POSITIVE_INFINITY,
            ease: "linear",
          }}
        />
      ))}
    </>
  )
}

interface Prediction {
  class: string
  confidence: string
}

export default function AnimalClassification() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [loading, setLoading] = useState(false)
  const [inferenceTime, setInferenceTime] = useState(0)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const onFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      setImageUrl(URL.createObjectURL(file))
      setPredictions([])
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files?.[0]
    if (file) {
      setSelectedFile(file)
      setImageUrl(URL.createObjectURL(file))
      setPredictions([])
    }
  }

  const uploadImage = async () => {
    if (!selectedFile) return

    setLoading(true)
    setPredictions([])

    const formData = new FormData()
    formData.append("file", selectedFile)

    try {
      const response = await axios.post("https://animal-api.onrender.com/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      setPredictions(response.data.predictions)
      setInferenceTime(response.data.inference_time)
    } catch (error) {
      console.error(error)
      setPredictions([
        {
          class: "Hata",
          confidence: "Bir hata oluştu",
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  const resetForm = () => {
    setSelectedFile(null)
    setImageUrl(null)
    setPredictions([])
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  const getConfidenceValue = (confidence: string) => {
    // Extract percentage value from string like "95.6%" and convert to number
    const match = confidence.match(/(\d+(\.\d+)?)%?/)
    return match ? Number.parseFloat(match[1]) : 0
  }

  return (
    <div className="min-h-screen w-full bg-gradient-to-b from-emerald-900 to-emerald-700 flex flex-col items-center px-4 py-12 sm:py-16">
      {/* Animated particles background */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute inset-0 bg-[url('/placeholder.svg?height=1080&width=1920')] bg-cover bg-center opacity-10"></div>
        <ClientParticles />
      </div>

      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-center mb-8 sm:mb-12 relative z-10"
      >
        <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold text-white mb-4">
          Hayvan Sınıflandırma <span className="text-emerald-300">AI</span>
        </h1>
        <p className="text-lg sm:text-xl text-emerald-100 max-w-2xl mx-auto">
          Bir hayvan fotoğrafı yükleyin ve yapay zeka hangi hayvan olduğunu tahmin etsin!
        </p>
      </motion.div>

      {/* Main Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
        className="w-full max-w-4xl"
      >
        <Card className="border-0 shadow-2xl overflow-hidden bg-white/95 backdrop-blur-sm">
          <CardContent className="p-0">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-0">
              {/* Left side - Upload */}
              <div className="p-6 sm:p-8 border-b md:border-b-0 md:border-r border-gray-200">
                <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
                  <Camera className="mr-2 h-5 w-5 text-emerald-600" />
                  Görsel Yükle
                </h2>

                {/* Upload area */}
                <div
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                  className={`
                    relative border-2 border-dashed rounded-xl 
                    ${imageUrl ? "border-emerald-300 bg-emerald-50" : "border-gray-300 bg-gray-50"} 
                    transition-all duration-300 hover:border-emerald-400 hover:bg-emerald-50
                    flex flex-col items-center justify-center p-8 cursor-pointer group
                  `}
                >
                  <input ref={fileInputRef} type="file" onChange={onFileChange} accept="image/*" className="hidden" />

                  <AnimatePresence mode="wait">
                    {!imageUrl ? (
                      <motion.div
                        key="upload-prompt"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="text-center"
                      >
                        <div className="mb-4 bg-emerald-100 p-4 rounded-full inline-flex items-center justify-center">
                          <Upload className="h-8 w-8 text-emerald-600 group-hover:scale-110 transition-transform duration-300" />
                        </div>
                        <p className="text-emerald-700 font-medium mb-2">Bir hayvan fotoğrafı seçmek için tıkla</p>
                        <p className="text-sm text-gray-500">veya dosyayı buraya sürükle</p>
                      </motion.div>
                    ) : (
                      <motion.div
                        key="image-preview"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="relative w-full aspect-square rounded-lg overflow-hidden"
                      >
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={imageUrl || "/placeholder.svg"}
                          alt="Yüklenen görsel"
                          className="w-full h-full object-cover"
                        />
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            resetForm()
                          }}
                          className="absolute bottom-3 right-3 bg-white/90 hover:bg-white p-2 rounded-full shadow-md transition-all duration-300"
                        >
                          <RefreshCw className="h-5 w-5 text-emerald-600" />
                        </button>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </div>

              {/* Right side - Prediction */}
              <div className="p-6 sm:p-8 flex flex-col">
                <h2 className="text-xl font-semibold text-gray-800 mb-4">Tahmin Sonuçları</h2>

                {/* Predict button */}
                <Button
                  onClick={uploadImage}
                  disabled={!selectedFile || loading}
                  className="w-full mb-6 bg-emerald-600 hover:bg-emerald-700 text-white py-6 rounded-lg transition-all duration-300 disabled:opacity-50"
                >
                  {loading ? (
                    <motion.div
                      className="flex items-center justify-center"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                    >
                      <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                      <span>Tahmin Yapılıyor...</span>
                    </motion.div>
                  ) : (
                    <motion.div
                      className="flex items-center justify-center"
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      <span>Tahmin Et</span>
                      <ArrowRight className="ml-2 h-5 w-5" />
                    </motion.div>
                  )}
                </Button>

                {/* Results */}
                <div className="flex-1">
                  <AnimatePresence mode="wait">
                    {predictions.length > 0 ? (
                      <motion.div
                        key="results"
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        transition={{ duration: 0.3 }}
                        className="bg-emerald-50 border border-emerald-200 rounded-lg p-4"
                      >
                        <h3 className="font-semibold text-emerald-800 mb-3">En İyi 3 Tahmin:</h3>
                        <div className="space-y-3">
                          {predictions.map((pred, index) => (
                            <motion.div
                              key={index}
                              initial={{ opacity: 0, x: -10 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ duration: 0.3, delay: index * 0.1 }}
                              className="bg-white rounded-lg p-3 shadow-sm"
                            >
                              <div className="flex justify-between items-center mb-1">
                                <span className="font-medium text-gray-800">
                                  {index + 1}. {getTurkishAnimalName(pred.class)}
                                </span>
                                <span className="bg-emerald-100 text-emerald-800 px-2 py-1 rounded text-sm font-medium">
                                  {pred.confidence}
                                </span>
                              </div>
                              <Progress value={getConfidenceValue(pred.confidence)} className="h-2 bg-emerald-100" />
                            </motion.div>
                          ))}
                        </div>
                      </motion.div>
                    ) : (
                      <motion.div
                        key="no-results"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="bg-gray-50 border border-gray-200 rounded-lg p-4 text-center flex-1 flex items-center justify-center"
                      >
                        <p className="text-gray-500">
                          {selectedFile
                            ? "Tahmin yapmak için 'Tahmin Et' butonuna tıklayın"
                            : "Henüz bir görsel yüklenmedi"}
                        </p>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Footer */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.4 }}
        className="mt-8 text-center text-emerald-200 text-sm"
      >
        <p>Hayvan Sınıflandırma AI &copy; {new Date().getFullYear()}</p>
      </motion.div>
    </div>
  )
}
