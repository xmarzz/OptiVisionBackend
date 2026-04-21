package main

import (
	"fmt"
	"image"
	_ "image/jpeg" // allow decoding jpegs
	_ "image/png"  // allow decoding pngs
	"mime/multipart"
	"net/http"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	ort "github.com/yalue/onnxruntime_go"
	"golang.org/x/image/draw"
)

// 1. Structure for clean JSON output
type Detection struct {
	ClassID    int       `json:"class_id"`
	ClassName  string    `json:"class"`
	Confidence float32   `json:"confidence"`
	BBox       []float32 `json:"bbox"` // [x_center, y_center, width, height]
}

// 2. Prepares the image for the YOLOv8 ONNX model
func preprocessImage(file multipart.File) ([]float32, error) {
	img, _, err := image.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %v", err)
	}

	// Resize to 416x416 (The size used during your YOLO training)
	bounds := image.Rect(0, 0, 416, 416)
	resized := image.NewRGBA(bounds)
	draw.BiLinear.Scale(resized, resized.Bounds(), img, img.Bounds(), draw.Src, nil)

	// Convert to flat float32 array [1, 3, 416, 416]
	floatArray := make([]float32, 3*416*416)
	for y := 0; y < 416; y++ {
		for x := 0; x < 416; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			floatArray[(0*416*416)+(y*416)+x] = float32(r) / 65535.0
			floatArray[(1*416*416)+(y*416)+x] = float32(g) / 65535.0
			floatArray[(2*416*416)+(y*416)+x] = float32(b) / 65535.0
		}
	}
	return floatArray, nil
}

func main() {
	// 3. Initialize the C++ ONNX Engine
	// ort.SetSharedLibraryPath("onnxruntime.dll")
	ort.SetSharedLibraryPath("D:\\MyApp\\personal-dev\\backend_OptiVisionEngine\\onnxruntime.dll")
	err := ort.InitializeEnvironment()
	if err != nil {
		fmt.Printf("Fatal Error: Could not initialize ONNX. Is onnxruntime.dll in the folder? Error: %v\n", err)
		return
	}
	defer ort.DestroyEnvironment()

	router := gin.Default()

	// Enable CORS so your frontend can talk to this backend
	router.Use(cors.New(cors.Config{
		AllowAllOrigins: true,
		AllowMethods:    []string{"POST", "GET", "OPTIONS"},
		AllowHeaders:    []string{"Origin", "Content-Type", "Accept"},
	}))

	router.POST("/detect", func(c *gin.Context) {
		fileHeader, err := c.FormFile("image")
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "No image uploaded"})
			return
		}

		file, err := fileHeader.Open()
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to open image"})
			return
		}
		defer file.Close()

		// 4. Preprocess the image
		tensorArray, err := preprocessImage(file)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to preprocess image"})
			return
		}

		// 5. Setup ONNX Tensors
		inputShape := ort.NewShape(1, 3, 416, 416)
		inputTensor, _ := ort.NewTensor(inputShape, tensorArray)
		defer inputTensor.Destroy()

		outputShape := ort.NewShape(1, 6, 3549)
		outputTensor, _ := ort.NewEmptyTensor[float32](outputShape)
		defer outputTensor.Destroy()

		// 6. Run AI Inference
		session, err := ort.NewAdvancedSession("best.onnx",
			[]string{"images"}, []string{"output0"},
			[]ort.ArbitraryTensor{inputTensor}, []ort.ArbitraryTensor{outputTensor}, nil)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Model failed to load"})
			return
		}
		defer session.Destroy()

		_ = session.Run()
		rawOutput := outputTensor.GetData()

		// 7. Post-Processing: Translate Raw Math to Clean JSON
		
		// var validDetections []Detection

		validDetections := []Detection{}

		classNames := map[int]string{
			0: "Helmet",
			1: "No Helmet",
		}

		highestConfidence := float32(0.0)

		for i := 0; i < 3549; i++ {
			x := rawOutput[0*3549+i]
			y := rawOutput[1*3549+i]
			w := rawOutput[2*3549+i]
			h := rawOutput[3*3549+i]
			confHelmet := rawOutput[4*3549+i]
			confNoHelmet := rawOutput[5*3549+i]

			maxConf := confHelmet
			classID := 0
			if confNoHelmet > maxConf {
				maxConf = confNoHelmet
				classID = 1
			}

			if maxConf > highestConfidence {
				highestConfidence = maxConf 
			}

			// Only keep boxes where the AI is over 50% confident
			if maxConf > 0.10 {
				validDetections = append(validDetections, Detection{
					ClassID:    classID,
					ClassName:  classNames[classID],
					Confidence: maxConf,
					BBox:       []float32{x, y, w, h},
				})
			}
		}

		fmt.Printf("Intefernece complete, highest ai confience score : %.2f%%/n",highestConfidence*100)

		// 8. Send the true coordinates back to the frontend
		c.JSON(http.StatusOK, gin.H{
			"status": "success",
			"data":   validDetections,
		})
	})

	fmt.Println("OptiVisionEngine Backend running on http://localhost:8080...")
	router.Run(":8080")
}