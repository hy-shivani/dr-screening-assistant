import express from "express";
import cors from "cors";
import multer from "multer";
import axios from "axios";
import FormData from "form-data";

const app = express();
const PORT = Number(process.env.PORT || 4000);
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://localhost:8000";
app.use(cors());
app.use(express.json());

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    const allowed = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
    if (allowed.includes(file.mimetype)) cb(null, true);
    else cb(new Error("Unsupported file type. Upload JPG, PNG, WEBP or BMP."));
  }
});

app.get("/health", (_req, res) => res.json({ status: "ok", service: "backend" }));

app.post("/api/screen", upload.single("image"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "Upload an image in the 'image' field." });
  try {
    const form = new FormData();
    form.append("image", req.file.buffer, {
      filename: req.file.originalname || "fundus-image.jpg",
      contentType: req.file.mimetype
    });
    const response = await axios.post(`${ML_SERVICE_URL}/predict`, form, {
      headers: form.getHeaders(), timeout: 120000, maxBodyLength: Infinity
    });
    res.json(response.data);
  } catch (error) {
    res.status(error.response?.status || 502).json({
      error: error.response?.data?.detail || "Could not get a result from the ML service.",
      hint: "Check that the Python ML service is running and reachable."
    });
  }
});

app.use((err, _req, res, _next) => {
  if (err instanceof multer.MulterError && err.code === "LIMIT_FILE_SIZE")
    return res.status(413).json({ error: "Maximum image size is 10 MB." });
  res.status(400).json({ error: err.message || "Bad request." });
});
app.listen(PORT, () => console.log(`Backend listening on http://localhost:${PORT}`));
