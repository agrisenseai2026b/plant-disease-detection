
const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");

const app = express();
const PORT = process.env.PORT || 3000;

const ROOT = __dirname;
const UPLOAD_DIR = path.join(ROOT, "uploads");
const PYTHON_SCRIPT = path.join(ROOT, "predict.py");

// Change this to your virtual environment's Python executable
// if your model dependencies are installed in a virtual environment.
const PYTHON = process.env.PYTHON_PATH || "python";

fs.mkdirSync(UPLOAD_DIR, { recursive: true });

app.set("view engine", "ejs");
app.set("views", path.join(ROOT, "views"));

app.use(express.static(path.join(ROOT, "public")));

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOAD_DIR),

  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    cb(null, `${Date.now()}-${require("crypto").randomUUID()}${ext}`);
  }
});

const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowed = new Set([
      ".jpg", ".jpeg", ".png", ".webp"
    ]);

    const ext = path.extname(file.originalname).toLowerCase();

    if (!allowed.has(ext) || !file.mimetype.startsWith("image/")) {
      return cb(new Error("Upload a JPG, PNG, or WebP leaf image."));
    }

    cb(null, true);
  }
});

app.get("/", (req, res) => {
  res.render("index", { error: null });
});

function runPrediction(imagePath) {
  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON, [PYTHON_SCRIPT, imagePath], {
      cwd: ROOT,
      windowsHide: true
    });

    let stdout = "";
    let stderr = "";

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");

    child.stdout.on("data", chunk => {
      stdout += chunk;
      if (stdout.length > 1024 * 1024) {
        child.kill();
        reject(new Error("Prediction output was too large."));
      }
    });

    child.stderr.on("data", chunk => {
      stderr += chunk;
    });

    child.on("error", reject);

    child.on("close", code => {
      try {
        const result = JSON.parse(stdout.trim());

        if (code !== 0 || !result.success) {
          return reject(
            new Error(result.error || "Prediction failed.")
          );
        }

        resolve(result);
      } catch {
        reject(new Error(
          "Could not read the Python prediction result. " +
          (stderr || "Check your Python environment.")
        ));
      }
    });
  });
}

app.post("/predict", (req, res) => {
  upload.single("leafImage")(req, res, async err => {
    if (err) {
      return res.status(400).render("index", {
        error: err.message
      });
    }

    if (!req.file) {
      return res.status(400).render("index", {
        error: "Please select a leaf image first."
      });
    }

    try {
      const result = await runPrediction(req.file.path);

      res.render("result", {
        result,
        imageUrl: `/uploads/${path.basename(req.file.path)}`
      });
    } catch (error) {
      console.error("Prediction error:", error.message);

      res.status(500).render("index", {
        error: "Prediction failed. Check your Python setup and model files."
      });
    } finally {
      // Remove the temporary uploaded image after prediction.
      // If you want to show the preview on the results page,
      // we will instead retain it temporarily and serve it securely.
      // For now, delete it after inference.
      fs.promises.unlink(req.file.path).catch(() => {});
    }
  });
});

app.use((err, req, res, next) => {
  console.error(err.message);
  res.status(500).send("An unexpected server error occurred.");
});

app.listen(PORT, () => {
  console.log(`Plant disease app running at http://localhost:${PORT}`);
});
