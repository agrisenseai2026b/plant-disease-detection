const input = document.getElementById("leafImage");
const preview = document.getElementById("preview");
const fileLabel = document.getElementById("fileLabel");
const form = document.getElementById("uploadForm");
const button = document.getElementById("submitButton");

if (input) {
  input.addEventListener("change", () => {
    const file = input.files[0];

    if (!file) return;

    fileLabel.textContent = file.name;

    if (preview.dataset.url) {
      URL.revokeObjectURL(preview.dataset.url);
    }

    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.dataset.url = url;
    preview.hidden = false;
  });
}

if (form) {
  form.addEventListener("submit", () => {
    button.disabled = true;
    button.textContent = "Analyzing leaf...";
  });
}