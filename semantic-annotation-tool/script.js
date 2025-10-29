
let dataset = []
let importedReviews = []
let currentReviewIndex = 0
let totalReviews = 0

document.addEventListener("DOMContentLoaded", () => {
  bindShortcuts()
  updateProgressBar()
})

function bindShortcuts() {
  document.addEventListener("keydown", (e) => {
    const key = e.key.toLowerCase()
    if (key === "1") setSentiment("positive")
    if (key === "2") setSentiment("neutral")
    if (key === "3") setSentiment("negative")
    if (key === "s") saveReview()
    if (e.key === "ArrowRight") nextReview()
    if (e.key === "ArrowLeft") prevReview()
  })
}

function setSentiment(value) {
  const el = document.querySelector(`input[name="sentiment"][value="${value}"]`)
  if (el) el.checked = true
}

function getSelectedSentiment() {
  return document.querySelector('input[name="sentiment"]:checked')?.value || null
}

function clearSentiment() {
  document.querySelectorAll('input[name="sentiment"]').forEach(r => r.checked = false)
}

function saveReview() {
  const reviewText = document.getElementById("review-input").value.trim()
  if (!reviewText) {
    showNotification("No review to label!")
    return
  }
  const sentiment = getSelectedSentiment()
  if (!sentiment) {
    showNotification("Please select a sentiment (1=pos, 2=neu, 3=neg)")
    return
  }

  const orderNumber = currentReviewIndex + 1

  // Upsert
  const idx = dataset.findIndex(d => d.orderNumber === orderNumber)
  const payload = { review: reviewText, sentiment, orderNumber }
  if (idx >= 0) {
    dataset[idx] = payload
  } else {
    dataset.push(payload)
  }

  showNotification("Label saved!", "success")
  updateProgressBar()
}

function updateProgressBar() {
  const progressFill = document.getElementById("progress-fill")
  const progressText = document.getElementById("progress-text")

  const labeled = dataset.length
  const total = totalReviews

  const pct = total > 0 ? Math.round((labeled / total) * 100) : 0
  progressFill.style.width = pct + "%"
  progressText.textContent = `${labeled}/${total} Reviews`
}

function importAnnotations() {
  const fileInput = document.getElementById("file-input")
  if (!fileInput.files.length) {
    showNotification("Please select a file!")
    return
  }

  const file = fileInput.files[0]
  const reader = new FileReader()
  reader.onload = (e) => {
    // Split by lines that begin with #number, or fallback to non-empty lines
    const text = e.target.result
    let parts = text.split(/\n#\d+/).map(s => s.trim()).filter(Boolean)
    if (parts.length === 0) {
      parts = text.split(/\n+/).map(s => s.trim()).filter(Boolean)
    }

    importedReviews = parts
    totalReviews = importedReviews.length
    currentReviewIndex = 0

    if (totalReviews === 0) {
      showNotification("File seems empty!")
      return
    }

    loadCurrentReview()
    showNotification(`Imported ${totalReviews} reviews`, "success")
    updateProgressBar()
  }
  reader.readAsText(file)
}

function loadCurrentReview() {
  const ta = document.getElementById("review-input")
  if (currentReviewIndex < 0 || currentReviewIndex >= importedReviews.length) {
    ta.value = ""
    clearSentiment()
    return
  }

  const currentText = importedReviews[currentReviewIndex]
  ta.value = currentText

  // Restore label if already saved
  const orderNumber = currentReviewIndex + 1
  const existing = dataset.find(d => d.orderNumber === orderNumber)
  clearSentiment()
  if (existing) {
    setSentiment(existing.sentiment)
  }
}

function nextReview() {
  if (importedReviews.length === 0) {
    showNotification("Import a file first!")
    return
  }
  if (currentReviewIndex < importedReviews.length - 1) {
    currentReviewIndex += 1
    loadCurrentReview()
  } else {
    showNotification("This is the last review!", "info")
  }
}

function prevReview() {
  if (importedReviews.length === 0) {
    showNotification("Import a file first!")
    return
  }
  if (currentReviewIndex > 0) {
    currentReviewIndex -= 1
    loadCurrentReview()
  } else {
    showNotification("This is the first review!", "info")
  }
}

function downloadAnnotations() {
  if (dataset.length === 0) {
    showNotification("No data to export!")
    return
  }
  // Remove orderNumber for the final export
  const data = dataset
    .slice()
    .sort((a,b) => a.orderNumber - b.orderNumber)
    .map(({orderNumber, ...rest}) => rest)

  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(data, null, 2))
  const a = document.createElement("a")
  a.setAttribute("href", dataStr)
  a.setAttribute("download", "semantic_annotations.json")
  a.click()
  showNotification("Dataset downloaded!", "success")
}

// Notification system
function showNotification(message, type = "error") {
  let notification = document.getElementById("notification")
  if (!notification) {
    notification = document.createElement("div")
    notification.id = "notification"
    notification.style.position = "fixed"
    notification.style.bottom = "20px"
    notification.style.right = "20px"
    notification.style.padding = "10px 20px"
    notification.style.borderRadius = "4px"
    notification.style.color = "white"
    notification.style.fontWeight = "bold"
    notification.style.zIndex = "1000"
    notification.style.boxShadow = "0 3px 6px rgba(0,0,0,0.16)"
    notification.style.transition = "all 0.3s ease"
    document.body.appendChild(notification)
  }
  if (type === "success") notification.style.backgroundColor = "var(--success-color)"
  else if (type === "info") notification.style.backgroundColor = "var(--info-color)"
  else if (type === "warning") notification.style.backgroundColor = "var(--warning-color)"
  else notification.style.backgroundColor = "var(--danger-color)"

  notification.textContent = message
  notification.style.opacity = "1"
  setTimeout(() => { notification.style.opacity = "0" }, 2000)
}
