let currentImageFilename = '';
let currentImageFilenameQ2 = '';

function showQuestion(questionNumber) {
    document.getElementById('question1').style.display = questionNumber === 1 ? 'block' : 'none';
    document.getElementById('question2').style.display = questionNumber === 2 ? 'block' : 'none';
}

function setLoading(isLoading, questionNumber) {
    const startButton = questionNumber === 1 ? document.getElementById('start') : document.getElementById('startQ2');
    startButton.disabled = isLoading;
    startButton.textContent = isLoading ? 'Processing...' : 'Start';
}

document.getElementById('selectImage').addEventListener('click', function() {
    document.getElementById('fileInput').click();
});

document.getElementById('fileInput').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        uploadFile(file, 1);
    }
});

document.getElementById('useSample').addEventListener('click', function() {
    fetch('/sample')
    .then(handleResponse)
    .then(data => {
        currentImageFilename = 'sample_source.jpg';
        document.getElementById('sourceImg').src = data.source;
        document.getElementById('groundTruthImg').src = data.groundtruth;
    })
    .catch(handleError);
});

document.getElementById('start').addEventListener('click', function() {
    processImage(1);
});

document.getElementById('selectImageQ2').addEventListener('click', function() {
    document.getElementById('fileInputQ2').click();
});

document.getElementById('fileInputQ2').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        uploadFile(file, 2);
    }
});

document.getElementById('startQ2').addEventListener('click', function() {
    processImage(2);
});

function uploadFile(file, questionNumber) {
    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(handleResponse)
    .then(data => {
        if (data.filename) {
            if (questionNumber === 1) {
                currentImageFilename = data.filename;
                document.getElementById('sourceImg').src = `/uploads/${data.filename}`;
                document.getElementById('groundTruthImg').src = '';
            } else {
                currentImageFilenameQ2 = data.filename;
                document.getElementById('sourceImgQ2').src = `/uploads/${data.filename}`;
            }
        } else {
            throw new Error('No filename received');
        }
    })
    .catch(handleError);
}

function processImage(questionNumber) {
    const filename = questionNumber === 1 ? currentImageFilename : currentImageFilenameQ2;
    if (!filename) {
        alert('Please select an image first');
        return;
    }

    setLoading(true, questionNumber);

    const endpoint = questionNumber === 1 ? '/process' : '/process_q2';

    fetch(endpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ filename: filename }),
    })
    .then(handleResponse)
    .then(data => {
        const timestamp = new Date().getTime();
        if (questionNumber === 1) {
            document.getElementById('algorithm1Img').src = `${data.result1}?t=${timestamp}`;
            document.getElementById('algorithm2Img').src = `${data.result2}?t=${timestamp}`;
        } else {
            document.getElementById('resultImgQ2').src = `${data.result}?t=${timestamp}`;
            document.getElementById('analysisTextQ2').innerText = data.analysis;
        }
    })
    .catch(handleError)
    .finally(() => setLoading(false, questionNumber));
}

function handleResponse(response) {
    if (!response.ok) {
        throw new Error('Network response was not ok');
    }
    return response.json();
}

function handleError(error) {
    console.error('Error:', error);
    alert('An error occurred: ' + error.message);
}

// Initialize the page to show Question 1
showQuestion(1);