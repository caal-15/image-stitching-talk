function readTextFile(file, fileDisplayArea) {
  var rawFile = new XMLHttpRequest()
  rawFile.open('GET', file, false)
  rawFile.onreadystatechange = function () {
    if(rawFile.readyState === 4) {
      if(rawFile.status === 200 || rawFile.status == 0) {
        var allText = rawFile.responseText
        fileDisplayArea.innerHTML = allText
        console.log(fileDisplayArea)
      }
    }
  }
  rawFile.send(null);
}

readTextFile('markdown/slides.md', document.getElementById('source'))
