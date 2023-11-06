async function highlightSelectedText() {
  const selection = window.getSelection();
  if (!selection.rangeCount) return;

  const selectedText = selection.toString();
  console.log(selectedText);
  const response = await fetch("http://116.103.227.228:8000/api/ner", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text: selectedText }),
  });
  const data = await response.json();
  console.log("API response:", data);

  const range = selection.getRangeAt(0);
  const span = document.createElement("span");
  span.innerHTML = data.text;
  console.log(data.text);

  range.deleteContents();
  range.insertNode(span);
}

document.body.addEventListener("mouseup", function () {
  highlightSelectedText();
});
