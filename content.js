
function highlightSelectedText() {
  const selection = window.getSelection();
  if (!selection.rangeCount) return;
  console.log(selection.toString());
  const range = selection.getRangeAt(0);
  const fragment = range.cloneContents();

  // Tạo một thẻ <span> để chứa nội dung đã chọn
  const span = document.createElement("span");
  span.style.color = "red";
  span.appendChild(fragment);

  // Thay thế phạm vi đã chọn bằng thẻ <span> mới
  range.deleteContents();
  range.insertNode(span);
}

document.body.addEventListener("mouseup", function () {
  highlightSelectedText();
});
