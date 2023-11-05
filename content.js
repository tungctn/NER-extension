function highlightSelectedText() {
  // const selection = window.getSelection();
  // if (!selection.rangeCount) return;
  // console.log(selection.toString());
  // const range = selection.getRangeAt(0);
  // const fragment = range.cloneContents();

  // // Tạo một thẻ <span> để chứa nội dung đã chọn
  // const span = document.createElement("span");
  // span.style.color = "red";
  // span.appendChild(fragment);

  // // Thay thế phạm vi đã chọn bằng thẻ <span> mới
  // range.deleteContents();
  // range.insertNode(span);
  const selection = window.getSelection();
  if (!selection.rangeCount) return;

  const selectedText = selection.toString();
  console.log(selectedText);

  // Gửi văn bản đã chọn đến API của bạn
  fetch("http://116.103.227.228:8000/api/ner", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      // Thêm bất kỳ headers nào khác cần thiết cho API của bạn
    },
    body: JSON.stringify({ text: selectedText }),
  })
    .then((response) => response.json())
    .then((data) => {
      console.log("API response:", data);
      // Tạo một thẻ <span> để chứa nội dung được API đánh dấu
      const range = selection.getRangeAt(0);
      const span = document.createElement("span");
      span.innerHTML = data.text; // Sử dụng innerHTML để chèn mã HTML từ API

      // Thay thế phạm vi đã chọn bằng thẻ <span> mới
      range.deleteContents();
      range.insertNode(span);
    })
    .catch((error) => {
      console.error("Error highlighting text:", error);
    });
}

document.body.addEventListener("mouseup", function () {
  highlightSelectedText();
});
