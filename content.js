async function highlightSelectedText() {
  const selection = window.getSelection();
  if (!selection.rangeCount) return false;

  let selectedText = selection.toString();
  if (!selectedText.trim().length) return false;

  // Get the range and the common ancestor element
  const selectionRange = selection.getRangeAt(0);
  const containerElement = selectionRange.commonAncestorContainer;

  // Check if the container is of type Node.TEXT_NODE, and get the parent if so
  const containingElement =
    containerElement.nodeType === Node.TEXT_NODE
      ? containerElement.parentNode
      : containerElement;

  // Now you have the containing element, you can get its HTML
  const containingElementHtml = containingElement.outerHTML;
  console.log("Containing element HTML:", containingElementHtml);

  console.log("Selected text:", selectedText);
  const response = await fetch("http://116.103.227.228:8000/api/ner", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text: selectedText }),
  });

  const data = await response.json();
  console.log("API response:", data);

  let markedText = selectedText;

  function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  let uniqueEntities = {};
  data.entities.forEach((entity) => {
    if (!uniqueEntities.hasOwnProperty(entity.text)) {
      uniqueEntities[entity.text] = entity;
    }
  });
  console.log("Unique entities:", uniqueEntities);
  entities = Object.values(uniqueEntities);
  console.log("Filtered entities:", entities);

  entities.forEach((entity) => {
    const style = getStyleByType(entity.type);
    const regex = new RegExp(escapeRegExp(entity.text), "g");
    markedText = markedText.replace(
      regex,
      `<span style="${style}">${entity.text}</span>`
    );
  });

  console.log("Marked text:", markedText);

  const range = selection.getRangeAt(0);
  range.deleteContents();

  const fragment = document.createRange().createContextualFragment(markedText);
  range.insertNode(fragment);

  selection.removeAllRanges();

  return true;
}

function getStyleByType(type) {
  switch (type) {
    case "PER":
      return "display: inline-block; padding: 5px; border: 2px solid red; background-color: #ffcccc; margin-right: 5px; border-radius: 10px;";
    case "ORG":
      return "display: inline-block; padding: 5px; border: 2px solid #00FFFF; background-color: #F0FFFF; margin-right: 5px; border-radius: 10px;";
    case "LOC":
      return "display: inline-block; padding: 5px; border: 2px solid #FFD700; background-color: #FFFDD0; margin-right: 5px; border-radius: 10px;";
    default:
      return "display: inline-block; padding: 5px; border: 2px solid #0FFF50; background-color: #ECFFDC; margin-right: 5px; border-radius: 10px;";
  }
}

document.body.addEventListener("mouseup", function () {
  highlightSelectedText().catch(console.error);
});
