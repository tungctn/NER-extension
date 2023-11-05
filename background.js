chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  if (request.action === "executeScript") {
    chrome.scripting.executeScript({
      target: { tabId: request.tabId },
      function: functionToExecute,
    });
  }
});

function functionToExecute() {
  return window.getSelection().toString();
}
