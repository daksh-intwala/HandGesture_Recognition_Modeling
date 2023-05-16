const mediaSelector = document.getElementById("media");
let selectedMedia = null;
  
// Handler function to handle the "change" event
// when the user selects some option
mediaSelector.addEventListener("change", (e) => {
    selectedMedia = e.target.value;
    document.getElementById(
      `${selectedMedia}-recorder`).style.display = "block";
    document.getElementById(
      `${otherRecorder(selectedMedia)}-recorder`)
      .style.display = "none";
});
  
function otherRecorder(selectedMedia) {
    return selectedMedia === "vid" ? "aud" : "vid";
}