// ==============================
// START IDS
// ==============================

const startBtn = document.getElementById("startBtn");

if(startBtn){
startBtn.onclick = async () => {

console.log("Starting IDS");

await fetch("http://localhost:5000/start");

};
}


// ==============================
// STOP IDS
// ==============================

const stopBtn = document.getElementById("stopBtn");

if(stopBtn){
stopBtn.onclick = async () => {

console.log("Stopping IDS");

await fetch("http://localhost:5000/stop");

};
}


// ==============================
// LOAD IDS DATA
// ==============================

async function loadData(){

try{

const response = await fetch("../ids_predictions.json");

const data = await response.json();

updateUI(data);

}catch(error){

console.log("Waiting for IDS data...");

}

}


// ==============================
// UPDATE UI
// ==============================

function updateUI(data){

const alertsDiv = document.getElementById("alerts");
const terminal = document.getElementById("terminalFeed");

if(alertsDiv) alertsDiv.innerHTML="";

data.slice(-5).reverse().forEach(x=>{

// Terminal Feed
if(terminal){

terminal.innerHTML += `
<div class="feed">
${x.time} | ${x.source} → ${x.destination} | ${x.prediction}
</div>
`;

terminal.scrollTop = terminal.scrollHeight;

}

// Alerts
if(x.prediction !== "Normal Traffic" && alertsDiv){

alertsDiv.innerHTML += `

<div class="alert">

🚨 ${x.prediction} Detected<br>
Source: ${x.source}<br>
Target: ${x.destination}<br>
Confidence: ${x.confidence} %

</div>

`;

}

});

}


// ==============================
// AUTO REFRESH
// ==============================

setInterval(loadData,2000);