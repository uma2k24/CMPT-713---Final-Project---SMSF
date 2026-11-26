'use strict';
function fitStatus(p) {
  return 'All six geometric controls applied. Maximum measurement residual: '+(p.max_normalized_feature_residual*100).toFixed(4)+'% of the defined amplitude. Human perceptual validation pending.';
}
async function runInstruction() {
  const ticket=++state.version,model=$('model').value;
  state.busy=true;$('run').disabled=true;status('Running frozen Model '+model+' on the local device…');
  try {
    const data=await api('/api/predict',{text:$('instruction').value,model});
    if(ticket!==state.version||state.mode!=='text')return;
    setMeshes(state.init.reference,data.vertices);showControls(data.prediction);
    $('resultLabel').textContent='Model '+model+' · geometric proxies';
    $('languageFact').textContent='Frozen Model '+model+(model==='A'?'':' · seed 13');
    status((data.prediction.status==='unsupported'?'Rule model did not support this wording; showing its zero fallback. ':'')+fitStatus(data.projection));
  } catch(error){if(ticket===state.version)status(error.message+' Last accepted mesh remains visible.',true);}
  finally {if(ticket===state.version){state.busy=false;$('run').disabled=false;}}
}
async function updateSemantic() {
  const ticket=++state.version,controls=labels.map((_,i)=>Number($('semantic'+i).value));
  controls.forEach((v,i)=>$('semanticValue'+i).textContent=(v>=0?'+':'')+v.toFixed(2));
  status('Fitting geometric controls…');
  try {
    const data=await api('/api/semantic',{controls});
    if(ticket!==state.version||state.mode!=='semantic')return;
    setMeshes(state.init.reference,data.vertices);status(fitStatus(data.projection));
  } catch(error){if(ticket===state.version)status(error.message+' Last accepted mesh remains visible.',true);}
}
async function updateBasis() {
  const ticket=++state.version,pc=Number($('pc').value),coefficient=Number($('coefficient').value),degrees=Number($('jaw').value);
  $('pcValue').textContent=pc;$('coefficientValue').textContent=coefficient.toFixed(2);$('jawValue').textContent=degrees.toFixed(1)+'°';
  try {
    const data=await api('/api/geometry',{pc_index:pc,coefficient,jaw_radians:degrees*Math.PI/180});
    if(ticket!==state.version||state.mode!=='lab')return;
    setMeshes(state.init.neutral,data.vertices);status('FLAME2019 decoded PC '+pc+' = '+coefficient.toFixed(2)+' and jaw = '+degrees.toFixed(1)+'°. Raw basis exploration; no semantic name is assigned.');
  } catch(error){if(ticket===state.version)status(error.message,true);}
}
function stopTour(){state.demo=false;$('demo').textContent='Play instruction tour';$('demo').setAttribute('aria-pressed','false');}
let debounce;
async function changeMode(mode) {
  stopTour();clearTimeout(debounce);state.mode=mode;state.version++;state.busy=false;$('run').disabled=false;
  for(const key of ['text','semantic','lab']) {
    $(key+'Panel').hidden=mode!==key;$(key+'Tab').classList.toggle('active',mode===key);$(key+'Tab').setAttribute('aria-selected',String(mode===key));
  }
  $('referenceLabel').textContent=mode==='lab'?'Neutral FLAME':'Open-jaw reference';
  $('resultLabel').textContent=mode==='lab'?'Raw expression + jaw':mode==='text'?'Language → geometric proxies':'Six direct geometric controls';
  $('viewTitle').textContent=mode==='lab'?'Neutral → FLAME basis':'Reference → six geometric controls';
  $('languageFact').textContent=mode==='text'?'Frozen Model '+$('model').value:'Not used in direct mode';$('demo').disabled=mode!=='text';
  const reference=mode==='lab'?state.init.neutral:state.init.reference;setMeshes(reference,reference);
  if(mode==='text')await runInstruction();else if(mode==='semantic')await updateSemantic();else await updateBasis();
}
labels.forEach((label,i)=>{
  const element=document.createElement('label');element.htmlFor='semantic'+i;element.textContent=i===3?'Lip-corner lift (smile proxy)':label;
  const value=document.createElement('output');value.id='semanticValue'+i;value.textContent='+0.00';element.append(value);
  const input=document.createElement('input');Object.assign(input,{id:'semantic'+i,type:'range',min:-1,max:1,step:.05,value:0});
  input.oninput=()=>{value.textContent=(Number(input.value)>=0?'+':'')+Number(input.value).toFixed(2);clearTimeout(debounce);debounce=setTimeout(updateSemantic,120);};
  $('semanticSliders').append(element,input);
});
$('resetSemantic').onclick=()=>{labels.forEach((_,i)=>$('semantic'+i).value=0);updateSemantic();};
$('run').onclick=()=>{stopTour();runInstruction();};$('model').onchange=runInstruction;
document.querySelectorAll('[data-prompt]').forEach(button=>button.onclick=()=>{stopTour();$('instruction').value=button.dataset.prompt;runInstruction();});
for(const mode of ['text','semantic','lab'])$(mode+'Tab').onclick=()=>changeMode(mode);
for(const id of ['pc','coefficient','jaw'])$(id).oninput=()=>{clearTimeout(debounce);debounce=setTimeout(updateBasis,100);};
$('resetBasis').onclick=()=>{for(const id of ['pc','coefficient','jaw'])$(id).value=0;updateBasis();};
$('animate').onclick=()=>{state.animate=!state.animate;$('animate').textContent=state.animate?'Pause animation':'Animate transition';$('animate').setAttribute('aria-pressed',String(state.animate));};
$('rotate').onclick=()=>{state.rotate=!state.rotate;$('rotate').textContent=state.rotate?'Stop rotation':'Auto rotate';$('rotate').setAttribute('aria-pressed',String(state.rotate));};
$('resetView').onclick=()=>{state.yaw=.17;state.pitch=0;state.zoom=1;};
$('demo').onclick=async()=>{
  if(state.demo){stopTour();return;}state.demo=true;$('demo').textContent='Stop instruction tour';$('demo').setAttribute('aria-pressed','true');
  const prompts=['Raise the eyebrows much.','Open the eyes much.','Widen the mouth much.','Increase the smile much.','Open the jaw much.','Raise the cheeks much.'];let i=0;
  while(state.demo){if(!state.busy){$('instruction').value=prompts[i++%prompts.length];await runInstruction();}await new Promise(resolve=>setTimeout(resolve,6500));}
};
async function start() {
  try {
    state.init=await api('/api/init');$('health').textContent='● Local '+state.init.device.toUpperCase();$('health').style.color='#7ce8d5';
    $('meshInfo').textContent=state.init.neutral.length.toLocaleString()+' vertices · '+state.init.faces.length.toLocaleString()+' faces';$('provenance').textContent='FLAME2019 SHA-256: '+state.init.flame_sha256;
    state.viewers=[new MeshView($('reference'),[.67,.73,.82]),new MeshView($('result'),[.48,.84,.78])];setMeshes(state.init.reference,state.init.reference);await runInstruction();let last=0;
    function frame(time){if(state.rotate)state.yaw+=(time-last)*.00014;last=time;const phase=state.animate?(1-Math.cos(time*.0015))/2:1;state.viewers[0].draw(0);state.viewers[1].draw(phase);$('phase').textContent=state.animate?'ANIMATING · '+Math.round(phase*100)+'%':'TARGET MESH';requestAnimationFrame(frame);}
    requestAnimationFrame(frame);
  } catch(error){status(error.message,true);$('health').textContent='Connection error';}
}
start();
