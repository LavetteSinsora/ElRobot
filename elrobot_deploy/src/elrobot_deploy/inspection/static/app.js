const $ = id => document.getElementById(id);
const DEG = 180 / Math.PI;
const format = (x, digits=2) => x === null || x === undefined ? '—' : Number(x).toFixed(digits);
let data = null, id = '', time = 0, playing = false, previousTime = 0, generation = 0;
let pending = false, busy = false, scene = null, currentSample = null;
const cameras = new Map();
const initial = new URLSearchParams(location.search);

async function json(url) {
  const r = await fetch(url), x = await r.json();
  if (!r.ok) throw new Error(x.error || `HTTP ${r.status}`);
  return x;
}
const api = (path, extra={}) => `/api/${path}?${new URLSearchParams({id,...extra})}`;
function error(e) { $('error').textContent = e.message || e; $('error').hidden = false; }
function permalink() { if(data) history.replaceState(null,'',`?${new URLSearchParams({rec:id,t:time.toFixed(3)})}`); }
function stop() { playing = false; $('play').textContent = 'Play'; permalink(); }
function options(select, entries) { select.replaceChildren(...entries.map(([value,text]) => new Option(text,value))); }

for (let i=1;i<=8;i++) $('joint').add(new Option(`Joint ${i}${i===8?' · gripper':''}`,i-1));
$('joint').value = '5';
for (let i=0;i<8;i++) {
  const tr = document.createElement('tr'); tr.dataset.joint = i;
  for (let j=0;j<5;j++) tr.appendChild(document.createElement('td'));
  tr.children[0].textContent = `${i+1}${i===7?' · grip':''}`;
  tr.addEventListener('click',()=>{ $('joint').value=i; charts(); }); $('joints').appendChild(tr);
}

async function refreshRecords() {
  const {records} = await json('/api/records');
  options($('record'),records.map(r=>[r.id,`${r.created?.replace('T',' ') || r.name}${r.complete?'':' · incomplete'}`]));
  if (!records.length) throw new Error('No ElRobot recordings were found in this directory.');
  const requested = initial.get('rec');
  $('record').value = records.some(r=>r.id===requested) ? requested : records[0].id;
  await loadRecord(true);
}

async function loadRecord(first=false) {
  stop(); data = null; currentSample = null; id = $('record').value;
  const token = ++generation; $('error').hidden = true; $('play').disabled = true;
  $('task').textContent = 'Loading recording…'; $('cameras').replaceChildren(); cameras.clear();
  if (scene) scene.update(null);
  try {
    const loaded = await json(api('record'));
    if (token !== generation) return;
    data = loaded;
    $('task').textContent = data.task;
    $('meta').textContent = `${data.created?.replace('T',' ') || data.name} · ${format(data.duration,1)} s · ${data.state_count.toLocaleString()} state samples · ${data.complete?'recording complete':'incomplete recording'}`;
    $('warnings').textContent = data.warnings.join(' '); $('warnings').hidden = !data.warnings.length;
    $('record-info').textContent = JSON.stringify({path:data.path,created:data.created,ended:data.ended,
      recording_complete:data.complete,task_verdict:data.finish,code:data.code,cameras:data.cameras},null,2);
    $('scrub').max = data.duration; $('play').disabled = false;
    for (const cam of data.cameras) {
      const pane = document.createElement('div'); pane.className='card camera-pane';
      const head=document.createElement('div');head.className='card-head';
      const title=document.createElement('h2');title.textContent=cam.name==='exo'?'iPhone · exo':cam.name==='wrist'?'Wrist · fisheye':cam.name;
      const count=document.createElement('span');count.className='muted';count.textContent=`${cam.frames.toLocaleString()} frames`;
      head.append(title,count);
      const figure=document.createElement('figure'),img=document.createElement('img'),caption=document.createElement('figcaption');
      img.alt=`Recorded ${cam.name} camera`;img.hidden=!cam.frames;caption.textContent=cam.frames?'Loading frame…':'No frames recorded';
      img.addEventListener('error',()=>{caption.textContent='Frame unavailable';});
      figure.append(img,caption);
      const meta=document.createElement('div');meta.className='camera-meta';meta.textContent=`${cam.sensor?.join(' × ') || ''} · ${cam.dropped} dropped frames`;
      pane.append(head,figure,meta);$('cameras').appendChild(pane);cameras.set(cam.name,{img,caption,meta,index:-1,spec:cam});
    }
    options($('action'),[['','Between actions'],...data.actions.map(a=>[a.action,`Action ${a.action}${a.interrupted?' · interrupted':''}`])]);
    $('action-strip').replaceChildren();
    for (const a of data.actions) {
      const b=document.createElement('button');b.className=`action-marker${a.interrupted?' interrupted':''}`;b.textContent=a.action;
      b.style.left=`${100*a.start/Math.max(data.duration,.001)}%`;b.style.width=`${100*(a.end-a.start)/Math.max(data.duration,.001)}%`;
      b.title=`Action ${a.action} · ${format(a.start)}–${format(a.end)} s`;
      b.addEventListener('click',()=>{stop();seek(a.start);});$('action-strip').appendChild(b);
    }
    $('events').replaceChildren();
    for (const event of data.events) {
      const row=document.createElement('div');row.className='event';
      const button=document.createElement('button');button.textContent=`${format(event.t,3)} s`;
      button.addEventListener('click',()=>{stop();seek(event.t);});
      const text=document.createElement('span');text.textContent=`${event.session} / ${event.bus}`;
      if (['RECOVERING','HOLDING'].includes(event.bus)) text.className='bad';
      row.append(button,text);$('events').appendChild(row);
    }
    const robot = await json(api('robot'));
    if (token !== generation) return;
    if (scene) scene.load(robot,id).then(()=>{ if (token===generation && currentSample) scene.update(currentSample); });
    $('model-status').textContent=robot?'URDF at recorded path':'Model unavailable';
    const start = Math.max(0,data.state.t[0] || 0);
    seek(first && initial.has('t') ? Number(initial.get('t')) : start);
  } catch(e) { if(token===generation) {error(e);$('task').textContent='Unable to open recording';} }
}

function seek(t) {
  if (!data) return;
  time = Math.min(data.duration,Math.max(0,Number.isFinite(t)?t:0));
  if(!playing) permalink();
  $('scrub').value=time;$('time').textContent=`${format(time,3)} / ${format(data.duration,3)} s`;
  actionInfo(); charts(); pending=true; samples();
}
async function samples() {
  if (busy) return; busy=true;
  try {
    while (pending && data) {
      pending=false;const token=generation, t=time;
      const sample=await json(api('sample',{t}));
      if(token!==generation || t!==time) continue;
      currentSample=sample;if(scene) scene.update(sample);
      $('session').textContent=`${sample.session || '—'} / ${sample.bus || '—'}`;
      $('residual').textContent=`Position ${format(sample.position_mm)} mm · Orientation ${format(sample.orientation_deg)}°`;
      Array.from($('joints').children).forEach((tr,i)=>{
        const q=sample.measured?.[i],c=sample.command?.[i],e=q==null||c==null?null:(q-c)*DEG;
        tr.children[1].textContent=format(q==null?null:q*DEG,1);
        tr.children[2].textContent=format(c==null?null:c*DEG,1);tr.children[3].textContent=format(e,1);
        tr.children[3].className=Math.abs(e)>15?'bad':'';tr.children[4].textContent=format(sample.load?.[i],3);
      });
      for (const [name,view] of cameras) {
        const frame=sample.cameras[name];if(!frame) continue;
        if(view.index!==frame.index) {view.index=frame.index;view.img.src=api('frame',{camera:name,index:frame.index});}
        view.caption.textContent=`${format(frame.t,3)} s`;
        view.meta.textContent=`frame ${frame.index+1} / ${view.spec.frames} · offset ${format(frame.delta_ms,1)} ms · ${view.spec.dropped} dropped`;
      }
    }
  } catch(e) {error(e);stop();} finally {busy=false;}
}

function actionInfo() {
  const active=data.actions.find(a=>time>=a.start && time<=a.end);
  $('action').value=active?.action ?? '';
  Array.from($('action-strip').children).forEach((b,i)=>b.classList.toggle('active',data.actions[i]===active));
  const a=active || data.actions.filter(a=>a.end<time).at(-1);
  if(!a) {$('action-info').textContent=data.actions.length?'Waiting for the first action.':data.episode_available?'This episode has no recorded actions.':'No Astra episode is associated with this recording.';$('action-json').textContent='—';return;}
  $('action-info').textContent=`${active?'Executing':'After'} action ${a.action} · ${a.speed}\nPlanned ${format(a.planned_duration_s)} s · Recorded execution ${format(a.duration_s)} s${a.interrupted?`\nInterrupted: ${a.interrupted}`:''}`;
  $('action-info').classList.toggle('bad',!!a.interrupted);
  $('action-json').textContent=JSON.stringify({request:a.request,outcome:{...a,request:undefined}},null,2);
}

function bounds() {
  if($('window').value==='all') return [0,data.duration];
  const width=Number($('window').value),lo=Math.max(0,Math.min(time-width/2,data.duration-width));
  return [lo,Math.min(data.duration,lo+width)];
}
function chart(canvas,series) {
  const rect=canvas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio,2),w=rect.width,h=rect.height;
  canvas.width=w*ratio;canvas.height=h*ratio;const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);
  const [lo,hi]=bounds(),left=48,right=w-12,top=12,bottom=h-24;
  const within=series.map(s=>({...s,points:s.t.map((t,i)=>[t,s.y[i]]).filter(([t,y])=>t>=lo&&t<=hi&&y!==null&&Number.isFinite(y))}));
  let min=Infinity,max=-Infinity;
  for(const s of within) for(const [,y] of s.points){min=Math.min(min,y);max=Math.max(max,y);}
  if(!Number.isFinite(min)){min=0;max=1;}const pad=Math.max((max-min)*.12,.2);min-=pad;max+=pad;
  const X=t=>left+(t-lo)/(hi-lo||1)*(right-left),Y=y=>bottom-(y-min)/(max-min)*(bottom-top);
  ctx.font='10px ui-monospace, monospace';ctx.lineWidth=1;
  for(let i=0;i<=4;i++){const y=min+(max-min)*i/4,yp=Y(y);ctx.strokeStyle='#293640';ctx.beginPath();ctx.moveTo(left,yp);ctx.lineTo(right,yp);ctx.stroke();ctx.fillStyle='#91a3b0';ctx.textAlign='right';ctx.fillText(y.toFixed(1),left-7,yp+3);}
  for(let i=0;i<=4;i++){const t=lo+(hi-lo)*i/4;ctx.textAlign='center';ctx.fillText(t.toFixed(1),X(t),h-5);}
  for(const s of within){ctx.strokeStyle=s.color;ctx.lineWidth=1.6;ctx.beginPath();s.points.forEach(([t,y],i)=>i?ctx.lineTo(X(t),Y(y)):ctx.moveTo(X(t),Y(y)));ctx.stroke();}
  ctx.strokeStyle='#68ddc6';ctx.setLineDash([4,3]);ctx.beginPath();ctx.moveTo(X(time),top);ctx.lineTo(X(time),bottom);ctx.stroke();ctx.setLineDash([]);
  canvas.onclick=event=>{const f=(event.clientX-rect.left-left)/(right-left);stop();seek(lo+Math.max(0,Math.min(1,f))*(hi-lo));};
}
function charts() {
  if(!data) return;const j=Number($('joint').value);
  Array.from($('joints').children).forEach((tr,i)=>tr.classList.toggle('selected',i===j));
  chart($('joint-chart'),[{t:data.state.t,y:data.state.q.map(q=>q[j]),color:'#efad5e'},
    {t:data.state.t,y:data.state.cmd.map(q=>q[j]),color:'#6da8ff'},
    {t:data.targets.t,y:data.targets.q.map(q=>q[j]),color:'#b99afb'}]);
  chart($('error-chart'),[{t:data.targets.t,y:data.targets[$('metric').value],color:'#ec7a9d'}]);
}
$('record').addEventListener('change',()=>loadRecord());$('reload').addEventListener('click',()=>loadRecord());
$('scrub').addEventListener('input',e=>{stop();seek(Number(e.target.value));});
$('play').addEventListener('click',()=>{if(!data)return;if(time>=data.duration)seek(0);playing=!playing;$('play').textContent=playing?'Pause':'Play';previousTime=performance.now();});
$('back').addEventListener('click',()=>{stop();seek(time-1/30);});$('forward').addEventListener('click',()=>{stop();seek(time+1/30);});
$('end').addEventListener('click',()=>{stop();$('window').value='10';seek(data?data.duration-3:0);});
$('action').addEventListener('change',()=>{const a=data?.actions.find(x=>String(x.action)===$('action').value);if(a){stop();seek(a.start);}});
for(const name of ['joint','window','metric']) $(name).addEventListener('change',charts);
window.addEventListener('resize',charts);
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA','BUTTON'].includes(document.activeElement.tagName))return;if(e.code==='Space'){e.preventDefault();$('play').click();}if(e.code==='ArrowRight')$('forward').click();if(e.code==='ArrowLeft')$('back').click();});
setInterval(()=>{const now=performance.now();if(playing&&data){seek(time+(now-previousTime)/1000*Number($('speed').value));if(time>=data.duration)stop();}previousTime=now;},1000/15);
try {const {createScene}=await import('./scene.js');scene=createScene($('scene'),$('scene-message'));}
catch(e){$('scene-message').textContent=`3D unavailable: ${e.message}`;}
refreshRecords().catch(error);
