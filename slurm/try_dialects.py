import yaml, time
from dlsd.dialect_rules import transform
cfg=yaml.safe_load(open("configs/experiment.yaml"))
s="The patient has been taking 500 mg of metformin twice daily, and he did not tell the doctor about his chest pain."
for d in cfg["conditions"]["dialects"]:
    t=time.time(); r=transform(s,d,cfg); print("RESULT",d,"|",r.text,"|",r.rules_fired,r.changed,round(time.time()-t,1),"s",flush=True)
a=transform(s,"Indian English",cfg); b=transform(s,"Indian English",cfg); print("RESULT deterministic:",a.text==b.text)
