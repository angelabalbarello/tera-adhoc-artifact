#!/usr/bin/env python3
"""Figura final para o artigo (v2): eixo X = Risk-transition duration d (s),
fontes maiores (padrao R2.6), sem suptitle."""
import csv, statistics as st
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.labelsize':12,
                     'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':9.5})
DURS=[2,6,10,14,18,22,31,42,50]; DT=10.0/96; DS=np.array([d*DT for d in DURS])
ID1=(0.42,1.04); GAP=(1.15,2.40); ID2=(2.50,4.06)
A=list(csv.DictReader(open('duration_only/frozen_results.csv')))
for r in A:
    for k in ('FR','TTDef','FSonset','ComputeRate'): r[k]=float(r[k])
    r['dur_frames']=int(r['dur_frames'])
B=list(csv.DictReader(open('iso_cost/iso_cost_results.csv')))
for r in B:
    for k in ('compute_rate','FSonset','FR'): r[k]=float(r[k])
    r['dur_frames']=int(r['dur_frames'])
def serA(c,k): 
    return (np.array([st.mean([r[k] for r in A if r['dur_frames']==d and r['config']==c]) for d in DURS]),
            np.array([st.stdev([r[k] for r in A if r['dur_frames']==d and r['config']==c]) for d in DURS]))
def serB(m,k):
    return (np.array([st.mean([r[k] for r in B if r['dur_frames']==d and r['method'].startswith(m)]) for d in DURS]),
            np.array([st.stdev([r[k] for r in B if r['dur_frames']==d and r['method'].startswith(m)]) for d in DURS]))
COLORS={'Baseline Fixed':'#888888','Baseline+MHEG':'#d62728','AF-TOI Fixed':'#1f77b4','AF-TOI+MHEG':'#2ca02c'}
def shade(ax):
    ax.axvspan(ID1[0],ID1[1],color='0.88',alpha=.8,zorder=0)
    ax.axvspan(ID2[0],ID2[1],color='0.88',alpha=.8,zorder=0)
    ax.set_xlim(0.1,5.45); ax.set_xlabel('Risk-transition duration $d$ (s)')
fig,axes=plt.subplots(2,2,figsize=(11.5,8.4))
ax=axes[0,0]
for c in COLORS:
    m,s=serA(c,'FR'); ax.errorbar(DS,m,yerr=s/np.sqrt(12),marker='o',ms=4,label=c,color=COLORS[c],lw=1.4)
ax.axhline(0.05,ls='--',c='k',lw=.9); ax.text(5.35,0.055,'FR$=$0.05',ha='right',fontsize=9)
shade(ax); ax.set_ylabel('FR'); ax.set_title('(a) FR, frozen operating points'); ax.legend()
ax=axes[0,1]
for c in COLORS:
    m,s=serA(c,'TTDef'); ax.errorbar(DS,m,yerr=s/np.sqrt(12),marker='o',ms=4,color=COLORS[c],lw=1.4)
shade(ax); ax.set_ylabel('TTD$_{\\mathrm{ef}}$ (s)'); ax.set_title('(b) TTD$_{\\mathrm{ef}}$, frozen operating points')
ax=axes[1,0]
for m_,lab,col in [('AF-TOI','AF-TOI+MHEG','#2ca02c'),('Baseline','Baseline+MHEG (matched budget)','#d62728')]:
    mm,ss=serB(m_,'FSonset'); ax.errorbar(DS,mm,yerr=ss/np.sqrt(12),marker='o',ms=4,label=lab,color=col,lw=1.4)
shade(ax); ax.set_ylabel('FS@onset'); ax.set_title('(c) FS@onset, matched computation cost'); ax.legend()
ax=axes[1,1]
for m_,lab,col in [('AF-TOI','AF-TOI+MHEG','#2ca02c'),('Baseline','Baseline+MHEG (matched budget)','#d62728')]:
    mm,ss=serB(m_,'compute_rate'); ax.errorbar(DS,mm,yerr=ss/np.sqrt(12),marker='o',ms=4,label=lab,color=col,lw=1.4)
ax2=ax.twinx()
for m_,col in [('AF-TOI','#2ca02c'),('Baseline','#d62728')]:
    mm,_=serB(m_,'FR'); ax2.plot(DS,mm,ls=':',color=col,lw=1.3)
ax2.axhline(0.05,ls='--',c='k',lw=.7); ax2.set_ylabel('FR (dotted)',fontsize=10)
shade(ax); ax.set_ylabel('Compute rate (solid)'); ax.set_title('(d) Budget matching and FR at matched cost'); ax.legend(loc='lower left')
fig.tight_layout()
fig.savefig('figures/fig_temporal_scale_v2.pdf'); fig.savefig('figures/fig_temporal_scale_v2.png',dpi=150)
print('ok')
