export function groupBy(items,keyFn){
const out=new Map()
for (const it of items){
 const k=keyFn(it)
 if(!out.has(k)) out.set(k,[])
 out.get(k).push(it)
}
return out
}
