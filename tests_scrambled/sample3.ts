export type Result<T,E>=
|{ ok:true; value:T}
|{ ok:false; error:E }

export function ok<T,E=never>(value:T):Result<T,E>{
return {ok:true,value}
}

export function err<T=never,E=unknown>(error:E):Result<T,E>{
 return {ok:false,error}
}
