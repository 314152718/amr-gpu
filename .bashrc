function repeat() { 
    local times="$1"; 
    local sec="$2"; 
    shift; 
    shift; 
    local cmd="$@"; 

    for ((i = 1; i <= $times; i++ )); do 
       eval "$cmd";
       sleep $sec;
    done 
 }