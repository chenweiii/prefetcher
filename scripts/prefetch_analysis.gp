set title "Prefetch distance analysis"
set xlabel "Prefetch Distance"
set ylabel "Execution time(us)"
set terminal png font " Times_New_Roman,12 "
set output "../statistic.png"
set xtics 1 ,1 ,32
set key left 

plot \
"../result.txt" using 1:2 with linespoints linewidth 2 title "SSE", \
"../result.txt" using 1:3 with linespoints linewidth 2 title "AVX", \
