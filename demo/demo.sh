echo "<h1>Comparación de Imágenes</h1>"

for archivo1 in `ls *.jpeg *.png`
do
for archivo2 in `ls *.jpeg *.png`
do

echo "<h2>$archivo1 vs $archivo2</h2>"
echo "<table><tr>"

echo "<td><img src='$archivo1' ></td>"
echo "<td><img src='$archivo2' ></td>"
echo "</tr></table>"
echo "<h3>"
python3 ../src/main.py $archivo1 $archivo2
echo "</h3>"
echo "<hr>"

done
done
