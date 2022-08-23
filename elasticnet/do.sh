let "ci = 1";
while [ $ci -le 10 ]; do
  python main_sac.py --episodes 1000 --steps 10 --seed $ci > "nohint_"$ci".txt"
  python main_sac.py --episodes 1000 --steps 10 --seed $ci --use_hint > "hint_"$ci".txt"
  let "ci = $ci + 1";
done
