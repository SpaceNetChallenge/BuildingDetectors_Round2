for var in "$@"
do
    ./train_one_city_a.sh "$var"
    ./train_one_city_b.sh "$var"
done
