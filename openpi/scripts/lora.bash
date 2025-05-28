cd assets
for dir in *; do
    if [ -d "$dir" ] && [[ ! "$dir" =~ -lora$ ]]; then
        ln -s "$dir" "${dir}-lora"
    fi
done
