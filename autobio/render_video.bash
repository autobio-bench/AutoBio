log_root=$1
if [ -z "$log_root" ]; then
    echo "Usage: $0 <log_root>"
    exit 1
fi
if [ ! -d "$log_root" ]; then
    echo "Directory $log_root does not exist."
    exit 1
fi
for dir in $log_root/*; do
    if [ ! -d "$dir" ]; then
        continue
    fi
    blender_dir=$dir/blender
    if [ ! -d "$blender_dir" ]; then
        echo "Directory $blender_dir does not exist."
        continue
    fi
    # for each subdir in $blender_dir/*; do
    for image_dir in $blender_dir/*; do
        if [ ! -d "$image_dir" ]; then
            continue
        fi
        image_dir_name=$(basename $image_dir)
        ffmpeg -framerate 50 -i $image_dir/%04d.png -vf "format=rgba,premultiply=inplace=1" -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p $blender_dir/$image_dir_name.mp4
    done
done
