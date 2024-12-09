res_dir=$1
anno_dir=$2

shopt -s nullglob
sub_sequences=( $(ls "$2/") )
#echo $sub_sequences
for subseq in "${sub_sequences[@]}"
do
   filename=$(basename $subseq)
   cameras=($1/$filename-cam-*)
   IFS=$'\n' cameras=($(sort <<<"${cameras[*]}")); unset IFS
   last_cam=${cameras[-1]}
   last_cam_name=$(basename $last_cam)
   last_cam_num=${last_cam_name:20:4}
   new_cam_num=$(printf "%05d" $(($(echo $last_cam_num | sed 's/^0*//') + 1)) )

   echo "copy from: $last_cam"
   new_dir=$(dirname "$last_cam")
   #echo $last_cam
   #echo $new_dir
   echo "new: $new_dir/$subseq-cam-$new_cam_num.txt"

   cp $last_cam "$new_dir/$subseq-cam-$new_cam_num.txt"
done