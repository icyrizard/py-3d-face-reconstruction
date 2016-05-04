files=`ls data/imm_face_db/ | grep -E "^[0-3][0-9].*.asf"`

for f in $files; do
    echo "data/imm_face_db/$f"
done
