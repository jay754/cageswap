# cageswap

curl -X POST -F "target_image=@/Users/jayengineer/Workspace/cage-faceswap/copy.jpg" -F "cage_image=@/Users/jayengineer/Workspace/cage-faceswap/nicolas-cage.jpg" http://127.0.0.1:5000/swap_faces --output swapped_face.jpg