# transform is like: {cell width} {zero} {xmin} {zero} {cell height} {ymin}
SNAIL_PROGRESS=1 snail split \
    --features GB_road_link_file.pq \
    --transform 10000 0 -100000 0 10000 -100000 \
    --width 100 \
    --height 150 \
    --output GB_road_link_file_chunks.gpkg

# make directory for python script to write parquet chunks to
mkdir -p GB_road_link_file_100k.parquet
