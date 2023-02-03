python3 mnistsrc_train.py > ./models/mnistsrc_train_out.txt &&
python3 mnistsrc_mnisttgt_train.py > ./models/mnistsrc_mnisttgt_train_out.txt &&
python3 mnistsrc_imdbtgt_train.py > ./models/mnistsrc_imdbtgt_train_out.txt &&
python3 imdbsrc_train.py > ./models/imdbsrc_train_out.txt &&
python3 imdbsrc_imdbtgt_train.py > ./models/imdbsrc_imdbtgt_train_out.txt &&
python3 imdbsrc_mnisttgt_train.py > ./models/imdbsrc_mnisttgt_train_out.txt;