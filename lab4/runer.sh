# sudo mount -t nfs -o rw,async,noatime,nodiratime,vers=3,wsize=1048576,rsize=1048576,noacl,nocto,timeo=600 nfs.coldstore:/speechpro/nid/workdata/ITMO_voice_recognition /mnt/storage


git pull origin main
jupyter nbconvert lab4_blank.ipynb --to script
ipython3 lab4_blank.py 


