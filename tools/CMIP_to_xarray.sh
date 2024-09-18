####
# Hopefully you made sure your restart files have been made modifiable ( chmod 666 should be fine)
#
####

for i in "$@"; do
  	ncrename -v y,yy $i
	ncrename -v x,xx $i
done
