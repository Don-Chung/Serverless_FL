STRING1="If you want to modify the parameters, please modify it in file para.ini"
echo $STRING1
STRING2="Start creating users..."
echo $STRING2
eval python Create_Users.py
STRING3="Start Running..."
echo $STRING3
eval python Users.py