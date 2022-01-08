
python server_untrusted.py &
SERVER_PID=$!
sleep 1
python client.py

sleep 1
kill $!
