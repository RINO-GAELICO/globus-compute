import pika

def on_open(connection):
    # Invoked when the connection is open
    pass

def on_close(connection, exception):
    # Invoked when the connection is closed
    connection.ioloop.stop()

# Create our connection object,
# passing in the on_open and on_close methods
connection = pika.SelectConnection(on_open_callback=on_open, on_close_callback=on_close)

try:
    # Loop so we can communicate with RabbitMQ
    connection.ioloop.start()
except KeyboardInterrupt:
    # Gracefully close the connection
    connection.close()
    # Loop until we're fully closed.
    # The on_close callback is required to stop the io loop
    connection.ioloop.start()
