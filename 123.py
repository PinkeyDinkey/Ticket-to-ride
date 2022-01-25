import pysimilar as ps
ps.extensions = '.py'
print(ps.compare('ticket_to_ride.py', 'ticket_to_ride (2).py', isfile=True))