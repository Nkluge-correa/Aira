from codecarbon import EmissionsTracker

tracker = EmissionsTracker()


tracker.start()

for i in range(100_000_000):
    pass

tracker.stop()