#!/usr/bin/python3
class Entity:
    def __init__(self, name, active=True, cardinality=None, count=0):
        self.active = active
        self.name = name
        self.cardinality = cardinality
        self.count = count


class TimeStamp:
    __time_stap = 0

    def __init__(self):
        self.time_stamp = TimeStamp.__time_stap
        TimeStamp.__time_stap += 1


class Information(TimeStamp):
    def __init__(self):
        super()
        self.entities = []
        self.connections = []


class State(Information):
    def __int__(self, name, timestamp=None):
        super()


class Action(Information):
    def __init__(self, name, timestamp=None):
        super()
