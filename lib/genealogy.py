# -*- coding: utf-8 -*-
import uuid


class Genealogy(object):
    """
    Directed tree.
    """

    def __init__(self, parents=[]):
        object.__init__(self)
        self._parents = parents
        self._children = []
        self.id = uuid.uuid1()

    def name_dict(self):
        """
        returns a dict as -as_dict- however
        """

    def as_dict(self):
        dict = {self: self.get_children()}
        for child in self.get_children():
            dict.update(child.as_dict())
        return dict

    def from_dict(self, dict):
        for child in dict[self]:
            child.link(self)
            child.from_dict(dict)

    def is_root(self):
        return len(self._parents) == 0

    def get_parents(self):
        return self._parents

    def get_children(self):
        return list(self._children)

    def get_siblings(self):
        if self.is_root():
            return []
        siblings = [p.get_children() for p in self.get_parents()]
        siblings.remove(self)
        return siblings

    def get_ancestors(self, n_generations=1):
        if self.is_root() or n_generations < 1: return []
        parents = self.get_parents()
        ancestors = [parents]
        for p in parents:
            ancestors += p.get_ancestors(n_generations - 1)
        return ancestors

    def get_descendants(self, n_generations=None):
        descendants = [self]
        if n_generations is None:
            for child in self.get_children():
                descendants += child.get_descendants()
            return descendants
        else:
            if n_generations > 0:
                for child in self.get_children():
                    descendants += child.get_descendants(n_generations=n_generations - 1)
            return descendants

    def is_child(self, group):
        return group in self.get_children()

    def has_children(self):
        return len(self.get_children()) > 0

    def link(self, parents):
        self._p_changed = 1
        if type(parents) != list:
            parents = [parents]
        elif parents is None:
            return
        self._parents = parents
        for parent in parents:
            parent._children.append(self)

    def unlink(self):
        self._p_changed = 1
        if not self.is_root():
            for parent in self._parents:
                parent._children.remove(self)

    def trace_back(self):
        trace = [self]
        generation = self
        while not generation.is_root():
            generation = generation.get_parents()
            trace.insert(0, generation)
        return trace

    def _new_id(self):
        self._p_changed = 1
        self.id = uuid.uuid1()

    def __iter__(self):
        # http://stackoverflow.com/questions/5434400/python-is-it-possible-to-make-a-class-iterable-using-the-standard-syntax
        for each in self.get_children():
            yield each
        #for each in self.__dict__.keys():
        #    yield self.__getattribute__(each)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.get_children()[key]
        else:
            start = 0 if key.start is None else key.start
            stop = None if key.stop is None else key.stop
            step = 1 if key.step is None else key.step
            return self.get_children()[start:stop:step]


