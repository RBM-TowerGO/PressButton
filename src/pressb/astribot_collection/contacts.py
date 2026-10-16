"""Route each PhysX contact once instead of scanning every pair in every env."""
from collections import defaultdict
import weakref


class ContactSubscription:
    def __init__(self, router, index):
        self.router = weakref.ref(router)
        self.index = index

    def __del__(self):
        router = self.router()
        if router is not None:
            router.callbacks.pop(self.index, None)


class ContactRouter:
    def __init__(self, prefixes, decode):
        self.prefixes = {prefix:index for index,prefix in enumerate(prefixes)}
        self.decode = decode
        self.paths = {}
        self.owners = {}
        self.callbacks = {}

    def subscribe(self, index, callback):
        if index in self.callbacks:
            raise ValueError('Duplicate contact subscription for environment')
        self.callbacks[index] = callback
        return ContactSubscription(self, index)

    def dispatch(self, headers, data):
        grouped = defaultdict(list)
        for header in headers:
            if not header.num_contact_data:
                continue
            owners = set()
            for actor in (header.actor0, header.actor1):
                if actor not in self.paths:
                    path = str(self.decode(actor))
                    self.paths[actor] = path
                    # Paths are /World/envs/env_N/<body>. An environment can
                    # still observe an unexpected contact with another env.
                    prefix = '/'.join(path.split('/')[:4])
                    self.owners[actor] = self.prefixes.get(prefix)
                owner = self.owners[actor]
                if owner in self.callbacks:
                    owners.add(owner)
            for owner in owners:
                grouped[owner].append(header)
        for owner, local_headers in grouped.items():
            self.callbacks[owner](local_headers, data, self.paths)
