from types import SimpleNamespace
from pressb.astribot_collection.contacts import ContactRouter


def test_route_once_preserves_foreign_contacts_and_offsets():
    paths = {1:'/World/envs/env_0/robot/hand',2:'/World/envs/env_0/button',
             3:'/World/envs/env_1/robot/hand',4:'/World/floor'}
    calls = []
    router = ContactRouter(['/World/envs/env_0','/World/envs/env_1'], lambda actor:paths[actor])
    subscriptions = [router.subscribe(i,lambda headers,data,cache,i=i:calls.append((i,headers,data,cache))) for i in range(2)]
    own = SimpleNamespace(actor0=1,actor1=2,num_contact_data=2,contact_data_offset=7)
    foreign = SimpleNamespace(actor0=1,actor1=3,num_contact_data=1,contact_data_offset=9)
    floor = SimpleNamespace(actor0=3,actor1=4,num_contact_data=1,contact_data_offset=10)
    data = object()
    router.dispatch([own,foreign,floor],data)
    assert calls[0][0:3] == (0,[own,foreign],data)
    assert calls[1][0:3] == (1,[foreign,floor],data)
    assert calls[0][3] == paths
    subscriptions.pop()
    calls.clear()
    router.dispatch([own,foreign,floor],data)
    assert len(calls) == 1 and calls[0][0:3] == (0,[own,foreign],data)
