


from threading import Thread
import time

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.retry_wait_time = retry_wait_time
        self.marketplace = marketplace
        self.carts = carts

    def run(self):
        for cos in self.carts:
            cos_id = self.marketplace.new_cart()
            for produs in cos:
                if produs['type'] == 'add':
                    contor = 0
                    while contor < produs['quantity']:
                        adaugat = self.marketplace.add_to_cart(cos_id, produs['product'])
                        while adaugat == False:
                            adaugat = self.marketplace.add_to_cart(cos_id, produs['product'])
                            time.sleep(self.retry_wait_time)
                        contor += 1
                else:
                    contor = 0
                    while contor < produs['quantity']:
                        self.marketplace.remove_from_cart(cos_id, produs['product'])
                        contor += 1
            produse_cumparate = self.marketplace.place_order(cos_id)
            for produs_cumparat in produse_cumparate:
                print(f"{self.name} bought {produs_cumparat}")

from threading import Condition

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.producatori = dict()
        self.cosuri = dict()
        self.cond = Condition()
        self.producatori_id = []
        self.cosuri_id = []
        self.contor_producator = 1
        self.contor_cos = 1

    def register_producer(self):
        


        with self.cond:
            self.contor_producator = sum(self.producatori_id)
            self.contor_producator += 1
            self.producatori_id.append(self.contor_producator)
            producator = dict()


            producator['produse'] = []
            self.producatori[self.contor_producator] = producator
            return self.contor_producator


    def publish(self, producer_id, product):
        
        with self.cond:
            for producator_id, lista_produse_publicate in self.producatori.items():
                if producator_id == producer_id:
                    lista_produse_publicate['produse'].append(product)
                    return True
            return False

    def new_cart(self):
        


        with self.cond:
            self.contor_cos = sum(self.cosuri_id)
            self.contor_cos += 2
            self.cosuri_id.append(self.contor_cos)
            cos = dict()
            cos['produse_rezervate'] = []
            self.cosuri[self.contor_cos] = cos
            return self.contor_cos


    def add_to_cart(self, cart_id, product):
        


        with self.cond:
            for cos, continut in self.cosuri.items():
                if cos == cart_id:
                    for producator, produse_publicate in self.producatori.items():
                        if product in produse_publicate['produse']:
                            continut['produse_rezervate'].append(product)
                            return True
            return False

    def remove_from_cart(self, cart_id, product):
        
        with self.cond:
            for cos, continut in self.cosuri.items():
                if cos == cart_id:
                    continut['produse_rezervate'].remove(product)

    def place_order(self, cart_id):
        
        with self.cond:
            for cos, continut in self.cosuri.items():
                if cos == cart_id:


                    return continut['produse_rezervate']
            return None


from threading import Thread
import time

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.kwargs = kwargs
        self.republish_wait_time = republish_wait_time
        self.marketplace = marketplace
        self.products = products
        self.daemon = True

    def run(self):
        producator_id = self.marketplace.register_producer()
        while True:
            for produs in self.products:
                contor = 0
                while contor < produs[1]:
                    in_market = self.marketplace.publish(producator_id, produs[0])
                    time.sleep(produs[2])
                    while in_market == False:
                        in_market = self.marketplace.publish(producator_id, produs[0])
                        time.sleep(self.republish_wait_time)
                    contor += 1


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    
    acidity: str
    roast_level: str
