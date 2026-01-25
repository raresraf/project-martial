

import time

from threading import Thread


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        
        Thread.__init__(self, kwargs=kwargs)
        self.name = kwargs['name']
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.carts_list =[]

    def afisare(self, lista):
        for item in lista:
            print(f'{self.name} bought {item}')

    def run(self):
        
        
        
        for item in self.carts:
            id_cart = self.marketplace.new_cart()
            self.carts_list.append(id_cart)
            for comanda in item:
                if comanda['type'] == 'add':
                    for _ in range(comanda['quantity']):
                        while self.marketplace.add_to_cart(id_cart, comanda['product']) is False:
                            time.sleep(self.retry_wait_time)
                if comanda['type'] == 'remove':
                    for _ in range(comanda['quantity']):
                        self.marketplace.remove_from_cart(
                            id_cart, comanda['product'])
        for cart in self.carts_list:
            self.afisare(self.marketplace.place_order(cart))

from threading import Semaphore


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        
        self.queue_size_per_producer = queue_size_per_producer
        self.id_producer = 0
        self.id_cart = 0
        
        


        self.database = {}

        
        
        self.carts = {}

        
        self.sem_cart = Semaphore(1)
        self.sem_cons = Semaphore(1)
        self.sem_remove = Semaphore(1)
        self.sem_register = Semaphore(1)
        self.sem_place = Semaphore(1)

    def register_producer(self):
        
        
        
        self.sem_register.acquire()
        id_producer_act = self.id_producer+1
        self.id_producer = self.id_producer+1
        self.database[f'id{str(id_producer_act)}'] = []
        self.sem_register.release()
        return f'id{str(id_producer_act)}'

    def publish(self, producer_id, product):
        
        
        if len(self.database[producer_id]) < self.queue_size_per_producer:
            self.database[producer_id].append(product)
            return True

        return False

    def new_cart(self):
        
        

        self.sem_cart.acquire()
        id_cart_actual = self.id_cart
        self.id_cart = self.id_cart+1
        self.carts[id_cart_actual] = []
        self.sem_cart.release()
        return id_cart_actual

    def add_to_cart(self, cart_id, product):
        
        
        self.sem_cons.acquire()
        for producator in self.database:
            for item in self.database[producator]:


                if item == product:
                    self.carts[cart_id].append((item, producator))
                    self.database[producator].remove(item)
                    self.sem_cons.release()
                    return True
        self.sem_cons.release()
        return False

    def remove_from_cart(self, cart_id, product):
        
        self.sem_remove.acquire()
        for produs, producator in self.carts[cart_id]:
            if produs == product:
                self.database[producator].append(produs)
                self.carts[cart_id].remove((produs, producator))
                self.sem_remove.release()
                break

    def place_order(self, cart_id):
        
        
        self.sem_place.acquire()
        result = []
        for produs in self.carts[cart_id]:
            result.append(produs[0])
        self.sem_place.release()
        return result


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        
        Thread.__init__(self, kwargs=kwargs)

        self.name = kwargs['name']
        self.daemon = kwargs['daemon']
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.products = products

    def run(self):
        
        id_prod = self.marketplace.register_producer()
        while True:
            for (produs, cantitate, timp) in self.products:
                for _ in range(cantitate):
                    while self.marketplace.publish(id_prod, produs) is False:
                        time.sleep(self.republish_wait_time)
                    time.sleep(timp)


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
