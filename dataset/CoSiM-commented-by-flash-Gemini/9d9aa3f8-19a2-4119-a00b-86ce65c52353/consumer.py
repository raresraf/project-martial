


from threading import Lock, Thread
import time


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs
        self.lock = Lock()

    def run(self):
        
        for i in self.carts:
            id_cart = self.marketplace.new_cart()
            
            for operation in i:
                operation_counter = 0
                
                while operation_counter < operation["quantity"]:
                    
                    with self.lock:
                        if operation["type"] == "add":
                            ret = self.marketplace.add_to_cart(id_cart, operation["product"])
                            if not ret:
                                time.sleep(self.retry_wait_time)
                            else:
                                operation_counter += 1

                        else:
                            self.marketplace.remove_from_cart(id_cart, operation["product"])
                            operation_counter += 1
            
            for product in self.marketplace.place_order(id_cart):
                print("%s bought %s" %(self.kwargs['name'], product))


import collections
from threading import Lock

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.id_producer = 0
        self.id_cart = 0

        
        self.products_lists = collections.defaultdict(list)

        
        self.carts_lists = collections.defaultdict(list)

        
        self.bought_items = collections.defaultdict(list)

        self.lock_carts = Lock()
        self.lock_producers = Lock()


    def register_producer(self):
        
        with self.lock_producers:
            self.id_producer += 1
            return self.id_producer


    def publish(self, producer_id, product):
        
        
        if producer_id in self.products_lists:
            if len(self.products_lists[producer_id]) >= self.queue_size_per_producer:
                return False
        
        self.products_lists[producer_id].append(product)
        return True

    def new_cart(self):
        
        with self.lock_carts:
            self.id_cart += 1
        return self.id_cart

    def add_to_cart(self, cart_id, product):
        
        
        for key, values in self.products_lists.items():
            
            for j in values:
                if j == product:
                    
                    self.bought_items[j].append(key)
                    
                    self.carts_lists[cart_id].append(j)
                    
                    self.products_lists[key].remove(j)
                    return True
        return False

    def remove_from_cart(self, cart_id, product):
        
        
        self.carts_lists[cart_id].remove(product)
        
        self.products_lists[self.bought_items[product].pop()].append(product)

    def place_order(self, cart_id):
        
        return self.carts_lists[cart_id]
        >>>> file: producer.py


from threading import Thread, Lock
import time

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        

        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.id_producer = self.marketplace.register_producer()
        self.lock = Lock()


    def run(self):
        
        while True:
            
            for i in self.products:
                operation_counter = 0
                
                while operation_counter < i[1]:
                    
                    with self.lock:
                        
                        if not self.marketplace.publish(self.id_producer, i[0]):
                            time.sleep(self.republish_wait_time)
                        else:
                            operation_counter += 1
                            
                            time.sleep(i[2])


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
