


from threading import Thread, Lock
import time

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self._carts = carts
        self._market = marketplace
        self.retry_time = retry_wait_time
        self._lock = Lock()

    def run(self):
        cart_id = self._market.new_cart()
        for op_list in self._carts:

            
            for op in op_list:

                op_type = op["type"]
                prod = op["product"]
                quantity = op["quantity"]

                if op_type == "add":

                    
                    while quantity > 0:
                        ret = self._market.add_to_cart(cart_id, prod)

                        if ret == True:
                            quantity -= 1
                        else:
                            time.sleep(self.retry_time)

                if op_type == "remove":

                    while quantity > 0:
                        self._market.remove_from_cart(cart_id, prod)
                        quantity -= 1

            with self._lock:
                products_list = self._market.place_order(cart_id)
                for prod in products_list:
                    print("cons" + str(cart_id) + " bought " + str(prod))

from threading import Lock

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self._producers_queue = {}  
        self._carts = {}            
        self._id_carts = 0          


        self._id_producers = 0      
        self._products = []         
        self._product_producer = {} 
        self._queue_size = queue_size_per_producer
        self._lock0 = Lock()
        self._lock1 = Lock()
        self._lock2 = Lock()

    def register_producer(self):
        
        with self._lock0:
            self._id_producers += 1

        return self._id_producers

    def publish(self, producer_id, product):
        
        
        if producer_id not in self._producers_queue:
            self._producers_queue[producer_id] = 0

        
        if self._producers_queue[producer_id] >= self._queue_size:
            return False

        
        self._producers_queue[producer_id] += 1

        
        self._products.append(product)

        
        self._product_producer[product] = producer_id

        return True

    def new_cart(self):
        
        with self._lock1:
            self._id_carts += 1

        return self._id_carts

    def add_to_cart(self, cart_id, product):
        

        
        
        with self._lock2:
            if cart_id not in self._carts:
                self._carts[cart_id] = []
            
            


            if product not in self._products:
                return False
            
            
            self._products.remove(product)

            
            pid = self._product_producer[product]
            self._producers_queue[pid] -= 1

            
            self._carts[cart_id].append(product)
        
        return True


    def remove_from_cart(self, cart_id, product):
        
        
        self._carts[cart_id].remove(product)

        


        pid = self._product_producer[product]
        self._producers_queue[pid] += 1

        
        self._products.append(product)

    def place_order(self, cart_id):
                
        cart_prods_copy = self._carts[cart_id].copy()
        self._carts[cart_id] = []
        
        return cart_prods_copy

import time
from threading import Thread

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self._prods = products
        self._market = marketplace
        self._id = marketplace.register_producer()
        self._rwait_time = republish_wait_time

    def run(self):
        while True:

            
            for product in self._prods:
                prod = product[0]
                quantity = product[1]
                repub_time = product[2]

                
                while quantity > 0:
                    ret = self._market.publish(self._id, prod)
                    if ret is True:
                        time.sleep(self._rwait_time)
                        quantity -= 1
                    else:
                        time.sleep(repub_time)


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