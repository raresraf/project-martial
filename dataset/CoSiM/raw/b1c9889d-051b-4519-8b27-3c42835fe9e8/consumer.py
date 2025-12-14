




from threading import Thread
import time

QUANTITY = "quantity"
PRODUCT = "product"
TYPE = "type"
ADD = "add"
REMOVE = "remove"

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):

        for cart in self.carts:
            
            id_cart = self.marketplace.new_cart()

            
            for element in cart:
                counter = 0
                
                while counter < element[QUANTITY]:
                    val = False
                    
                    if element[TYPE] == ADD:
                        val = self.marketplace.add_to_cart(id_cart, element[PRODUCT])
                    elif element[TYPE] == REMOVE:
                        val = self.marketplace.remove_from_cart(id_cart, element[PRODUCT])
                    
                    
                    if val:
                        counter += 1
                    elif not val:
                        time.sleep(self.retry_wait_time)
            
            self.marketplace.place_order(id_cart)>>>> file: marketplace.py

import uuid
from threading import Lock, currentThread

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        
        self.producer_size = []
        
        self.producer_products = []

        
        self.producers_items = {}
        
        self.cart_items = {}
        
        self.number_of_carts = 0

        
        self.lock_number_carts = Lock()
        self.lock_add = Lock()
        self.lock_remove = Lock()
        self.lock_register = Lock()
        self.lock_print = Lock()

    def register_producer(self):
        
        
        
        
        with self.lock_register:
            prod_id = len(self.producer_size)
            self.producer_size.append(0)
        return prod_id


    def publish(self, producer_id, product):
        

        
        
        if self.producer_size[producer_id] >= self.queue_size_per_producer:
            return False
        
        
        self.producer_size[producer_id] += 1
        
        self.producer_products.append(product)
        
        self.producers_items[product] = producer_id

        return True

    def new_cart(self):
        
        
        
        with self.lock_number_carts:
            self.number_of_carts += 1
            new_cart_id = self.number_of_carts 

        
        
        self.cart_items[new_cart_id] = []
        return new_cart_id

    def add_to_cart(self, cart_id, product):
        
        
        
        
        with self.lock_add:
            if product not in self.producer_products:
                return False
            
            
            self.producer_size[self.producers_items[product]] -= 1
            
            
            self.producer_products.remove(product)

        
        self.cart_items[cart_id].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        
        
        self.cart_items[cart_id].remove(product)
        
        self.producer_products.append(product)

        with self.lock_remove:
            
            self.producer_size[self.producers_items[product]] += 1
        return True
        
    def place_order(self, cart_id):
        
        
        my_prods = self.cart_items.pop(cart_id, None)

        
        for elem in my_prods:


            with self.lock_print:
                print(currentThread().getName() + " bought " + str(elem))

        return my_prods


from threading import Thread
import time

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.id = self.marketplace.register_producer()

    def run(self):
        
        while 1:
            
            for (product, number_product, time_to_wait) in self.products:
                counter = 0
                
                while counter < number_product:
                    published = self.marketplace.publish(self.id, product)

                    if published:
                        time.sleep(time_to_wait)
                        counter += 1
                    else:
                        time.sleep(self.republish_wait_time)
                        


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
