


from threading import Thread
from time import sleep


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)

        self.marketplace = marketplace
        self.carts = carts
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def run(self):
        for cart in self.carts:
            new_cart = self.marketplace.new_cart()


            for prod in cart:
                i = 0
                while i < prod["quantity"]:
                    if prod["type"] == "add":
                        if self.marketplace.add_to_cart(new_cart, prod["product"]):
                            i += 1
                        else:
                            sleep(self.retry_wait_time)


                    elif prod["type"] == "remove":
                        self.marketplace.remove_from_cart(new_cart, prod["product"])
                        i += 1
            order = self.marketplace.place_order(new_cart)
            for product in order:
                print(self.name, "bought", product)>>>> file: marketplace.py

from threading import Lock, currentThread


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.q_max_size_per_producer = queue_size_per_producer

        
        
        


        self.register_lock = Lock()

        
        
        self.new_cart_lock = Lock()

        
        

        self.add_lock = Lock()
        self.remove_lock = Lock()

        
        self.no_producers = 0
        self.no_carts = 0

        
        self.no_products = {}
        
        self.products = {}
        
        self.market_products = []
        
        self.carts = {}

    def register_producer(self):
        
        with self.register_lock:
            producer_id = self.no_producers
            self.no_producers += 1
            self.no_products[producer_id] = 0

        return producer_id

    def publish(self, producer_id, product):
        
        if self.no_products[producer_id] >= self.q_max_size_per_producer:
            return False
        self.products[product] = producer_id
        self.no_products[producer_id] += 1
        self.market_products.append(product)

        return True

    def new_cart(self):
        
        self.new_cart_lock.acquire()
        cart_id = self.no_carts
        self.no_carts += 1
        self.carts[cart_id] = []
        self.new_cart_lock.release()

        return cart_id

    def add_to_cart(self, cart_id, product):
        
        self.add_lock.acquire()
        if product not in self.market_products:
            self.add_lock.release()
            return False
        if cart_id not in self.carts.keys():
            self.add_lock.release()
            return False
        self.market_products.remove(product)
        self.no_products[self.products[product]] -= 1
        self.carts[cart_id].append(product)
        self.add_lock.release()
        return True

    def remove_from_cart(self, cart_id, product):
        
        self.remove_lock.acquire()
        if product not in self.carts[cart_id]:
            self.remove_lock.release()
            return False
        self.market_products.append(product)
        self.no_products[self.products[product]] += 1
        self.carts[cart_id].remove(product)
        self.remove_lock.release()
        return True

    def place_order(self, cart_id):
        
        if cart_id not in self.carts.keys():
            return None
        return self.carts[cart_id]


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.daemon = kwargs['daemon']
        self.name = kwargs['name']
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        while True:
            for product in self.products:
                i = 0


                while i < product[1]:
                    if self.marketplace.publish(self.producer_id, product[0]):
                        time.sleep(product[2])
                        i += 1
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
