


from threading import Thread
from time import sleep

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        for cart in self.carts:
            
            cart_id = self.marketplace.new_cart()

            for opperation in cart:
                if opperation["type"] == "add":
                    gotten_quantity = 0

                    
                    while gotten_quantity < opperation["quantity"]:
                        if self.marketplace.add_to_cart(cart_id, opperation["product"]):
                            
                            gotten_quantity += 1
                        else:
                            
                            sleep(self.retry_wait_time)

                elif opperation["type"] == "remove":
                    
                    for _ in range(opperation["quantity"]):
                        self.marketplace.remove_from_cart(cart_id, opperation["product"])

            
            
            self.marketplace.place_order(cart_id)


from threading import Lock
from threading import currentThread

import logging
from logging.handlers import RotatingFileHandler
import time


logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = RotatingFileHandler('marketplace.log', maxBytes=1000000, backupCount=3)

formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(funcName)s: %(message)s', \
    '%Y-%m-%d %H:%M:%S')
formatter.converter = time.gmtime
handler.setFormatter(formatter)

logger.addHandler(handler)


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        logging.info("enter: %s", queue_size_per_producer)
        self.queue_size_per_producer = queue_size_per_producer

        
        self.unused_producer_id = 0

        
        self.products = []

        
        self.register_lock = Lock()

        
        self.unused_cart_id = 0

        
        
        self.carts = []

        
        self.cart_lock = Lock()

        
        
        self.inventory = {}

        
        
        self.products_lock = Lock()

        
        self.consumer_print_lock = Lock()



        logging.info("exit")



    def register_producer(self):
        
        logging.info("enter")

        
        with self.register_lock:
            new_id = self.unused_producer_id
            self.unused_producer_id += 1

        
        
        self.products.append([])

        logging.info("exit: %s", new_id)

        return new_id

    def publish(self, producer_id, product):
        
        logging.info("enter: %s, %s", producer_id, product)

        if len(self.products[producer_id]) < self.queue_size_per_producer:
            self.products[producer_id].append(product)

            
            self.inventory.setdefault(product, [])
            self.inventory[product].append(producer_id)

            
            logging.info("exit: True")
            return True

        
        logging.info("exit: False")
        return False



    def new_cart(self):
        
        logging.info("enter")

        
        with self.cart_lock:
            new_id = self.unused_cart_id
            self.unused_cart_id += 1

        
        
        self.carts.append([])

        logging.info("exit: %s", new_id)
        return new_id

    def add_to_cart(self, cart_id, product):
        
        logging.info("enter: %s, %s", cart_id, product)

        with self.products_lock:
            if product in self.inventory:
                if len(self.inventory[product]) > 0:
                    
                    producer_id = self.inventory[product].pop()
                    
                    self.carts[cart_id].append((product, producer_id))
                    
                    self.products[producer_id].remove(product)

                    logging.info("exit: True")
                    return True

        
        logging.info("exit: False")
        return False

    def remove_from_cart(self, cart_id, product):
        
        logging.info("enter: %s, %s", cart_id, product)

        
        
        cart_list = [tup for tup in self.carts[cart_id] if product in tup]

        if len(cart_list) > 0:
            (_, producer_id) = cart_list[0]
            
            self.carts[cart_id].remove(cart_list[0])
            
            self.inventory[product].append(producer_id)
            


            self.products[producer_id].append(product)

        logging.info("exit")

    def place_order(self, cart_id):
        
        logging.info("enter: %s", cart_id)

        
        with self.consumer_print_lock:
            
            
            for item in [product for (product, _) in self.carts[cart_id]]:
                print(f"{currentThread().getName()} bought {item}")

        logging.info("exit")


from threading import Thread
from time import sleep

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

        self.producer_id = marketplace.register_producer()

    def run(self):
        
        while True:
            for product in self.products:
                published_quantity = 0

                
                while published_quantity < product[1]:
                    if self.marketplace.publish(self.producer_id, product[0]):
                        
                        published_quantity += 1
                        sleep(self.republish_wait_time)
                    else:
                        
                        sleep(product[2])


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
