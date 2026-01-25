


import time
from threading import Thread


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.id_cart = 0

    def wait(self):
        
        time.sleep(self.retry_wait_time)

    def print_output(self):
        
        cart = self.marketplace.place_order(self.id_cart)
        for product in cart:
            print(self.name + ' bought ' + str(product))

    def run(self):
        
        for cart in self.carts:
            self.id_cart = self.marketplace.new_cart()

            
            for operation in cart:
                quantity = operation['quantity']
                product = operation['product']

                
                if operation['type'] == "add":
                    while quantity > 0:
                        
                        if self.marketplace.add_to_cart(self.id_cart, product) is False:
                            self.wait()
                        else:
                            quantity -= 1

                
                if operation['type'] == "remove":
                    while quantity > 0:
                        
                        self.marketplace.remove_from_cart(self.id_cart, product)
                        quantity -= 1

            
            self.print_output()


import logging
from logging.handlers import RotatingFileHandler
import time
import threading
from threading import currentThread

logging.basicConfig(handlers=
                    [RotatingFileHandler(filename='./marketplace.log', maxBytes=400000,
                                         backupCount=10)],
                    level=logging.INFO,
                    format="[%(asctime)s]::%(levelname)s::%(message)s")
logging.Formatter.converter = time.gmtime


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.list_of_carts = []
        self.list_of_producers = []
        self.id_producer = -1
        self.id_cart = -1

    def register_producer(self):
        

        
        logging.info("register_producer() called by Thread %s",
                     currentThread().getName())

        
        register_lock = threading.Lock()

        producers = []

        
        self.id_producer += 1

        
        with register_lock:
            self.list_of_producers.append(producers)

        
        logging.info("Thread %s exited register_producer()",
                     currentThread().getName())

        return str(self.id_producer)

    def publish(self, producer_id, product):
        

        
        logging.info("publish() called by Thread %s with producer_id %s to register product %s",
                     currentThread().getName(), str(producer_id), str(product))

        
        quantity = product[1]
        sleep_time = product[2]
        id_prod = int(producer_id)

        
        publish_lock = threading.Lock()

        
        publish_check = False

        
        if len(self.list_of_producers[id_prod]) == self.queue_size_per_producer:
            
            logging.info("Thread %s with producer_id %s exited publish() with %s",
                         currentThread().getName(), str(producer_id), str(publish_check))

            return publish_check

        
        if len(self.list_of_producers[id_prod]) + quantity < self.queue_size_per_producer:
            with publish_lock:
                while quantity > 0:
                    time.sleep(sleep_time)
                    self.list_of_producers[id_prod].append(product[0])
                    quantity -= 1
        else:
            
            logging.info("Thread %s with producer_id %s exited publish() with %s",
                         currentThread().getName(), str(producer_id), str(publish_check))

            return publish_check

        publish_check = True

        
        logging.info("Thread %s with producer_id %s exited publish() with %s",
                     currentThread().getName(), str(producer_id), str(publish_check))

        return publish_check

    def new_cart(self):
        

        
        logging.info("new_cart() called by Thread %s",
                     currentThread().getName())

        
        
        cart = []
        new_cart_lock = threading.Lock()

        
        with new_cart_lock:
            self.list_of_carts.append(cart)
            self.id_cart += 1

        return self.id_cart

    def add_to_cart(self, cart_id, product):
        

        
        logging.info("add_to_cart() called by Thread %s for the cart %s to add product %s",
                     currentThread().getName(), str(cart_id), str(product))

        
        product_existence = False

        
        prod_list = []

        
        add_to_cart_lock = threading.Lock()

        
        for prod_list in self.list_of_producers:
            if product in prod_list:
                product_existence = True
                break

        
        if product_existence is True:
            with add_to_cart_lock:
                self.list_of_carts[cart_id].append(product)
                prod_list.remove(product)

        
        logging.info("Thread %s exited add_to_cart() with %s",
                     currentThread().getName(), str(product_existence))

        return product_existence

    def remove_from_cart(self, cart_id, product):
        

        
        logging.info("remove_from_cart() called by Thread %s for the cart %s to remove product %s",
                     currentThread().getName(), str(cart_id), str(product))

        
        remove_from_cart_lock = threading.Lock()

        
        
        if product in self.list_of_carts[cart_id]:
            with remove_from_cart_lock:
                self.list_of_carts[cart_id].remove(product)
                self.list_of_producers[0].append(product)

        


        logging.info("Thread %s exited remove_from_cart()",
                     currentThread().getName())

    def place_order(self, cart_id):
        

        
        logging.info("place_order() called by Thread %s for the cart %s",
                     currentThread().getName(), str(cart_id))

        return_list = self.list_of_carts[cart_id]

        
        logging.info("Thread %s exited place_order()",
                     currentThread().getName())

        return return_list


import time
from threading import Thread


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.id_producer = self.marketplace.register_producer()
        self.republish_wait_time = republish_wait_time

    def wait(self):
        
        time.sleep(self.republish_wait_time)

    def run(self):
        
        while True:
            for product in self.products:
                while self.marketplace.publish(self.id_producer, product) is False:
                    self.wait()


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
