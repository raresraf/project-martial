


from threading import Thread
from time import sleep


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.cart_id = -1
        self.marketplace = marketplace
        self.wait_time = retry_wait_time
        self.name = kwargs["name"]

    def run(self):
        for cart in self.carts:
            self.cart_id = self.marketplace.new_cart()

            for cart_op in cart:
                op_type = cart_op["type"]
                quantity = cart_op["quantity"]
                prod = cart_op["product"]

                if op_type == "add":
                    while quantity > 0:
                        while True:
                            ret = self.marketplace.add_to_cart(self.cart_id, prod)
                            if ret:
                                break
                            sleep(self.wait_time)
                        quantity -= 1
                else:
                    while quantity > 0:
                        self.marketplace.remove_from_cart(self.cart_id, prod)
                        quantity -= 1

            lista = self.marketplace.place_order(self.cart_id)
            for cart_item in lista:
                print("{} bought {}".format(self.name, cart_item))

import logging
from logging import getLogger
from re import L
from threading import Lock
from logging.handlers import RotatingFileHandler

class Marketplace:
    
    def __init__(self, queue_size_per_producer, ):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.producatori = {}
        self.cosuri = {}
        self.producatori_id = []
        self.cosuri_id = []
        self.prod_id = 1
        self.cos_id = 1
        self.logger = getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.handlers.RotatingFileHandler("marketplace.log"))

    def register_producer(self):
        
        lock = Lock()


        with lock:
            self.prod_id = sum(self.producatori_id)
            self.prod_id += 1
            self.producatori_id.append(self.prod_id)
            producator = {'produse': []}
            self.producatori[self.prod_id] = producator
            self.logger.info("Function  producator_id: %d",
                             "register_producer", self.prod_id)
            return self.prod_id

    def publish(self, producer_id, product):
        
        lock = Lock()
        with lock:
            for prod_id, produse_publicate in self.producatori.items():
                if prod_id == producer_id:
                    if len(produse_publicate['produse']) < self.queue_size_per_producer:
                        produse_publicate['produse'].append(product)
                        self.logger.info("Function %s called by:%d,product:%s ,return: TRUE\n",
                                         "publish", producer_id, product)
                        return True
            self.logger.info("Function  called by producer_id: %d, product:%s, return: FALSE\n",
                             "publish", producer_id, product)
            return False



    def new_cart(self):
        
        lock = Lock()


        with lock:
            self.cos_id = sum(self.cosuri_id)
            self.cos_id += 2
            self.cosuri_id.append(self.cos_id)
            cosuri = {'produse_rezervate': []}


            self.cosuri[self.cos_id] = cosuri
            return self.cos_id

    def add_to_cart(self, cart_id, product):
        
        lock = Lock()


        with lock:
            for cos, continut in self.cosuri.items():
                if cos == cart_id:
                    for producator, produse_publicate in self.producatori.items():
                        if product in produse_publicate['produse']:
                            continut['produse_rezervate'].append(product)
                            self.logger.info("Function %s called by: %d, product:%s,return:TRUE\n",
                                             "add_to_cart", cart_id, product)
                            return True
        self.logger.info("Function  called by cart_id: %d, product:%s , return: FALSE\n",
                         "add_to_cart", cart_id, product)
        return False


    def remove_from_cart(self, cart_id, product):
        
        lock = Lock()
        with lock:
            for cos, continut in self.cosuri.items():
                if cos == cart_id:
                    continut['produse_rezervate'].remove(product)
        self.logger.info("Function %s called by cart_id:%d and product: %s\n", "remove_from-cart",
                         cart_id, product)

    def place_order(self, cart_id):
        
        lock = Lock()
        with lock:
            for cos, continut in self.cosuri.items():
                if cos == cart_id:
                    self.logger.info("Function  cart_id: %d",
                                     "place_order", cart_id)


                    return continut['produse_rezervate']
            return None
            >>>> file: producer.py

from threading import Thread
from time import sleep

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.prod_id = self.marketplace.register_producer()

        self.republish_wait_time = republish_wait_time
        self.name = kwargs["name"]

    def run(self):
        while True:
            for produs in self.products:
                product = produs[0]
                quantity = produs[1]
                waiting_time = produs[2]

                while quantity > 0:
                    while True:
                        ret = self.marketplace.publish(self.prod_id, product)
                        if ret:
                            break
                        sleep(self.republish_wait_time)
                    quantity -= 1
                    sleep(waiting_time)


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
