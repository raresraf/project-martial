


from threading import Thread
import time


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time: int = retry_wait_time
        Thread.__init__(self, **kwargs)

    def print_carts(self, id_cart):
        

        
        list_order = self.marketplace.place_order(id_cart)
        for product in list_order:
            self.marketplace.print_cons(self.name, product)


    def add_product_to_cart(self, id_cart, prod):
        
        
        go_next = self.marketplace.add_to_cart(id_cart,prod)
        if go_next is False:
            time.sleep(self.retry_wait_time)
            self.add_product_to_cart(id_cart, prod)


    def run(self):
        id_cart = self.marketplace.new_cart()
        for products in self.carts:
            for produs in products:
                for _ in range(produs["quantity"]):
                    if produs["type"] == "remove":
                        self.marketplace.remove_from_cart(id_cart, produs)
                    else:
                        self.add_product_to_cart(id_cart, produs)
        self.print_carts(id_cart)

from threading import Lock
import logging
from logging.handlers import RotatingFileHandler
import time


logger = logging.getLogger('loggerOne')
logger.setLevel(logging.INFO)


handler = RotatingFileHandler('file.log', maxBytes=500000, backupCount=10)


formatter = logging.Formatter('%(asctime)s %(levelname)8s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


logging.Formatter.converter = time.gmtime


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        

        logger.info("init Marketplace, argument qsise: %d", queue_size_per_producer)
        self.queue_size_per_producer = queue_size_per_producer
        self.id_prod = 0
        self.id_cart = 0
        self.producers = []
        self.producers.append(0)
        self.products = []
        self.products.append([])
        self.carts = []
        
        self.add_to_cart_lock = Lock()
        self.publish_lock = Lock()
        self.print_lock = Lock()
        self.new_cart_lock = Lock()
        self.register_producer_lock = Lock()
        self.remove_from_cart_lock = Lock()

        logger.info("init Marketplace, all")

    def register_producer(self):


        
        
        self.register_producer_lock.acquire()
        logger.info("register_producer, id_prod =%d", self.id_prod)
        self.producers.append(self.queue_size_per_producer)
        self.id_prod = self.id_prod + 1
        self.products.append([])


        self.register_producer_lock.release()

        logger.info("register_producer, id_prod-exit =%d", self.id_prod)
        return self.id_prod

    def publish(self, producer_id, product):
        
        
        it_produced = False
        
        self.publish_lock.acquire()
        logger.info("publish; id_producer =%d", producer_id)
        if self.producers[producer_id] > 0:
            self.products[producer_id].append(product[0])
            self.producers[producer_id] = self.producers[producer_id] -1
            it_produced = True

        logger.info("publish; exit =%d", producer_id, )
        self.publish_lock.release()
        return it_produced


    def new_cart(self):


        
        

        self.new_cart_lock.acquire()
        logger.info("new_cart;")

        self.carts.append([])
        current_cart_id = self.id_cart
        self.id_cart = self.id_cart+1

        logger.info("new_cart; iese")
        self.new_cart_lock.release()

        return current_cart_id


    def add_to_cart(self, cart_id, product):
        
        
        producer_found_id = 0
        product_found = False
        available_product = []

        
        
        self.add_to_cart_lock.acquire()
        logger.info("ad_to_cart %d;", cart_id)

        for producer in self.products:
            for available_product in producer:
                if product["product"] == available_product:
                    product_found = True
                    break

            if product_found:
                break
            producer_found_id = producer_found_id + 1

        if product_found:
            self.products[producer_found_id].remove(available_product)
            self.producers[producer_found_id] = self.producers[producer_found_id] + 1
            self.carts[cart_id].append(available_product)
            logger.info("ad_to_cart exit True;")
            self.add_to_cart_lock.release()
            return True



        logger.info("ad_to_cart exit False;")
        self.add_to_cart_lock.release()
        return False


    def remove_from_cart(self, cart_id, product):
        

        product_found = False
        producer_found_id = 0
        available_product = []

        
        
        self.remove_from_cart_lock.acquire()
        logger.info("reomce_from_cart start True;%d", cart_id)
        for available_product in self.carts[cart_id]:
            if product["product"] == available_product:
                product_found = True
                break
            producer_found_id = producer_found_id + 1

        if product_found:
            del self.carts[cart_id][producer_found_id]
            self.products[0].append(available_product)

        logger.info("ad_to_cart exit;")
        self.remove_from_cart_lock.release()


    def place_order(self, cart_id):
        
        
        logger.info("place_order start %d;", cart_id)
        copie =self.carts[cart_id]
        self.carts[cart_id] = []
        logger.info("place_order end;")
        return copie


    def print_cons(self, name, product):
        
        
        self.print_lock.acquire()
        logger.info("print_cons start; name=%s;", name)


        print(name, "bought", product)
        self.print_lock.release()


from threading import Thread
import time
class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        

        Thread.__init__(self, **kwargs)
        self.products  = products

        self.marketplace = marketplace
        self.republish_wait_time: int = republish_wait_time
        self.my_id = self.marketplace.register_producer()

    def run(self):
        while True:
            for  prod in self.products:
                for _ in range(prod[1]):
                    it_worked = self.marketplace.publish(self.my_id, prod)
                    if it_worked:
                        time.sleep(prod[2])
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
