


import time
from threading import Thread


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs.get("name")

    def run(self):
        
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for operation in cart:
                product = operation.get("product")
                quantity = operation.get("quantity")
                for _ in range(quantity):
                    if operation.get("type") == "add":
                        res = False
                        while not res:
                            res = self.marketplace.add_to_cart(cart_id, product)
                            time.sleep(self.retry_wait_time)
                    elif operation.get("type") == "remove":
                        self.marketplace.remove_from_cart(cart_id, product)

            products = self.marketplace.place_order(cart_id)

            for product in products:
                print(f"{self.name} bought {product}")


import logging
from logging.handlers import RotatingFileHandler
import functools
import inspect
import time

def setup_logger():
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler('marketplace.log', maxBytes=500000, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logging.Formatter.converter = time.gmtime
    logger.addHandler(handler)


def log_function(wrapped_function):
    

    @functools.wraps(wrapped_function)
    def wrapper(*args):
        logger = logging.getLogger(__name__)

        
        func_name = wrapped_function.__name__
        logger.info("Entering %s", {func_name})

        
        func_args = inspect.signature(wrapped_function).bind(*args).arguments
        func_args_str = '\n\t'.join(
            f"{var_name} = {var_value}"
            for var_name, var_value
            in func_args.items()
        )
        logger.info("\t%s", func_args_str)

        
        out = wrapped_function(*args)

        
        logger.info("Return: %s - %s", type(out), out)
        logger.info("Done running %s", func_name)

        return out

    return wrapper


from threading import Lock
from .logger import setup_logger, log_function

class Cart:
    

    def __init__(self):
        
        self.products = []
        self.producer_ids = []

    def add_to_cart(self, product, producer_id):
        
        self.products.append(product)
        self.producer_ids.append(producer_id)

    def remove_from_cart(self, product):
        
        for i in range(len(self.products)):
            if self.products[i] == product:
                producer_id = self.producer_ids[i]
                self.products.remove(product)
                self.producer_ids.remove(producer_id)
                return producer_id
        return None


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        setup_logger()
        self.queue_size_per_producer = queue_size_per_producer

        
        self.producer_queues = {}
        self.producer_queues_locks = {}

        self.producer_id_counter = 0
        self.producer_id_lock = Lock()

        
        self.carts = {}

        self.cart_id_counter = 0
        self.cart_id_lock = Lock()

    @log_function
    def register_producer(self):
        
        with self.producer_id_lock:
            producer_id = self.producer_id_counter
            self.producer_queues[producer_id] = []
            self.producer_queues_locks[producer_id] = Lock()
            self.producer_id_counter += 1
            return producer_id

    @log_function
    def publish(self, producer_id, product):
        
        with self.producer_queues_locks[producer_id]:
            if len(self.producer_queues[producer_id]) <= self.queue_size_per_producer:
                self.producer_queues[producer_id].append(product)
                return True
        return False

    @log_function
    def new_cart(self):
        
        with self.cart_id_lock:
            cart_id = self.cart_id_counter
            self.carts[cart_id] = Cart()
            self.cart_id_counter += 1
            return cart_id

    @log_function
    def add_to_cart(self, cart_id, product):
        
        producers_no = 0
        with self.producer_id_lock:
            producers_no = self.producer_id_counter

        for i in range(producers_no):
            with self.producer_queues_locks[i]:
                if product in self.producer_queues[i]:
                    self.producer_queues[i].remove(product)
                    self.carts[cart_id].add_to_cart(product, i)
                    return True
        return False

    @log_function
    def remove_from_cart(self, cart_id, product):
        
        producer_id = self.carts[cart_id].remove_from_cart(product)
        with self.producer_queues_locks[producer_id]:
            self.producer_queues[producer_id].append(product)

    @log_function
    def place_order(self, cart_id):
        
        return self.carts[cart_id].products


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        thread_arg = kwargs["daemon"]
        Thread.__init__(self, daemon=thread_arg)
        self.operations = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        
        producer_id = self.marketplace.register_producer()
        while True:
            for operation in self.operations:
                product = operation[0]
                quantity = operation[1]
                sleep_time = operation[2]
                time.sleep(sleep_time)
                for _ in range(quantity):
                    if not self.marketplace.publish(producer_id, product):
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
