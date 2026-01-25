


import time
from threading import Thread
import threading


class Consumer(Thread):
    
    print_lock = threading.Lock()

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        super().__init__(**kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        cart_id = self.marketplace.new_cart()



        for cart in self.carts:
            for item in cart:
                for _ in range(item["quantity"]):
                    successful = False
                    while not successful:
                        if item["type"] == "add":
                            successful = self.marketplace.add_to_cart(
                                cart_id, item["product"])
                            if not successful:
                                time.sleep(self.retry_wait_time)
                        else:
                            self.marketplace.remove_from_cart(
                                cart_id, item["product"])
                            successful = True

            with Consumer.print_lock:
                for product in self.marketplace.place_order(cart_id):
                    print(f"cons{cart_id + 1} bought {product}")


import time
import logging
import threading
import unittest

from typing import Dict, List
from .product import MarketProduct, CartProduct
from .rwlock import RWLock
from logging.handlers import RotatingFileHandler
from logging import Formatter


logging.Formatter.converter = time.gmtime
logger = logging.getLogger("marketplace")
handler = RotatingFileHandler(
    'marketplace.log',
    maxBytes=1024 * 1024 * 32,
    backupCount=10)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
handler.setFormatter(
    Formatter("[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s"))


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer

        self.producers: Dict[int, List[MarketProduct]] = {}

        self.carts: Dict[int, List[CartProduct]] = {}

        self.producer_locks: Dict[int, threading.Lock] = {}

        self.prod_lock = RWLock()

        self.cart_lock = RWLock()

    def register_producer(self):
        
        
        with self.prod_lock.write_lock():
            
            next_producer_id = len(self.producers)

            
            
            self.producers[next_producer_id] = []

            
            self.producer_locks[next_producer_id] = threading.Lock()



            logger.info(f"new producer {next_producer_id} registered")
            return next_producer_id

    def publish(self, producer_id, product):
        
        
        with self.producer_locks[producer_id]:
            
            if len(self.producers[producer_id]) < self.queue_size_per_producer:
                
                self.producers[producer_id].append(
                    MarketProduct(product=product, reserved_for=None))

                logger.info(f"producer {producer_id} published {product}")
                return True



            logger.info(f"producer {producer_id} failed publish {product}")
            
            return False

    def new_cart(self):
        
        
        with self.cart_lock.write_lock():
            
            new_cart_id = len(self.carts)

            


            self.carts[new_cart_id] = []

            logger.info(f"new cart {new_cart_id} registered")

            return new_cart_id

    def add_to_cart(self, cart_id, product):
        
        
        with self.prod_lock.read_lock():
            for id, products in self.producers.items():
                
                with self.producer_locks[id]:
                    for available_product in products:
                        
                        if available_product.product == product and available_product.reserved_for is None:
                            

                            self.carts[cart_id].append(
                                CartProduct(product=product, bought_from=id))

                            
                            available_product.reserved_for = cart_id

                            logger.info(
                                f"product {product} from producer {id} added to cart {cart_id}")
                            return True

            logger.info(f"product {product} not found by {cart_id}")
            return False

    def remove_from_cart(self, cart_id, product):
        
        
        with self.cart_lock.read_lock():
            for cart_product in self.carts[cart_id]:
                
                if cart_product.product == product:
                    
                    with self.producer_locks[cart_product.bought_from]:
                        
                        for available_product in self.producers[cart_product.bought_from]:
                            
                            if available_product.product == product and available_product.reserved_for == cart_id:
                                
                                
                                available_product.reserved_for = None

                                self.carts[cart_id].remove(cart_product)
                                logger.info(
                                    f"product {product} by producer {cart_product.bought_from} \
                                    removed from {cart_id}")
                                return

    def delete_cart_product_from_producer(
            self, cart_id, carted_product: CartProduct):
        with self.producer_locks[carted_product.bought_from]:
            try:
                self.producers[carted_product.bought_from].remove(
                    MarketProduct(product=carted_product.product, reserved_for=cart_id)
                )
            except BaseException:
                logger.critical(
                    f"failed to remove item {carted_product} from {cart_id}")
                raise SystemExit

    def place_order(self, cart_id):
        
        products_in_cart = []

        with self.cart_lock.write_lock():
            for cart_product in self.carts[cart_id]:
                self.delete_cart_product_from_producer(cart_id, cart_product)
                products_in_cart.append(cart_product.product)

            self.carts[cart_id] = []

        return products_in_cart


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        super().__init__(**kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        id_producer = self.marketplace.register_producer()

        while True:
            for product, quantity, delay in self.products:
                for _ in range(quantity):
                    successful = False
                    while not successful:
                        successful = self.marketplace.publish(
                            id_producer, product)

                        if not successful:
                            time.sleep(self.republish_wait_time)
                    time.sleep(delay)


from dataclasses import dataclass
from typing import Optional


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


@dataclass(init=True, repr=True, order=False)
class MarketProduct:
    
    product: Product
    reserved_for: Optional[int]


@dataclass(init=True, repr=True, order=False)
class CartProduct:
    
    product: Product
    bought_from: int
import threading
from contextlib import contextmanager


class RWLock:
    def __init__(self) -> None:
        self._access_lock = threading.Lock()
        self._readers_lock = threading.Lock()
        self._readers = 0

    def _read_acquire(self):
        self._readers_lock.acquire()

        self._readers += 1

        if self._readers == 1:
            self._access_lock.acquire()

        self._readers_lock.release()

    def _read_release(self):
        self._readers_lock.acquire()

        self._readers -= 1

        if self._readers == 0:
            self._access_lock.release()

        self._readers_lock.release()

    def _write_acquire(self):
        self._access_lock.acquire()

    def _write_release(self):
        self._access_lock.release()

    @contextmanager
    def read_lock(self):
        

        try:
            self._read_acquire()
            yield
        finally:
            self._read_release()

    @contextmanager
    def write_lock(self):
        

        try:
            self._write_acquire()
            yield
        finally:
            self._write_release()
