


from threading import Thread
import time

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def add_to_cart(self, cart_id, product, quantity):
        while quantity > 0:
            while not self.marketplace.add_to_cart(cart_id, product):
                time.sleep(self.retry_wait_time)
            quantity = quantity - 1

    def remove_from_cart(self, cart_id, product, quantity):
        while quantity > 0:
            while not self.marketplace.remove_from_cart(cart_id, product):
                time.sleep(self.retry_wait_time)
            quantity = quantity - 1

    def order_cart(self, cart_id):
        products = self.marketplace.place_order(cart_id)
        for prod in products:
            print(f"{self.name} bought {prod}")

    def run(self):
        cart_id = self.marketplace.new_cart()
        for cart in self.carts:
            for instruction in cart:
                instr_type = instruction.get("type")
                product = instruction.get("product")
                quantity = instruction.get("quantity")
                if instr_type == "add":
                    self.add_to_cart(cart_id, product, quantity)
                elif instr_type == "remove":
                    self.remove_from_cart(cart_id, product, quantity)
            self.order_cart(cart_id)

import logging
import unittest
from logging.handlers import RotatingFileHandler
from threading import RLock


class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer

        self.producers = 0
        self.producer_queue_size = {}
        self.producer_lock = RLock()

        self.carts = 0
        self.cart_lock = RLock()

        self.cart_id_data = {}
        self.cart_id_lock = {}

        self.products = []
        self.product_lock = RLock()

        self.logger = logging.getLogger("Marketplace")
        self.logger.addHandler(RotatingFileHandler("marketplace.log", maxBytes=10000, backupCount=5))
        self.logger.setLevel(logging.INFO)

    def register_producer(self):
        
        self.producer_lock.acquire()
        self.producers = self.producers + 1
        producer_id = self.producers


        self.producer_queue_size[producer_id] = self.queue_size_per_producer
        self.producer_lock.release()

        self.logger.info(f"Producer {producer_id} has been registered")

        return producer_id

    def publish(self, producer_id, product):
        
        if self.producer_queue_size[producer_id] > 0:
            self.product_lock.acquire()
            self.products.append((product, producer_id))
            self.product_lock.release()

            self.producer_lock.acquire()
            self.producer_queue_size[producer_id] = self.producer_queue_size[producer_id] - 1
            self.producer_lock.release()

            self.logger.info(f"Product {product} has been published by producer {producer_id}")
            return True
        return False

    def new_cart(self):
        
        self.cart_lock.acquire()

        self.carts = self.carts + 1
        cart_id = self.carts
        self.cart_id_data[cart_id] = []
        self.cart_id_lock[cart_id] = RLock()



        self.cart_lock.release()

        self.logger.info(f"Customer created cart {cart_id}")
        return cart_id

    def add_to_cart(self, cart_id, product):
        
        self.product_lock.acquire()
        for prod in self.products:
            if prod[0] == product:
                self.products.remove(prod)
                self.product_lock.release()

                self.cart_id_lock[cart_id].acquire()
                self.cart_id_data[cart_id].append(prod)
                self.cart_id_lock[cart_id].release()

                self.logger.info(f"Customer added product {product} to cart {cart_id}")
                return True



        self.product_lock.release()
        self.logger.info(f"Customer failed to add product {product} to cart {cart_id}")
        return False

    def remove_from_cart(self, cart_id, product):
        
        self.cart_id_lock[cart_id].acquire()
        self.product_lock.acquire()

        for prod in self.cart_id_data[cart_id]:
            if prod[0] == product:
                self.cart_id_data[cart_id].remove(prod)
                self.products.append(prod)



                self.cart_id_lock[cart_id].release()
                self.product_lock.release()

                self.logger.info(f"Customer removed product {prod} from cart {cart_id}")
                return True

    def place_order(self, cart_id):
        
        products = []
        self.cart_id_lock[cart_id].acquire()
        for prod, producer_id in self.cart_id_data[cart_id]:
            products.append(prod)

            self.producer_lock.acquire()
            self.producer_queue_size[producer_id] = self.producer_queue_size[producer_id] + 1
            self.producer_lock.release()

        self.cart_id_data[cart_id].clear()
        self.cart_id_lock[cart_id].release()
        self.logger.info(f"Customer placed an order and emptied cart {cart_id}")
        return products


class TestMarketplace(unittest.TestCase):
    def setUp(self) -> None:
        self.marketplace = Marketplace(2)

    def test_register_producer(self):
        producers = []
        for _ in range(1, 50):
            producers.append(self.marketplace.register_producer())
        self.assertEqual(len(producers), len(list(dict.fromkeys(producers))))

    def test_publish(self):
        producer = self.marketplace.register_producer()

        product = "Coffee"
        self.assertEqual(self.marketplace.publish(producer, product), True)
        self.assertEqual(self.marketplace.products[0], (product, producer))

        product = "Tea"
        self.assertEqual(self.marketplace.publish(producer, product), True)
        self.assertEqual(self.marketplace.products[1], (product, producer))

        self.assertEqual(self.marketplace.publish(producer, product), False)

    def test_add_to_cart(self):
        cart_id = self.marketplace.new_cart()
        product = "Coffee"
        self.assertEqual(self.marketplace.add_to_cart(cart_id, product), False)

        producer = self.marketplace.register_producer()
        self.marketplace.publish(producer, product)

        self.assertEqual(self.marketplace.products[0], (product, producer))
        self.assertEqual(self.marketplace.add_to_cart(cart_id, product), True)
        self.assertEqual(self.marketplace.cart_id_data[cart_id], [(product, producer)])
        self.assertEqual(self.marketplace.products, [])

    def test_remove_from_cart(self):
        cart_id = self.marketplace.new_cart()
        product = "Tea"
        producer = self.marketplace.register_producer()

        self.marketplace.publish(producer, product)
        self.marketplace.add_to_cart(cart_id, product)
        self.assertEqual(self.marketplace.remove_from_cart(cart_id, product), True)
        self.assertEqual(self.marketplace.products[0], (product, producer))

    def test_place_order(self):
        cart_id = self.marketplace.new_cart()
        product = "Coffee"
        producer = self.marketplace.register_producer()

        self.marketplace.publish(producer, product)
        self.marketplace.add_to_cart(cart_id, product)
        self.assertEqual(self.marketplace.place_order(cart_id), product)


from threading import Thread
import time


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def publish(self, product, quantity, publish_wait_time, producer_id):
        while quantity > 0:
            while not self.marketplace.publish(producer_id, product):
                time.sleep(self.republish_wait_time)
            time.sleep(publish_wait_time)
            quantity = quantity - 1

    def run(self):
        producer_id = self.marketplace.register_producer()
        while True:
            for product in self.products:
                self.publish(product[0], product[1], product[2], producer_id)


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
