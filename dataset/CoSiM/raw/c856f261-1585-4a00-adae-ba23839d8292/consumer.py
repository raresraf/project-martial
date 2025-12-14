


from threading import Thread
from time import sleep

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        self.carts = carts
        self.marketplace = marketplace
        self.wait_time = retry_wait_time
        self.curr_cart = []
        self.cart_id = -1
        self.kwargs = kwargs
        Thread.__init__(self, **kwargs)


    def print_cart(self, cart):
        
        lock = self.marketplace.get_print_lock()
        lock.acquire()
        for prod in cart:
            for _ in range(len(cart[prod])):
                print(self.kwargs['name'] + " bought " + str(prod))
        lock.release()


    def add_to_cart(self, product, cart_id):
        
        res = False
        while res is False:
            res = self.marketplace.add_to_cart(cart_id, product)
            if res is False:
                sleep(self.wait_time)

    def run(self):
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for cmd in cart:
                cmd_type = cmd['type']
                product = cmd['product']
                quantity = cmd['quantity']
                for _ in range(0, quantity):
                    if cmd_type == "add":
                        self.add_to_cart(product, cart_id)
                    else:
                        self.marketplace.remove_from_cart(cart_id, product)
            cart = self.marketplace.place_order(cart_id)
            self.print_cart(cart)

from threading import Lock
import time
import unittest

import logging
import logging.handlers

LOGGER = logging.getLogger("marketlogger")
LOGGER.setLevel(logging.INFO)
HANDLER = logging.handlers.RotatingFileHandler(
    "marketplace.log", maxBytes=20000, backupCount=5)
FORMATTER = logging.Formatter("%(asctime)s;%(message)s")
HANDLER.setFormatter(FORMATTER)
logging.Formatter.converter = time.gmtime
LOGGER.addHandler(HANDLER)


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        self.queue_size = queue_size_per_producer
        self.producer_id_indexer = 0
        self.cart_id_indexer = 0
        self.producers_dict = {}
        self.all_products = {}
        self.lock = Lock()
        self.cart_lock = Lock()
        self.print_lock = Lock()
        self.carts = {}

    def get_print_lock(self):
        
        LOGGER.info("print_lock returned")
        return self.print_lock

    def register_producer(self):
        
        LOGGER.info(" (register) started")
        self.lock.acquire()
        new_id = self.producer_id_indexer
        self.producer_id_indexer += 1
        self.producers_dict[new_id] = 0
        self.lock.release()
        LOGGER.info("(register) id %d", new_id)
        return new_id

    def publish(self, producer_id, product):
        
        LOGGER.info("(publish) %d, %s", producer_id, str(product))
        self.lock.acquire()
        if self.producers_dict[producer_id] == self.queue_size:
            self.lock.release()
            LOGGER.info(" (publish) producer has too many items")
            return False

        self.producers_dict[producer_id] += 1



        if product not in self.all_products:
            self.all_products[product] = [producer_id]
        else:
            self.all_products[product].append(producer_id)

        self.lock.release()
        LOGGER.info(" (publish) the product was published")
        return True

    def new_cart(self):
        
        LOGGER.info("(new cart) started")
        self.cart_lock.acquire()

        new_id = self.cart_id_indexer
        self.cart_id_indexer += 1
        self.carts[new_id] = {}

        self.cart_lock.release()
        LOGGER.info(" (new_cart) created cart with id %d", new_id)
        return new_id

    def add_to_cart(self, cart_id, product):
        
        LOGGER.info(" (add_to_cart) params %d, %s",
                    cart_id, str(product))
        self.lock.acquire()
        if product not in self.all_products or len(self.all_products[product]) == 0:
            LOGGER.info(
                "(add_to_cart) no product %s is published", str(product))
            self.lock.release()
            return False

        producer_id = self.all_products[product].pop(0)

        self.cart_lock.acquire()

        if product not in self.carts[cart_id]:
            self.carts[cart_id][product] = [producer_id]
        else:
            self.carts[cart_id][product].append(producer_id)

        self.lock.release()
        self.cart_lock.release()
        LOGGER.info(" (add_to_cart) product %s was added", str(product))
        return True

    def remove_from_cart(self, cart_id, product):
        
        LOGGER.info('(remove from cart) params %s %d',
                    str(product), cart_id)
        if product in self.carts[cart_id] and len(self.carts[cart_id][product]) != 0:
            self.cart_lock.acquire()
            self.lock.acquire()
            producer_id = self.carts[cart_id][product].pop(0)
            self.all_products[product].append(producer_id)
            self.cart_lock.release()
            self.lock.release()
        LOGGER.info(" (remove_from_cart) finished %s was removed %d",
                    str(product), cart_id)

    def place_order(self, cart_id):
        
        LOGGER.info(" (place order) param %d", cart_id)
        self.lock.acquire()

        products = self.carts[cart_id]
        for _, ids in products.items():
            for producer_id in ids:
                self.producers_dict[producer_id] -= 1
        self.lock.release()
        LOGGER.info(" (place order) %d was placed", cart_id)
        return self.carts[cart_id]


QUEUE_SIZE = 3


class TestMarketplace(unittest.TestCase):
    

    def publish_products(self, prod_id, number_of_products):
        
        for i in range(number_of_products):
            self.marketplace.publish(prod_id, str("prod_" + str(i)))

    def setUp(self):
        
        self.marketplace = Marketplace(QUEUE_SIZE)

    def test_register_producer(self):
        
        val = self.marketplace.register_producer()
        self.assertEqual(val, 0)
        val = self.marketplace.register_producer()
        self.assertEqual(val, 1)

    def test_publish(self):
        
        prod_id = self.marketplace.register_producer()
        for i in range(3):
            val = self.marketplace.publish(prod_id, str("prod_" + str(i)))
            self.assertEqual(val, True)
        val = self.marketplace.publish(id, "some_product")
        self.assertEqual(val, False)

    def test_publish_with_2_publishers(self):
        
        id_1 = self.marketplace.register_producer()


        id_2 = self.marketplace.register_producer()
        self.marketplace.publish(id_1, "prod")
        self.marketplace.publish(id_2, "prod")
        no_of_producers = len(self.marketplace.all_products["prod"])
        self.assertEqual(no_of_producers, 2)

    def test_new_cart(self):
        
        val = self.marketplace.new_cart()
        self.assertEqual(val, 0)
        val = self.marketplace.new_cart()
        self.assertEqual(val, 1)

    def test_add_to_cart(self):
        
        prod_id = self.marketplace.register_producer()
        self.publish_products(prod_id, 2)
        cart_id = self.marketplace.new_cart()
        val = self.marketplace.add_to_cart(cart_id, "prod_1")
        self.assertEqual(val, True)
        val = self.marketplace.add_to_cart(cart_id, "non_existent_prod")
        self.assertEqual(val, False)
        returned_prod_id = self.marketplace.carts[cart_id]["prod_1"][0]
        self.assertEqual(returned_prod_id, prod_id)

    def test_add_to_cart_same_prod_twice(self):
        
        id1 = self.marketplace.register_producer()
        id2 = self.marketplace.register_producer()


        self.marketplace.publish(id1, "prod")
        self.marketplace.publish(id2, "prod")
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, "prod")
        self.marketplace.add_to_cart(cart_id, "prod")
        no_of_producers = len(self.marketplace.carts[cart_id]["prod"])
        self.assertEqual(no_of_producers, 2)

    def test_remove_from_cart(self):
        
        prod_id = self.marketplace.register_producer()
        self.publish_products(prod_id, 2)
        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, "prod_1")

        self.marketplace.remove_from_cart(cart_id, "prod_1")

        self.assertTrue(len(self.marketplace.carts[cart_id]["prod_1"]) == 0)

    def test_place_order(self):
        
        prod_id = self.marketplace.register_producer()
        self.publish_products(prod_id, 3)


        cart_id = self.marketplace.new_cart()
        self.marketplace.add_to_cart(cart_id, "prod_1")
        self.marketplace.add_to_cart(cart_id, "prod_0")
        self.marketplace.add_to_cart(cart_id, "prod_2")
        self.marketplace.remove_from_cart(cart_id, "prod_1")

        products = self.marketplace.place_order(cart_id)
        self.assertTrue(len(products["prod_0"]) == 1)
        self.assertTrue(len(products["prod_1"]) == 0)
        self.assertTrue(len(products["prod_2"]) == 1)
        number_of_products_of_producer = self.marketplace.producers_dict[id]
        self.assertEqual(number_of_products_of_producer, 1)

    def test_get_print_lock(self):
        
        lock = self.marketplace.get_print_lock()
        self.assertTrue(isinstance(lock, type(Lock())))


from threading import Thread
from time import sleep

PRODUCT_POS = 0

NUMBER_OF_PRODUCTS_POS = 1

WAITING_TIME_POS = 2
class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        self.products = products
        self.marketplace = marketplace
        self.wait_time = republish_wait_time
        Thread.__init__(self, **kwargs)


    def run(self):
        prod_id = self.marketplace.register_producer()
        while True:
            for prod in self.products:
                product = prod[PRODUCT_POS]
                no_prods = prod[NUMBER_OF_PRODUCTS_POS]
                pause_time = prod[WAITING_TIME_POS]
                for _ in range(0, no_prods):
                    res = False
                    while res is False:
                        res = self.marketplace.publish(prod_id, product)
                        if res is False:
                            sleep(self.wait_time)
                    sleep(pause_time)


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
