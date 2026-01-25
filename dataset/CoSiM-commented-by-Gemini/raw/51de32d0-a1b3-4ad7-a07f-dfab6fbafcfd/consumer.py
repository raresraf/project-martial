


from threading import Thread
from time import sleep

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def safe_add_to_cart(self, cart_id, product):
        
        while not self.marketplace.add_to_cart(cart_id, product):
            sleep(self.retry_wait_time)

    def run(self):
        for cart in self.carts:
            my_cart = self.marketplace.new_cart()

            for crt_op in cart:
                if crt_op["type"] == "add":
                    for _ in range(crt_op["quantity"]):
                        self.safe_add_to_cart(my_cart, crt_op["product"])
                elif crt_op["type"] == "remove":
                    for _ in range(crt_op["quantity"]):
                        self.marketplace.remove_from_cart(my_cart, crt_op["product"])
                else:
                    print("[Error] No such operation")

            ordered_prods = self.marketplace.place_order(my_cart)
            for prod in ordered_prods:
                self.marketplace.print_lock.acquire()
                print(self.name, "bought", prod)
                self.marketplace.print_lock.release()

import time
import unittest
from threading import Lock
import logging
from logging.handlers import RotatingFileHandler

logging.basicConfig(
    handlers=[RotatingFileHandler('./marketplace.log', maxBytes=2000, backupCount=5)],


    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s [%(funcName)s:%(lineno)d] %(message)s"
)
logging.Formatter.converter = time.gmtime

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.next_cart_id = 0
        self.next_producer_id = 0
        self.producers = {}
        self.carts = {}
        self.producers_stock = {}

        self.producer_reg_lock = Lock()
        self.new_cart_lock = Lock()
        self.products_lock = Lock()
        self.print_lock = Lock()

    def register_producer(self):
        
        logging.info('register_producer was called')
        self.producer_reg_lock.acquire()
        new_producer_id = str(self.next_producer_id)
        self.next_producer_id = self.next_producer_id + 1
        self.producer_reg_lock.release()

        self.producers[new_producer_id] = []
        self.producers_stock[new_producer_id] = 0

        logging.info("register_producer done: id = %s", new_producer_id)
        return new_producer_id

    def publish(self, producer_id, product):
        
        logging.info("publish was called: producer_id = %s; \
                    prod = %s", producer_id, str(product))



        if self.producers_stock[producer_id] >= self.queue_size_per_producer:
            logging.info("publish done")
            return False

        self.producers[producer_id].append(product)
        self.producers_stock[producer_id] = self.producers_stock[producer_id] + 1

        logging.info("publish done")
        return True

    def new_cart(self):
        
        logging.info("new_cart was called")
        self.new_cart_lock.acquire()
        my_id = self.next_cart_id
        self.next_cart_id = self.next_cart_id + 1
        self.new_cart_lock.release()

        self.carts[my_id] = []
        logging.info("new_cart done: id = %s", str(my_id))
        return my_id

    def add_to_cart(self, cart_id, product):
        
        logging.info("add_to_cart was called: cart_id = %s; \
                    prod = %s", str(cart_id), str(product))
        if cart_id not in self.carts:
            return False

        self.products_lock.acquire()
        for producer in self.producers:
            for crt_prod in self.producers[producer]:
                if crt_prod == product:
                    self.carts[cart_id].append((product, producer))
                    self.producers[producer].remove(crt_prod)
                    self.products_lock.release()
                    logging.info("add_to_cart done: True")
                    return True


        self.products_lock.release()
        logging.info("add_to_cart done: False")
        return False

    def remove_from_cart(self, cart_id, product):
        
        logging.info("remove_from_cart was called: cart_id = %s; \
                        product = %s", str(cart_id), str(product))
        if cart_id not in self.carts:
            logging.info("remove_from_cart done")
            return

        for (prod, producer) in self.carts[cart_id]:
            if prod == product:
                self.products_lock.acquire()
                self.producers[producer].append(prod)
                self.products_lock.release()


                self.carts[cart_id].remove((prod, producer))
                break
        logging.info("remove_from_cart done")

    def place_order(self, cart_id):
        
        logging.info("place_order called")
        result = []

        if cart_id not in self.carts:
            logging.info("place_order done: %s", result)
            return result

        for (prod, producer) in self.carts[cart_id]:
            result.append(prod)
            self.products_lock.acquire()
            self.producers_stock[producer] = self.producers_stock[producer] - 1
            self.products_lock.release()

        logging.info("place_order done: %s", result)
        return result

class TestMarketplace(unittest.TestCase):
    
    def setUp(self):
        
        self.marketplace = Marketplace(2)

    def test_register_producer(self):
        
        for i in range(1000):
            self.assertEqual(self.marketplace.register_producer(), str(i))

    def test_publish(self):
        
        self.marketplace.register_producer()
        for i in range(2):
            self.assertEqual(self.marketplace.publish("0", str(i)), True)
        for i in range(2):
            self.assertEqual(self.marketplace.publish("0", str(i)), False)

        self.marketplace.register_producer()
        for i in range(2):
            self.assertEqual(self.marketplace.publish("1", str(i)), True)
        for i in range(2):
            self.assertEqual(self.marketplace.publish("1", str(i)), False)

    def test_new_cart(self):
        
        for i in range(1000):
            self.assertEqual(self.marketplace.new_cart(), i)

    def test_add_to_cart(self):
        
        for i in range(10):
            self.assertEqual(self.marketplace.new_cart(), i)
            self.assertEqual(self.marketplace.register_producer(), str(i))
        for i in range(10):
            self.assertEqual(self.marketplace.publish(str(i), str(i + 1000)), True)
        for i in range(10):
            self.assertEqual(self.marketplace.add_to_cart(i, str(i + 1000)), True)
        for i in range(10):
            self.assertEqual(self.marketplace.add_to_cart(i, str(i + 1000)), False)

    def test_remove_from_cart(self):
        
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertEqual(self.marketplace.register_producer(), "0")
        self.assertEqual(self.marketplace.publish("0", "00"), True)
        self.assertEqual(self.marketplace.publish("0", "01"), True)
        self.assertEqual(self.marketplace.add_to_cart(0, "00"), True)
        self.assertEqual(self.marketplace.add_to_cart(0, "01"), True)
        self.assertEqual(self.marketplace.add_to_cart(0, "00"), False)
        self.marketplace.remove_from_cart(0, "00")
        self.assertEqual(self.marketplace.add_to_cart(0, "00"), True)

    def test_place_order(self):
        
        self.assertEqual(self.marketplace.new_cart(), 0)
        self.assertEqual(self.marketplace.register_producer(), "0")
        self.assertEqual(self.marketplace.publish("0", "00"), True)
        self.assertEqual(self.marketplace.publish("0", "01"), True)
        self.assertEqual(self.marketplace.add_to_cart(0, "00"), True)
        self.assertEqual(self.marketplace.add_to_cart(0, "01"), True)
        self.marketplace.remove_from_cart(0, "01")
        self.marketplace.remove_from_cart(0, "02")
        self.assertEqual(self.marketplace.add_to_cart(0, "00"), False)
        self.marketplace.remove_from_cart(0, "00")
        self.assertEqual(self.marketplace.add_to_cart(0, "01"), True)
        self.assertEqual(self.marketplace.add_to_cart(0, "00"), True)
        self.assertEqual(self.marketplace.place_order(0), ["01", "00"])


from threading import Thread
from time import sleep


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def safe_publish(self, product, producer_id):
        
        while not self.marketplace.publish(producer_id, product):
            sleep(self.republish_wait_time)

    def run(self):
        my_id = self.marketplace.register_producer()

        while True:
            for (id_prod, quantity_prod, wait_time_prod) in self.products:
                sleep(wait_time_prod)

                for _ in range(quantity_prod):
                    self.safe_publish(id_prod, my_id)


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
