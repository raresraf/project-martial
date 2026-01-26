


from threading import Thread
from time import sleep

ADD_OPTION = "add"
NAME_ARG = "name"
QUANTITY = "quantity"
PRODUCT_ARG = "product"
TYPE = "type"


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        

        super().__init__(**kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.cons_name = kwargs[NAME_ARG]

    def run(self):
        cons_id = self.marketplace.new_cart()

        for entries in self.carts:
            for cart in entries:
                for _ in range(cart[QUANTITY]):
                    if cart[TYPE] == ADD_OPTION:
                        while True:
                            done = self.marketplace.add_to_cart(cons_id, cart[PRODUCT_ARG])

                            if done:
                                break

                            
                            sleep(self.retry_wait_time)
                    else:
                        self.marketplace.remove_from_cart(cons_id, cart[PRODUCT_ARG])

            groceries = self.marketplace.place_order(cons_id)

            for product in groceries:
                print(f"{self.cons_name} bought {product}")


from threading import Lock
import logging
from logging.handlers import RotatingFileHandler
import time
import unittest
import tema.product as p


MARKETPLACE_LOGGER = logging.getLogger("market_logger")

MARKETPLACE_LOGGER.setLevel(logging.INFO)

HANDLER = RotatingFileHandler("marketplace.log", maxBytes=100000, backupCount=16)

FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
FORMATTER.converter = time.gmtime

HANDLER.setFormatter(FORMATTER)
MARKETPLACE_LOGGER.addHandler(HANDLER)

TEST_QUEUE_SIZE = 3


class ProductEntry:
    

    def __init__(self, product, producer_id):
        

        self.product = product
        self.producer_id = producer_id
        self.is_available = True

    def __str__(self):
        return f"prod = {self.product}, id = {self.producer_id}, av = {self.is_available}"


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        

        MARKETPLACE_LOGGER.info("start __init__ with args: %d", queue_size_per_producer)

        self.queue_size = queue_size_per_producer
        self.producers = {}
        self.consumers = {}
        self.curr_prod_id = 0
        self.curr_cons_id = 0
        self.reg_prod_lock = Lock()
        self.reg_cart_lock = Lock()
        self.producer_locks = {}



        MARKETPLACE_LOGGER.info("exit __init__")

    def register_producer(self):
        

        MARKETPLACE_LOGGER.info("start register_producer")

        
        
        with self.reg_prod_lock:
            new_prod_id = self.curr_prod_id
            self.producers[new_prod_id] = []
            self.producer_locks[new_prod_id] = Lock()
            self.curr_prod_id += 1

            MARKETPLACE_LOGGER.info("exit register_producer")

            return new_prod_id

    def publish(self, producer_id, product):
        

        MARKETPLACE_LOGGER.info("start publish with args: %d; %s", producer_id, product)

        if len(self.producers[producer_id]) < self.queue_size:
            self.producers[producer_id].append(ProductEntry(product, producer_id))

            MARKETPLACE_LOGGER.info("exit publish")

            return True

        MARKETPLACE_LOGGER.info("exit publish")

        return False

    def new_cart(self):
        

        MARKETPLACE_LOGGER.info("start new_cart")

        
        
        with self.reg_cart_lock:
            new_cart_id = self.curr_cons_id
            self.consumers[new_cart_id] = []
            self.curr_cons_id += 1

            MARKETPLACE_LOGGER.info("exit new_cart")

            return new_cart_id

    def add_to_cart(self, cart_id, product):
        

        MARKETPLACE_LOGGER.info("start add_to_cart with args: %id; %s", cart_id, product)

        for id_producer, product_entries in self.producers.items():
            
            with self.producer_locks[id_producer]:
                for product_entry in product_entries:
                    if product == product_entry.product and product_entry.is_available:
                        product_entry.is_available = False
                        self.consumers[cart_id].append(product_entry)
                        return True

        MARKETPLACE_LOGGER.info("exit add_to_cart")

        return False

    def remove_from_cart(self, cart_id, product):
        

        MARKETPLACE_LOGGER.info("start remove_from_cart with args: %d; %s", cart_id, product)

        to_remove = None

        for product_entry in self.consumers[cart_id]:
            if product == product_entry.product:
                
                with self.producer_locks[product_entry.producer_id]:
                    to_remove = product_entry
                    product_entry.is_available = True
                    break



        self.consumers[cart_id].remove(to_remove)

        MARKETPLACE_LOGGER.info("exit remove_from_cart")

    def place_order(self, cart_id):
        

        MARKETPLACE_LOGGER.info("start place_order with args: %d", cart_id)

        
        groceries = []

        for product_entry in self.consumers[cart_id]:
            
            with self.producer_locks[product_entry.producer_id]:
                groceries.append(product_entry.product)
                self.producers[product_entry.producer_id].remove(product_entry)

        
        self.consumers[cart_id].clear()

        MARKETPLACE_LOGGER.info("exit place_order")

        return groceries


class TestMarketplace(unittest.TestCase):
    

    def setUp(self):
        self.marketplace = Marketplace(TEST_QUEUE_SIZE)
        self.coffee_product_0 = p.Coffee("coffee0", 0, "0", "l")
        self.coffee_product_1 = p.Coffee("coffee1", 1, "1", "m")
        self.coffee_product_2 = p.Coffee("coffee2", 2, "2", "h")
        self.tea_product_0 = p.Tea("tea0", 3, "t0")
        self.tea_product_1 = p.Tea("tea1", 4, "t1")
        self.producer_0 = self.marketplace.register_producer()
        self.producer_1 = self.marketplace.register_producer()
        self.consumer_0 = self.marketplace.new_cart()
        self.consumer_1 = self.marketplace.new_cart()

    def test_register_producer(self):
        

        self.assertEqual(0, self.producer_0, "wrong id for producer_0")
        self.assertEqual(1, self.producer_1, "wrong id for producer_1")

        id_test = []

        for _ in range(1024):
            id_test.append(self.marketplace.register_producer())

        for i in range(1024):
            self.assertEqual(i + 2, id_test[i], f"wrong id: got {id_test[i]}, should be {i + 2}")

    def test_publish(self):
        

        
        result = self.marketplace.publish(self.producer_0, self.coffee_product_0)
        self.assertEqual(result, True, "should be True")
        self.assertEqual(self.marketplace.producers[self.producer_0][0].product,
                         self.coffee_product_0,
                         "wrong product")

        
        result = self.marketplace.publish(self.producer_0, self.coffee_product_1)
        self.assertEqual(result, True, "should be True")
        self.assertEqual(self.marketplace.producers[self.producer_0][1].product,
                         self.coffee_product_1,
                         "wrong product")

        result = self.marketplace.publish(self.producer_0, self.coffee_product_2)
        self.assertEqual(result, True, "should be True")
        self.assertEqual(self.marketplace.producers[self.producer_0][2].product,
                         self.coffee_product_2,
                         "wrong product")

        
        result = self.marketplace.publish(self.producer_0, self.tea_product_0)
        self.assertEqual(result, False, "should be False")

    def test_new_cart(self):
        

        self.assertEqual(0, self.consumer_0, "wrong id for consumer_0")
        self.assertEqual(1, self.consumer_1, "wrong id for consumer_1")

        id_test = []

        for _ in range(1024):
            id_test.append(self.marketplace.new_cart())

        for i in range(1024):
            self.assertEqual(i + 2, id_test[i], f"wrong id: got {id_test[i]}, should be {i + 2}")

    def test_add_to_cart(self):
        

        
        self.marketplace.publish(self.producer_1, self.tea_product_1)
        result = self.marketplace.add_to_cart(self.consumer_0, self.tea_product_1)
        self.assertEqual(result, True, "should be True")

        
        result = self.marketplace.add_to_cart(self.consumer_0, self.coffee_product_0)
        self.assertEqual(result, False, "should be False")

        
        result = self.marketplace.add_to_cart(self.consumer_1, self.tea_product_1)
        self.assertEqual(result, False, "should be False")

        
        self.marketplace.publish(self.producer_0, self.coffee_product_0)
        self.marketplace.publish(self.producer_1, self.coffee_product_1)
        self.marketplace.publish(self.producer_1, self.coffee_product_2)
        result = self.marketplace.add_to_cart(self.consumer_1, self.coffee_product_0)
        self.assertEqual(result, True, "should be True")
        result = self.marketplace.add_to_cart(self.consumer_1, self.coffee_product_2)
        self.assertEqual(result, True, "should be True")
        result = self.marketplace.add_to_cart(self.consumer_1, self.coffee_product_1)
        self.assertEqual(result, True, "should be True")
        self.assertEqual(3,
                         len(self.marketplace.consumers[self.consumer_1]),
                         "wrong number of products in cart")

    def test_remove_from_cart(self):
        

        
        self.marketplace.publish(self.producer_0, self.coffee_product_0)
        self.marketplace.add_to_cart(self.consumer_0, self.coffee_product_0)
        self.marketplace.remove_from_cart(self.consumer_0, self.coffee_product_0)
        self.assertEqual(0,
                         len(self.marketplace.consumers[self.consumer_0]),
                         "wrong number of products in cart")

        
        self.marketplace.publish(self.producer_1, self.tea_product_0)
        self.marketplace.add_to_cart(self.consumer_0, self.tea_product_0)
        result = self.marketplace.add_to_cart(self.consumer_1, self.tea_product_0)
        self.assertEqual(result, False, "should be False")
        self.marketplace.remove_from_cart(self.consumer_0, self.tea_product_0)
        result = self.marketplace.add_to_cart(self.consumer_1, self.tea_product_0)
        self.assertEqual(result, True, "should be True")

    def test_place_order(self):
        

        self.marketplace.publish(self.producer_0, self.tea_product_0)
        self.marketplace.publish(self.producer_1, self.coffee_product_1)
        self.marketplace.publish(self.producer_1, self.coffee_product_2)
        self.marketplace.publish(self.producer_0, self.coffee_product_2)
        self.marketplace.publish(self.producer_1, self.tea_product_1)
        self.marketplace.add_to_cart(self.consumer_0, self.coffee_product_2)
        self.marketplace.add_to_cart(self.consumer_0, self.coffee_product_2)
        self.marketplace.add_to_cart(self.consumer_0, self.coffee_product_1)
        self.marketplace.add_to_cart(self.consumer_0, self.tea_product_1)
        self.marketplace.remove_from_cart(self.consumer_0, self.coffee_product_1)
        self.marketplace.add_to_cart(self.consumer_0, self.tea_product_0)

        
        self.assertEqual(4,
                         len(self.marketplace.consumers[self.consumer_0]),
                         "wrong number of products in cart")

        groceries_list = self.marketplace.place_order(self.consumer_0)

        
        self.assertEqual(groceries_list,
                         [self.coffee_product_2, self.coffee_product_2,
                          self.tea_product_1, self.tea_product_0],
                         "wrong products in cart")

        
        self.assertEqual(0, len(self.marketplace.producers[self.consumer_0]), "should be empty")


from threading import Thread
from time import sleep

PRODUCTS = 0
QUANTITY = 1
PRODUCE_TIME = 2


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        

        super().__init__(**kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        prod_id = self.marketplace.register_producer()

        while True:
            for product in self.products:

                for _ in range(product[QUANTITY]):
                    while True:
                        done = self.marketplace.publish(prod_id, product[PRODUCTS])

                        if done:
                            break

                        
                        sleep(self.republish_wait_time)

                    
                    sleep(product[PRODUCE_TIME])


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
