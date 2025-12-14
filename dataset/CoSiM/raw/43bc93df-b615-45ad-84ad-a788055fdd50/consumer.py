


from threading import Thread
import time

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def add_operation(self, quantity, cart_id, product):
        
        for _ in range(quantity):
            while not self.marketplace.add_to_cart(cart_id, product):
                time.sleep(self.retry_wait_time)

    def remove_operation(self, quantity, cart_id, product):
        
        for _ in range(quantity):
            self.marketplace.remove_from_cart(cart_id, product)

    def run(self):
        for cart in self.carts:
            id_cart = self.marketplace.new_cart()
            for operation in cart:
                if operation["type"] == "add":
                    self.add_operation(operation["quantity"], id_cart, operation["product"])
                else:
                    self.remove_operation(operation["quantity"], id_cart, operation["product"])

            self.marketplace.place_order(id_cart)


from threading import Lock, currentThread
import unittest
import logging
import time
from logging.handlers import RotatingFileHandler
from tema.product import Coffee, Tea

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

FORMATTER = logging.Formatter('$asctime : $levelname : $name : $message', style='$')
HANDLER = RotatingFileHandler('marketplace.log', maxBytes=20000, backupCount=5)
HANDLER.setFormatter(FORMATTER)

FORMATTER.converter = time.gmtime

LOGGER.addHandler(HANDLER)


class TestMarketplace(unittest.TestCase):
    
    def setUp(self):
        
        self.queue_size_per_producer = 15
        self.marketplace = Marketplace(self.queue_size_per_producer)
        self.marketplace.producer_id = 3
        self.marketplace.cart_id = 5
        self.products = [Coffee(name='Indonezia', price=1, acidity=5.05, roast_level='MEDIUM'),
                         Tea(name='Linden', price=9, type='Herbal'),
                         Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM'),
                         Tea(name='Wild Cherry', price=5, type='Black'),
                         Tea(name='Cactus fig', price=3, type='Green'),
                         Coffee(name='Ethiopia', price=10, acidity=5.09, roast_level='MEDIUM')]


    def test_register_producer(self):
        
        self.assertIsNotNone(self.marketplace.register_producer())
        self.assertEqual(self.marketplace.register_producer(), 5)

    def test_publish(self):
        
        producer_id = self.marketplace.register_producer()
        dict1 = {producer_id: {self.products[0]: 2}}
        self.marketplace.products[producer_id][self.products[0]] = 1

        self.assertTrue(self.marketplace.publish(producer_id, self.products[0]))
        self.assertDictEqual(self.marketplace.products, dict1)

    def test_new_cart(self):
        
        self.assertIsNotNone(self.marketplace.new_cart())
        self.assertEqual(self.marketplace.new_cart(), 7)

    def test_add_to_cart(self):
        
        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, self.products[0])
        self.marketplace.publish(producer_id, self.products[1])
        cart_id = self.marketplace.new_cart()
        carts = {cart_id: {(self.products[1], producer_id): 1}}

        self.assertFalse(self.marketplace.add_to_cart(cart_id, self.products[3]))
        self.assertTrue(self.marketplace.add_to_cart(cart_id, self.products[1]))
        self.assertDictEqual(self.marketplace.carts[cart_id], carts[cart_id])

    def test_remove_from_cart(self):
        
        producer_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()

        self.marketplace.publish(producer_id, self.products[0])
        self.marketplace.publish(producer_id, self.products[1])
        self.marketplace.publish(producer_id, self.products[1])

        self.marketplace.add_to_cart(cart_id, self.products[0])
        self.marketplace.add_to_cart(cart_id, self.products[1])
        self.marketplace.add_to_cart(cart_id, self.products[1])
        self.marketplace.remove_from_cart(cart_id, self.products[0])
        carts = {cart_id: {(self.products[1], producer_id): 2}}

        self.assertDictEqual(self.marketplace.carts[cart_id], carts[cart_id])

    def test_place_order(self):
        
        producer_id = self.marketplace.register_producer()
        cart_id = self.marketplace.new_cart()

        self.marketplace.publish(producer_id, self.products[0])
        self.marketplace.publish(producer_id, self.products[1])
        self.marketplace.publish(producer_id, self.products[1])

        self.marketplace.add_to_cart(cart_id, self.products[0])
        self.marketplace.add_to_cart(cart_id, self.products[1])
        self.marketplace.add_to_cart(cart_id, self.products[1])
        self.marketplace.remove_from_cart(cart_id, self.products[0])
        self.marketplace.place_order(cart_id)

        self.assertEqual(self.marketplace.products_published[producer_id], 1)

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        self.queue_size_per_producer = queue_size_per_producer
        self.consumer_lock = Lock()
        self.producer_lock = Lock()
        self.products = {} 
        self.carts = {} 
        self.products_published = {} 
        self.producer_id = -1
        self.cart_id = -1

    def register_producer(self):
        
        with self.producer_lock:
            LOGGER.info('[OLD]Last producer id:%d', self.producer_id)
            self.producer_id += 1
            self.products_published[self.producer_id] = 0
            self.products[self.producer_id] = {}
            LOGGER.info('[UPDATE]New producer id:%d', self.producer_id)
            return self.producer_id

    def publish(self, producer_id, product):
        
        with self.producer_lock:
            LOGGER.info('[INPUT]Producer_id: %s and Product: %s', producer_id, product)

            res = False
            
            if self.products_published[producer_id] <= self.queue_size_per_producer:
                self.products_published[producer_id] += 1
                if product in list(self.products[producer_id].keys()):
                    self.products[producer_id][product] += 1
                else:
                    self.products[producer_id][product] = 1
                res = True

            LOGGER.info('[OUTPUt]Method returns: %r', res)
            return res

    def new_cart(self):
        
        with self.consumer_lock:
            LOGGER.info('[OLD]:Cart_id: %d', self.cart_id)
            self.cart_id += 1
            self.carts[self.cart_id] = {}
            LOGGER.info('[UPDATE]:Cart_id: %d', self.cart_id)
            return self.cart_id

    def add_to_cart(self, cart_id, product):
        
        with self.consumer_lock:
            LOGGER.info('[INPUT]Cart_id: %d and Product: %s', cart_id, product)

            for producer in list(self.products.keys()):
                if product in list(self.products[producer].keys()):
                    self.products[producer][product] -= 1
                    if (product, producer) in list(self.carts[cart_id].keys()):
                        self.carts[cart_id][(product, producer)] += 1
                    else:
                        self.carts[cart_id][(product, producer)] = 1

                    if self.products[producer][product] == 0:
                        self.products[producer].pop(product, 0)
                    LOGGER.info('[OUTPUT]Method returns: True')
                    return True

        LOGGER.info('[OUTPUT]Method returns: False')
        return False

    def remove_from_cart(self, cart_id, product):
        
        with self.consumer_lock:
            LOGGER.info('[INPUT]Cart_id: %d and Product: %s', cart_id, product)
            for (prod, producer_id) in list(self.carts[cart_id].keys()):
                if prod == product:
                    self.carts[cart_id][(product, producer_id)] -= 1
                    if self.carts[cart_id][(product, producer_id)] == 0:
                        self.carts[cart_id].pop((product, producer_id), 0)

                    if product in list(self.products[producer_id].keys()):
                        self.products[producer_id][product] += 1
                    else:
                        self.products[producer_id][product] = 1
                    break

    def place_order(self, cart_id):
        
        with self.consumer_lock:
            LOGGER.info('[INPUT]Place order for cart_id: %d', cart_id)
            for (product, producer), quantity in self.carts[cart_id].items():
                for _ in range(quantity):
                    self.products_published[producer] -= 1
                    print(f"{currentThread().getName()} bought {product}")
            LOGGER.info('[OUTPUT]The cart was printed: %d', cart_id)



from threading import Thread
import time

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()
        self.kwargs = kwargs

    def run(self):


        while True:
            for product, quantity, prod_time in self.products:
                for _ in range(quantity):
                    result = self.marketplace.publish(self.producer_id, product)

                    if result is True:
                        time.sleep(prod_time)
                    else:
                        while not self.marketplace.publish(self.producer_id, product):
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
