


from threading import Thread
import time

class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        
        for cart in self.carts:
            
            cart_id = self.marketplace.new_cart()
            
            for item in cart:
                
                if item["type"] == "add":
                    
                    for _ in range(item["quantity"]):
                        
                        while True:
                            res = self.marketplace.add_to_cart(cart_id, item["product"])
                            if res is False:
                                time.sleep(self.retry_wait_time)
                            else:
                                break
                
                elif item["type"] == "remove":
                    
                    for _ in range(item["quantity"]):
                        self.marketplace.remove_from_cart(cart_id, item["product"])
            
            result_cart = self.marketplace.place_order(cart_id)

            
            for item in result_cart:
                print(f'{self.name} bought {item}')

from random import shuffle
from threading import Lock
import unittest
import logging
from logging.handlers import RotatingFileHandler
import time
from tema.product import Coffee, Tea

class Marketplace:
    
    def __init__(self, queue_size_per_producer):
        
        
        self.producer_id = -1
        self.cart_id = -1

        self.queue_size_per_producer = queue_size_per_producer

        
        self.products = []

        
        self.locks = []

        
        self.carts = []

        
        self.producer_lock = Lock()
        self.cart_lock = Lock()

        self.logger = logging.getLogger('marketplace-logger')

        
        self.logger.setLevel(logging.INFO)

        
        formater = logging.Formatter("[%(levelname)s] - %(asctime)s: %(message)s")

        
        handler = RotatingFileHandler("marketplace.log", maxBytes=5000, backupCount=10)
        handler.setFormatter(formater)
        logging.Formatter.converter = time.gmtime
        self.logger.addHandler(handler)

        self.logger.info("Marketplace created with limit: %s", queue_size_per_producer)

    def register_producer(self):
        
        with self.producer_lock:
            
            self.producer_id += 1

            
            self.products.append([])

            
            self.locks.append(Lock())

            self.logger.info("New producer registered: Producer%s", self.producer_id)
            return self.producer_id

    def publish(self, producer_id, product):
        
        self.logger.info("Producer%s tries to publish new product: %s", producer_id, product)
        
        if (producer_id > self.producer_id or producer_id < 0):
            self.logger.error("Bad id for Producer%s", producer_id)
            return False

        
        if len(self.products[producer_id]) == self.queue_size_per_producer:
            self.logger.info("Max size reached by Producer%s", producer_id)
            return False

        
        self.products[producer_id].append([product, -1])
        self.logger.info("Producer%s published %s", producer_id, product)
        return True

    def new_cart(self):
        
        with self.cart_lock:
            
            self.carts.append([])
            self.cart_id += 1

            self.logger.info("New cart registered: Cart%s", self.cart_id)
            return self.cart_id

    def add_to_cart(self, cart_id, product):
        
        self.logger.info("Cart%s tries to add new product: %s", cart_id, product)
        
        if (cart_id > self.cart_id or cart_id < 0):
            self.logger.error("Bad id for Cart%s", cart_id)
            return False

        
        shuffled_indexes = list(range(len(self.products)))
        shuffle(shuffled_indexes)

        
        for producer_id in shuffled_indexes:
            for producer_product in self.products[producer_id]:
                with self.locks[producer_id]:
                    
                    if producer_product[0] == product and producer_product[1] == -1:
                        
                        self.carts[cart_id].append([producer_product[0], producer_id])
                        
                        producer_product[1] = cart_id
                        self.logger.info("Cart%s added %s", cart_id, product)
                        return True

        
        self.logger.info("Product %s not found", product)
        return False

    def remove_from_cart(self, cart_id, product):
        
        self.logger.info("Cart%s tries to remove product: %s", cart_id, product)
        
        if (cart_id > self.cart_id or cart_id < 0):
            self.logger.error("Bad id for Cart%s", cart_id)
            return False

        product_information = []
        
        for product_info in self.carts[cart_id]:
            if product_info[0] == product:
                product_information = product_info

        
        if product_information == []:
            self.logger.error("Product %s not found", product)
            return False

        
        with self.locks[product_information[1]]:
            for product_info in self.products[product_information[1]]:
                if product_info[0] == product and product_info[1] == cart_id:
                    
                    product_info[1] = -1
                    break

        
        self.carts[cart_id].remove(product_information)
        self.logger.info("Cart%s removed %s", cart_id, product)
        return True

    def place_order(self, cart_id):
        
        self.logger.info("Cart%s tries to place order", cart_id)
        
        if (cart_id > self.cart_id or cart_id < 0):
            self.logger.error("Bad id for Cart%s", cart_id)
            return None

        result = []
        
        for product_info in self.carts[cart_id]:
            [product, producer_id] = product_info
            with self.locks[producer_id]:
                
                for idx, producer_product in enumerate(self.products[producer_id]):
                    if product == producer_product[0] and producer_product[1] == cart_id:
                        
                        del self.products[producer_id][idx]
                        result.append(product)
                        break
        self.logger.info("Cart%s placed order", cart_id)
        return result

class TestMarketplace(unittest.TestCase):
    
    def setUp(self):
        
        self.marketplace = Marketplace(3)
        self.products = [Tea("Linden", "Herbal", 9), Coffee("Indonezia", 5.05, "MEDIUM", 1)]

    def test_register_producer(self):
        
        for i in range(4):
            self.assertEqual(self.marketplace.register_producer(), i)

    def test_publish(self):
        
        
        self.assertEqual(self.marketplace.publish(-1, self.products[0]), False)
        self.assertEqual(self.marketplace.publish(10, self.products[0]), False)

        self.marketplace.register_producer()


        self.assertEqual(self.marketplace.publish(0, self.products[0]), True)
        self.assertEqual(self.marketplace.publish(0, self.products[0]), True)
        self.assertEqual(self.marketplace.publish(0, self.products[0]), True)


        self.assertEqual(self.marketplace.publish(0, self.products[0]), False)

    def test_new_cart(self):
        
        for i in range(4):
            self.assertEqual(self.marketplace.new_cart(), i)

    def test_add_to_cart(self):
        
        cart_id = self.marketplace.new_cart()
        
        self.assertEqual(self.marketplace.add_to_cart(-1, self.products[0]), False)
        self.assertEqual(self.marketplace.add_to_cart(2, self.products[0]), False)
        self.assertEqual(self.marketplace.add_to_cart(cart_id, self.products[0]), False)

        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, self.products[0])

        
        self.assertEqual(self.marketplace.add_to_cart(cart_id, self.products[1]), False)

        self.assertEqual(self.marketplace.add_to_cart(cart_id, self.products[0]), True)

        
        self.assertEqual(self.marketplace.publish(0, self.products[0]), True)


        self.assertEqual(self.marketplace.publish(0, self.products[0]), True)
        self.assertEqual(self.marketplace.publish(0, self.products[0]), False)

    def test_remove_from_cart(self):
        
        cart_id = self.marketplace.new_cart()
        
        self.assertEqual(self.marketplace.remove_from_cart(-1, self.products[0]), False)
        self.assertEqual(self.marketplace.remove_from_cart(2, self.products[0]), False)
        self.assertEqual(self.marketplace.remove_from_cart(cart_id, self.products[0]), False)

        
        self.assertEqual(self.marketplace.remove_from_cart(cart_id, self.products[1]), False)

        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, self.products[0])
        self.marketplace.add_to_cart(cart_id, self.products[0])
        self.assertEqual(self.marketplace.remove_from_cart(cart_id, self.products[0]), True)

    def test_place_order(self):
        
        cart_id = self.marketplace.new_cart()
        
        self.assertEqual(self.marketplace.place_order(-1), None)
        self.assertEqual(self.marketplace.place_order(2), None)
        self.assertEqual(self.marketplace.place_order(cart_id), [])

        producer_id = self.marketplace.register_producer()
        self.marketplace.publish(producer_id, self.products[0])
        self.marketplace.publish(producer_id, self.products[0])
        self.marketplace.publish(producer_id, self.products[1])

        self.marketplace.add_to_cart(cart_id, self.products[0])
        self.marketplace.add_to_cart(cart_id, self.products[0])
        cart = self.marketplace.place_order(cart_id)
        self.assertEqual(cart, [self.products[0], self.products[0]])

        
        self.assertEqual(self.marketplace.publish(cart_id, self.products[1]), True)
        self.assertEqual(self.marketplace.publish(cart_id, self.products[1]), True)
        self.assertEqual(self.marketplace.publish(cart_id, self.products[1]), False)



        self.marketplace.add_to_cart(0, self.products[1])
        
        self.marketplace.add_to_cart(0, self.products[0])
        self.assertEqual(self.marketplace.publish(cart_id, self.products[1]), False)
        self.marketplace.remove_from_cart(0, self.products[0])
        self.assertEqual(self.marketplace.publish(cart_id, self.products[1]), False)

        cart = self.marketplace.place_order(cart_id)
        self.assertEqual(cart, [self.products[1]])

        
        self.assertEqual(self.marketplace.publish(cart_id, self.products[1]), True)
        self.assertEqual(self.marketplace.publish(cart_id, self.products[1]), False)


from threading import Thread
import time

class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        
        producer_id = self.marketplace.register_producer()

        
        while True:
            
            for product in self.products:
                for _ in range(product[1]):
                    res = self.marketplace.publish(producer_id, product[0])

                    
                    if res is False:
                        time.sleep(self.republish_wait_time)
                    else:
                        
                        time.sleep(product[2])


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
