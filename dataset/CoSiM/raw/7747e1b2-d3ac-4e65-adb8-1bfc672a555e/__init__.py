


from threading import Thread
import time
import sys


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        
        
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def run(self):
        for cart in self.carts:
            
            cart_id = self.marketplace.new_cart()

            for action in cart:
                
                action_type = action['type']
                product = action['product']
                quantity = action['quantity']

                
                if action_type == "add":
                    
                    
                    for _ in range(quantity):
                        while not self.marketplace.add_to_cart(cart_id, product):
                            
                            
                            time.sleep(self.retry_wait_time)

                    
                else:
                    
                    for _ in range(quantity):
                        self.marketplace.remove_from_cart(cart_id, product)

            
            for order in self.marketplace.place_order(cart_id):
                sys.stdout.flush()
                print(f"{self.name} bought {order}")


from threading import Lock
import time
import logging
from logging.handlers import RotatingFileHandler
from typing import Counter
import unittest
from tema.product import Coffee, Tea


def set_logger():
    

    
    formatter = logging.Formatter(
        '[%(asctime)s] --> %(levelname)s: %(message)s')
    
    formatter.converter = time.gmtime

    
    handler = RotatingFileHandler(
        'marketplace.log', maxBytes=100000, backupCount=10)
    handler.setFormatter(formatter)

    
    logger = logging.getLogger('marketplace info logger')
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger


class Marketplace:
    

    
    logger = set_logger()

    def __init__(self, queue_size_per_producer):
        
        self.logger.info(
            "Marketplace initialized with maximum queue size: %s.", queue_size_per_producer)

        
        self.queue_size_per_producer = queue_size_per_producer

        
        self.producer_number = 0
        
        self.consumer_cart_number = 0

        
        
        self.producers_products = {}
        
        
        self.consumers_carts = {}
        
        
        self.products_queue = {}

        
        self.producer_lock = Lock()
        self.consumer_lock = Lock()
        self.queue_lock = Lock()



        self.logger.info("Initiated marketplace parameters.")

    def register_producer(self):
        
        self.logger.info("A new producer tries to obtain an id.")

        
        with self.producer_lock:
            producer_id = self.producer_number
            self.producer_number = self.producer_number + 1

        
        self.producers_products[producer_id] = 0

        self.logger.info(
            "Generated the producer id number %s.", producer_id)

        return str(producer_id)

    def publish(self, producer_id, product):
        
        self.logger.info(
            "Producer with id %s wants to publish %s.", producer_id, product)

        producer_id = int(producer_id)

        
        if self.producers_products[producer_id] >= self.queue_size_per_producer:

            
            self.logger.info(
                , producer_id, product)
            return False

        
        self.producers_products[producer_id] += 1

        with self.queue_lock:
            
            
            if not product in self.products_queue:
                self.products_queue[product] = []

            
            self.products_queue[product].append((producer_id, True))

        self.logger.info(
            "Producer with id %s published %s.", producer_id, product)

        
        return True

    def new_cart(self):
        
        self.logger.info("A consumer tries to obtain a new cart id.")

        
        with self.consumer_lock:
            consumer_cart_id = self.consumer_cart_number
            self.consumer_cart_number = self.consumer_cart_number + 1

        
        self.consumers_carts[consumer_cart_id] = {}

        self.logger.info(
            "Generated the cart id number %s.", consumer_cart_id)

        return consumer_cart_id

    def add_to_cart(self, cart_id, product):
        
        self.logger.info(
            "Consumer with cart id %s wants to add %s.", cart_id, product)

        with self.queue_lock:
            
            if not product in self.products_queue:

                
                self.logger.info(, cart_id, product)
                return False

            for index in range(len(self.products_queue[product])):
                
                product_queue = self.products_queue[product][index]

                
                
                if product_queue[1] is True:

                    
                    self.products_queue[product][index] = (
                        product_queue[0], False)

                    
                    
                    if not product in self.consumers_carts[cart_id]:
                        self.consumers_carts[cart_id][product] = []

                    
                    self.consumers_carts[cart_id][product].append(index)

                    
                    
                    with self.producer_lock:
                        self.producers_products[self.products_queue[product]
                                                [index][0]] -= 1

                    self.logger.info(
                        "Consumer with cart id %s added %s.", cart_id, product)

                    
                    return True

        self.logger.info(, cart_id, product)

        
        return False

    def remove_from_cart(self, cart_id, product):
        
        self.logger.info(
            "Consumer with cart id %s wants to remove %s.", cart_id, product)

        with self.queue_lock:
            
            if len(self.consumers_carts[cart_id][product]) == 0:
                raise Exception("No product to be removed from cart")

            
            index = self.consumers_carts[cart_id][product].pop()

            product_queue = self.products_queue[product][index]
            
            self.products_queue[product][index] = (product_queue[0], True)

            
            
            with self.producer_lock:
                self.producers_products[self.products_queue[product]
                                        [index][0]] += 1

        self.logger.info(
            "Consumer with cart id %s removed %s.", cart_id, product)

    def place_order(self, cart_id):
        
        self.logger.info(
            "Consumer with cart id %s wants to place order.", cart_id)

        order = []

        consumer_cart = self.consumers_carts[cart_id]
        
        for product in consumer_cart.keys():
            for _ in consumer_cart[product]:
                order.append(product)

        
        self.consumers_carts[cart_id] = {}

        self.logger.info(
            "Consumer with cart id %s placed order: %s.", cart_id, order)

        return order


class TestMarketplace(unittest.TestCase):
    

    def setUp(self):
        
        
        queue_size_per_producer = 5
        
        
        self.marketplace = Marketplace(queue_size_per_producer)

        
        self.teas = []
        self.teas.append(Tea("Tabiets", 5, "Black"))
        self.teas.append(Tea("Aroma Tea", 7.5, "Mint"))
        self.teas.append(Tea("Honey", 5, "Green"))

        
        self.coffees = []
        self.coffees.append(Coffee("Davidoff", 10, 4.5, "STRONG"))
        self.coffees.append(Coffee("Romantique", 6, 3.0, "MILD"))
        self.coffees.append(Coffee("Costa", 8, 5.0, "EXTRA STRONG"))

    def test_register_producer(self):
        
        
        self.assertEqual(self.marketplace.register_producer(),
                         '0', "Incorrect producer id.")
        self.assertEqual(self.marketplace.register_producer(),
                         '1', "Incorrect producer id.")
        self.assertEqual(self.marketplace.register_producer(),
                         '2', "Incorrect producer id.")
        self.assertEqual(self.marketplace.register_producer(),
                         '3', "Incorrect producer id.")

    def test_publish(self):
        
        
        self.marketplace.register_producer()
        self.marketplace.register_producer()

        
        self.marketplace.new_cart()

        
        self.assertTrue(self.marketplace.publish(
            '0', self.teas[0]), "Did not publish tea")
        self.assertTrue(self.marketplace.publish(
            '0', self.teas[1]), "Did not publish tea")
        self.assertTrue(self.marketplace.publish(
            '0', self.teas[0]), "Did not publish tea")
        self.assertTrue(self.marketplace.publish(
            '0', self.teas[2]), "Did not publish tea")
        self.assertTrue(self.marketplace.publish(
            '0', self.coffees[0]), "Did not publish coffee")
        
        
        self.assertFalse(self.marketplace.publish(
            '0', self.coffees[1]), "Reached max queue")

        
        self.marketplace.add_to_cart(0, self.coffees[0])

        
        self.assertTrue(self.marketplace.publish(
            '0', self.coffees[1]), "Did not publish coffee")

        
        self.assertTrue(self.marketplace.publish(
            '1', self.teas[2]), "Did not publish tea")
        self.assertTrue(self.marketplace.publish(
            '1', self.coffees[2]), "Did not publish coffee")

    def test_new_cart(self):
        
        
        self.assertEqual(self.marketplace.new_cart(), 0, "Incorrect cart id.")
        self.assertEqual(self.marketplace.new_cart(), 1, "Incorrect cart id.")
        self.assertEqual(self.marketplace.new_cart(), 2, "Incorrect cart id.")
        self.assertEqual(self.marketplace.new_cart(), 3, "Incorrect cart id.")

    def test_add_to_cart(self):
        
        
        self.marketplace.register_producer()
        self.marketplace.register_producer()

        
        self.marketplace.new_cart()
        self.marketplace.new_cart()

        
        
        self.assertFalse(self.marketplace.add_to_cart(
            0, self.coffees[1]), "Inexistent coffee")

        
        self.marketplace.publish('0', self.teas[0])
        self.marketplace.publish('0', self.teas[1])

        
        self.assertTrue(self.marketplace.add_to_cart(
            1, self.teas[0]), "Did not add tea")
        
        self.assertFalse(self.marketplace.add_to_cart(
            0, self.teas[0]), "Inexistent tea")

        
        self.marketplace.publish('1', self.coffees[1])
        self.marketplace.publish('1', self.coffees[2])

        
        self.assertFalse(self.marketplace.add_to_cart(
            1, self.coffees[0]), "Inexistent coffee")
        
        self.assertTrue(self.marketplace.add_to_cart(
            0, self.coffees[1]), "Did not add coffee")
        self.assertTrue(self.marketplace.add_to_cart(
            0, self.coffees[2]), "Did not add coffee")

    def test_remove_from_cart(self):
        
        
        self.marketplace.register_producer()
        self.marketplace.register_producer()

        
        self.marketplace.new_cart()
        self.marketplace.new_cart()

        
        self.assertRaises(Exception, self.marketplace.remove_from_cart,
                          0, self.coffees[0], "No product to be removed from cart")

        


        self.marketplace.publish('1', self.coffees[0])

        
        self.marketplace.add_to_cart(1, self.coffees[0])
        
        self.assertFalse(self.marketplace.add_to_cart(
            0, self.coffees[0]), "Inexistent coffee")

        
        self.marketplace.remove_from_cart(1, self.coffees[0])
        
        self.assertTrue(self.marketplace.add_to_cart(
            0, self.coffees[0]), "Did not add coffee")

        
        self.marketplace.publish('0', self.teas[0])
        self.marketplace.publish('0', self.teas[1])
        self.marketplace.publish('0', self.teas[2])
        self.marketplace.publish('0', self.teas[0])
        self.marketplace.publish('0', self.teas[1])

        
        self.marketplace.add_to_cart(0, self.teas[0])
        


        self.marketplace.publish('0', self.teas[2])

        
        
        self.marketplace.add_to_cart(0, self.teas[1])
        
        
        self.marketplace.remove_from_cart(0, self.teas[1])

        
        self.assertFalse(self.marketplace.publish(
            '0', self.coffees[0]), "Reached max queue")

    def test_place_order(self):
        
        
        self.marketplace.register_producer()
        self.marketplace.register_producer()

        
        self.marketplace.new_cart()
        self.marketplace.new_cart()

        
        self.marketplace.publish('0', self.teas[0])
        self.marketplace.publish('1', self.coffees[0])
        self.marketplace.publish('0', self.teas[2])
        self.marketplace.publish('0', self.teas[0])
        self.marketplace.publish('1', self.coffees[1])

        


        self.marketplace.add_to_cart(0, self.teas[0])
        self.marketplace.add_to_cart(0, self.coffees[1])

        
        self.assertEqual(Counter(self.marketplace.place_order(0)),
                         Counter([self.teas[0], self.coffees[1]]))

        
        self.marketplace.add_to_cart(0, self.coffees[0])

        
        self.assertEqual(Counter(self.marketplace.place_order(0)),
                         Counter([self.coffees[0]]))

        


        self.marketplace.add_to_cart(1, self.teas[0])
        self.marketplace.remove_from_cart(1, self.teas[0])

        
        self.assertEqual(self.marketplace.place_order(1), [])


from threading import Thread
from time import sleep


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        
        Thread.__init__(self)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        
        self.producer_id = self.marketplace.register_producer()
        self.daemon = True

    def run(self):
        
        while True:
            for product in self.products:
                
                product_data = product[0]
                quantity = product[1]
                wait_time = product[2]

                
                
                for _ in range(quantity):
                    while not self.marketplace.publish(self.producer_id, product_data):
                        
                        
                        sleep(self.republish_wait_time)

                
                sleep(wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    
    name: str
    price: int

    
    def __hash__(self):
        return hash((self.name, self.price))

    
    def __eq__(self, other):
        return (self.name, self.price) == (other.name, other.price)


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    
    acidity: str
    roast_level: str
