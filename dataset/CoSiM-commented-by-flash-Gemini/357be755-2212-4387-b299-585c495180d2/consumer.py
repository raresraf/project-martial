

"""
@05314305-b286-4c5b-a80e-5c46defa6a97/arch/arm/crypto/Makefile consumer.py
@brief Implements a simulated e-commerce system with producers, a marketplace, and consumers, demonstrating concurrent operations.
This module integrates several components: `Consumer` threads for purchasing products from the marketplace,
the `Marketplace` itself for managing product inventory and carts, `Producer` threads for supplying products,
and `Product` dataclasses (Coffee, Tea) for defining product types.
It utilizes threading and synchronization primitives to simulate a concurrent environment.
"""

from threading import Thread
from time import sleep

class Consumer(Thread):
    """
    @brief Represents a consumer thread in the e-commerce simulation.
    Each consumer attempts to purchase products defined in its shopping carts
    from the marketplace, handling retries if products are not immediately available.
    """
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a Consumer thread.
        @param carts A list of shopping cart definitions, where each cart is a list of operations (add/remove product).
        @param marketplace The `Marketplace` instance from which to buy products.
        @param retry_wait_time The time in seconds to wait before retrying an add-to-cart operation if it fails.
        @param kwargs Arbitrary keyword arguments, including a "name" for the consumer.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.id = kwargs["name"] # Unique identifier for the consumer, typically its thread name.

    def run(self):
        """
        @brief The main execution loop for the consumer thread.
        Iterates through its assigned carts, performs add/remove operations on products,
        and finally places an order, retrying add operations if products are unavailable.
        """
        for cart in self.carts:
            # Block Logic: Acquire marketplace lock to safely create a new cart.
            self.marketplace.lock.acquire()
            cart_id = self.marketplace.new_cart()
            self.marketplace.lock.release()


            for operation in cart:
                type = operation['type']
                product = operation['product']
                quantity = operation['quantity']
                for i in range(quantity):
                    self.marketplace.lock.acquire() # Acquire marketplace lock for cart modification.
                    if type == "add":
                        # Block Logic: Continuously try to add product to cart until successful.
                        # Releases lock, sleeps, and re-acquires lock if product is unavailable.
                        while not self.marketplace.add_to_cart(cart_id, product):
                            self.marketplace.lock.release()
                            sleep(self.retry_wait_time)
                            self.marketplace.lock.acquire()
                    else:
                        # Block Logic: Remove product from cart.
                        self.marketplace.remove_from_cart(cart_id, product)
                    self.marketplace.lock.release() # Release marketplace lock after cart modification.

            # Place the final order for the current cart.
            products = self.marketplace.place_order(cart_id)

            # Print purchased products.
            for product in products:
                print(self.id + " bought " + str(product))

import logging
import time
import unittest
from threading import Lock

from tema.product import Coffee, Product, Tea


class Marketplace:
    

    def __init__(self, queue_size_per_producer):
        
        self.max_queue_size = queue_size_per_producer
        self.available_products = {}
        self.no_available_products = {}

        self.carts_in_use = {}
        self.last_cart_id = -1

        self.last_producer_id = 0

        self.lock = Lock()
        logging.Formatter.converter = time.gmtime
        logging.basicConfig(filename="marketplace.log",
                            filemode="a",
                            format='%(asctime)s,%(msecs)d %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)
        self.logger = logging.getLogger()

    def add_available_product(self, producer_id, product):
        self.logger.info(
            'Entered function "add_available_product" with parameter: ' + str(producer_id) + ' and ' + str(product))
        if product not in self.available_products[producer_id]:
            self.available_products[producer_id][product] = 1
        else:
            self.available_products[producer_id][product] += 1

        self.no_available_products[producer_id] += 1
        self.logger.info('Exit function "add_available_product"')

    def register_producer(self):
        
        self.logger.info('Entered function "register_producer"')

        self.last_producer_id += 1

        self.no_available_products[self.last_producer_id] = 0
        self.available_products[self.last_producer_id] = {}

        self.logger.info('Exit function "register_producer" with value: ' + str(self.last_producer_id))

        return self.last_producer_id

    def publish(self, producer_id, product):
        
        self.logger.info('Entered function "publish" with parameters ' + str(producer_id) + ' and  ' + str(product))

        if self.no_available_products[producer_id] >= self.max_queue_size:
            self.logger.info('Exit function "publish" with value: False')
            return False

        self.add_available_product(producer_id, product)

        self.logger.info('Exit function "publish" with value: True')

        return True

    def new_cart(self):
        
        self.logger.info('Entered function "new_cart"')

        self.last_cart_id += 1
        self.carts_in_use[self.last_cart_id] = []

        self.logger.info('Exit function "new_cart" with value: ' + str(self.last_cart_id))

        return self.last_cart_id

    def add_to_cart(self, cart_id, product):
        

        self.logger.info('Entered function "add_to_cart" with parameters: ' + str(cart_id) + ' and  ' + str(product))

        for producer in self.available_products.keys():
            if product in self.available_products[producer]:
                number_of_products = self.available_products[producer][product]
                number_of_products -= 1
                if number_of_products <= 0:
                    del self.available_products[producer][product]
                else:
                    self.available_products[producer][product] = number_of_products

                self.carts_in_use[cart_id].append((product, producer))

                self.no_available_products[producer] -= 1

                self.logger.info('Exit function "add_to_cart" with value: True')

                return True

        self.logger.info('Exit function "add_to_cart" with value: False')

        return False

    def remove_from_cart(self, cart_id, product):
        
        self.logger.info(
            'Entered function "remove_from_cart" with parameters: ' + str(cart_id) + ' and ' + str(product))

        if cart_id not in self.carts_in_use:
            return

        for product_producer in self.carts_in_use[cart_id]:
            current_product = product_producer[0]
            current_producer = product_producer[1]
            if current_product == product:
                self.add_available_product(current_producer, product)
                self.carts_in_use[cart_id].remove(product_producer)
                break

        self.logger.info('Exit function "remove_from_cart"')

    def place_order(self, cart_id):
        
        self.logger.info('Entered function "place_order" + with parameter: ' + str(cart_id))

        if cart_id not in self.carts_in_use:
            self.logger.info('Exit function "place_order" with value: []')
            return []

        self.logger.info('Exit function "place_order" with value: ' + str(self.carts_in_use[cart_id]))

        produce_list = []
        for product_producer in self.carts_in_use[cart_id]:
            produce_list.append(product_producer[0])

        return produce_list


class TestMarketplace(unittest.TestCase):
    def setUp(self):
        self.marketplace = Marketplace(1)
        self.coffee1 = Coffee(name="Ethiopia", price=10, acidity="6", roast_level='MEDIUM')
        self.coffee2 = Coffee(name="China", price=10, acidity="1", roast_level='HIGH')
        self.tea = Tea(name="Ethiopia", price=10, type="Black")

    def test_register_producer(self):
        id = self.marketplace.register_producer()
        self.assertEqual(1, id)

    def test_publish(self):
        id = self.marketplace.register_producer()
        self.marketplace.publish(id, self.coffee1)
        self.marketplace.publish(id, self.coffee2)
        self.assertEqual(1, self.marketplace.available_products[id][self.coffee1])
        self.assertEqual(False, self.coffee2 in self.marketplace.available_products[id])

    def test_new_cart(self):
        id = self.marketplace.new_cart()
        self.assertEqual(id, 0)

    def test_add_to_cart(self):
        id_cart = self.marketplace.new_cart()


        id_producer = self.marketplace.register_producer()
        self.marketplace.publish(id_producer, self.tea)
        self.marketplace.add_to_cart(id_cart, self.tea)
        self.assertEqual(True, (self.tea, id_producer) in self.marketplace.carts_in_use[id_cart])

    def test_remove_from_cart(self):
        id_cart = self.marketplace.new_cart()
        id_producer = self.marketplace.register_producer()


        self.marketplace.publish(id_producer, self.tea)
        self.marketplace.add_to_cart(id_cart, self.tea)
        self.marketplace.remove_from_cart(id_cart, self.tea)
        self.assertEqual(False, (self.tea, id_producer) in self.marketplace.carts_in_use[id_cart])

    def test_place_order(self):


        id_cart = self.marketplace.new_cart()

        id_producer1 = self.marketplace.register_producer()
        id_producer2 = self.marketplace.register_producer()

        self.marketplace.publish(id_producer1, self.coffee1)
        self.marketplace.publish(id_producer2, self.coffee2)



        self.marketplace.add_to_cart(id_cart, self.coffee1)
        self.marketplace.add_to_cart(id_cart, self.coffee2)

        order_list = self.marketplace.place_order(id_cart)

        self.assertEqual([self.coffee1, self.coffee2], order_list)

    def test_add_available_product(self):
        id = self.marketplace.register_producer()
        self.marketplace.add_available_product(id, self.coffee1)
        self.assertEqual(1, self.marketplace.available_products[id][self.coffee1])


from threading import Thread
from time import sleep


class Producer(Thread):
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.id = -1

    def run(self):
        self.marketplace.lock.acquire()
        self.id = self.marketplace.register_producer()
        self.marketplace.lock.release()

        while True:
            for product in self.products:
                real_product = product[0]
                quantity = product[1]
                time_to_produce = product[2]
                for i in range(quantity):
                    sleep(time_to_produce)
                    self.marketplace.lock.acquire()
                    while not self.marketplace.publish(self.id, real_product):
                        self.marketplace.lock.release()
                        sleep(self.republish_wait_time)
                        self.marketplace.lock.acquire()
                    self.marketplace.lock.release()


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
