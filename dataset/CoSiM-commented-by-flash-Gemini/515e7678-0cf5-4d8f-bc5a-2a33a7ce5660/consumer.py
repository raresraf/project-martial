
"""
@515e7678-0cf5-4d8f-bc5a-2a33a7ce5660/consumer.py
@brief Threaded multi-agent marketplace with state synchronization.
This file implements a concurrent architecture for a product exchange system. 
It defines Consumer and Producer threads that interact with a central 
Marketplace coordinator. The system ensures thread safety through mutex locks 
and provides a rotating logging mechanism for transaction auditing.

Domain: Concurrent Systems, Synchronization, Resource Management.
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Functional Utility: Represent a consumer thread that executes a series of shopping actions.
    Logic: For each cart in its assigned list, it performs acquisitions ('add') 
    or returns ('remove') via the marketplace. It utilizes a retry mechanism with 
    backoff for stock unavailability.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Constructor: Binds the consumer to specific carts and the shared marketplace.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        Execution Logic: Iterative processing of shopping carts.
        Invariant: All operations within a single cart are completed before the 
        final order is placed and result printed.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for cart_op in cart:
                quantity = cart_op["quantity"]
                if cart_op["type"] == "add":
                    /**
                     * Block Logic: Synchronized product acquisition.
                     * Logic: Repeatedly attempts to add the product. If rejected (out of stock), 
                     * it enters a timed sleep before re-evaluating marketplace state.
                     */
                    while quantity > 0:
                        res = self.marketplace.add_to_cart(cart_id, cart_op["product"])
                        while not res:
                            sleep(self.retry_wait_time)
                            res = self.marketplace.add_to_cart(cart_id, cart_op["product"])
                        quantity -= 1
                elif cart_op["type"] == "remove":
                    /**
                     * Block Logic: Product return.
                     */
                    while quantity > 0:
                        self.marketplace.remove_from_cart(cart_id, cart_op["product"])
                        quantity -= 1
            
            # Finalizes the current cart transaction.
            order = self.marketplace.place_order(cart_id)
            for element in order:
                print(self.name + ' bought ' + str(element))


import time
import unittest
from threading import Lock
import logging
from logging.handlers import RotatingFileHandler
from tema.product import Tea, Coffee


class Marketplace:
    """
    Functional Utility: Synchronized broker for product publication and acquisition.
    Logic: Manages producer registration, atomic inventory updates, and consumer 
    cart lifecycles. It employs multiple locks (producer_id_lock, cart_id_lock, 
    producer_queue_lock) to minimize lock contention across different domains 
    of the system.
    """

    def __init__(self, queue_size_per_producer):
        """
        Constructor: Initializes internal storage and synchronization primitives.
        """
        self.queue_size_per_producers = queue_size_per_producer

        
        self.producers_dict = {}
        self.producer_id_seed = 0
        self.producer_id_lock = Lock()

        
        self.carts_dict = {}
        self.cart_id_lock = Lock()
        self.cart_id_seed = 0

        
        self.product_to_producer_id = {}

        
        self.producers_queue_sizes = {}
        self.producer_queue_lock = Lock()

        # Block Logic: Audit logging configuration.
        logging.basicConfig(
            handlers=[RotatingFileHandler("marketplace.log",
                                          maxBytes=10000, backupCount=10)],
            level=logging.INFO,
            format='%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logging.Formatter.converter = time.gmtime
        logging.info('Marketplace start!')

    def register_producer(self):
        """
        Functional Utility: Issues a unique identification token to a new producer.
        """
        with self.producer_id_lock:
            self.producer_id_seed += 1
        producer_id = self.producer_id_seed
        self.producers_dict[producer_id] = []
        self.producers_queue_sizes[producer_id] = 0
        logging.info('new producer id registered: %s', str(producer_id))
        return producer_id

    def publish(self, producer_id, product):
        """
        Functional Utility: Allows a producer to submit a product for sale.
        Logic: Checks against per-producer capacity. Updates both the product 
        listing and the inverse mapping for lookup optimization.
        Returns: True if successful, False if queue limit reached.
        """
        logging.info('publish method arguments: (producer_id:%s), (product:%s)',
                     str(producer_id), str(product))
        if self.producers_queue_sizes[producer_id] >= self.queue_size_per_producers:
            logging.info('publish method return=%s', 'False')
            return False
        self.producers_dict[producer_id].append(product)
        self.product_to_producer_id[product] = producer_id
        


        with self.producer_queue_lock:
            self.producers_queue_sizes[producer_id] += 1
        logging.info('publish method return=%s', 'True')
        return True

    def new_cart(self):
        """
        Functional Utility: Allocates a new shopping cart ID for a consumer.
        """
        with self.cart_id_lock:
            self.cart_id_seed += 1
            cart_id = self.cart_id_seed
        self.carts_dict[cart_id] = []
        logging.info('new_cart method return cart_id=%s', str(cart_id))
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Functional Utility: Atomically transfers a product from a producer to a consumer cart.
        Logic: Resolves product to producer, verifies stock, removes from source 
        listing and appends to destination cart.
        """
        logging.info('add_to_cart method arguments: (cart_id:%s), (product:%s)',
                     str(cart_id), str(product))
        
        producer_id = self.product_to_producer_id.get(product, None)
        if producer_id is None:
            logging.info('add_to_cart method return=%s', 'False')
            return False
        if product not in self.producers_dict[producer_id]:
            logging.info('add_to_cart method return=%s', 'False')
            return False
        
        self.producers_dict[producer_id].remove(product)


        self.carts_dict[cart_id].append((producer_id, product))
        logging.info('add_to_cart method return=%s', 'True')
        return True

    def remove_from_cart(self, cart_id, product):
        """
        Functional Utility: Returns an item from a cart back to the producer's inventory.
        """
        logging.info('remove_from_cart method arguments: (cart_id=%s), (product=%s)',
                     cart_id, product)
        producer_id = -1
        list_cart = self.carts_dict[cart_id]

        
        for _, cart_tuple in enumerate(list_cart):
            if product == cart_tuple[1]:
                producer_id = cart_tuple[0]
                break
        if producer_id == -1:
            logging.info('remove_from_cart method return=%s', 'False')
            return False

        
        self.carts_dict[cart_id].remove((producer_id, product))
        
        self.producers_dict[producer_id].append(product)
        logging.info('remove_from_cart method return=%s', 'True')
        return True

    def place_order(self, cart_id):
        """
        Functional Utility: Finalizes the order and updates producer availability.
        Logic: Consumes the cart, decrements published counts for each producer 
        involved, and returns the final product sequence.
        """
        logging.info('place_order method argument: (cart_id:%s)', cart_id)
        res = self.carts_dict.pop(cart_id)
        final_list = []
        for _, cart_tuple in enumerate(res):
            with self.producer_queue_lock:
                self.producers_queue_sizes[cart_tuple[0]] -= 1
            final_list.append(cart_tuple[1])
        logging.info('place_order method return=%s', str(final_list))
        return final_list


class TestMarketplace(unittest.TestCase):
    """
    Functional Utility: Integrity validation suite for Marketplace logic.
    """
    def setUp(self):
        self.marketplace = Marketplace(3)
        self.product1 = Tea(name='Wild Cherry', price=5, type='Black')
        self.product2 = Coffee(name='Brasil', price=7, acidity=5.09, roast_level='MEDIUM')

    def test_register_producer(self):
        res = self.marketplace.register_producer()
        self.assertEqual(res, 1, "Register wrong!")
        res = self.marketplace.register_producer()
        self.assertEqual(res, 2, "Register wrong!")

    def test_publish(self):
        self.marketplace.register_producer()
        self.marketplace.register_producer()
        self.assertEqual(self.marketplace.publish(1, self.product1),
                         True, "incorrect return value for publish")
        self.assertEqual(self.marketplace.publish(1, self.product2),
                         True, "incorrect return value for publish")
        self.assertEqual(self.marketplace.publish(1, self.product1),
                         True, "incorrect return value for publish")
        self.assertEqual(self.marketplace.publish(1, self.product2),
                         False, "incorrect return value for publish")
        self.assertEqual(self.marketplace.publish(2, self.product1),
                         True, "incorrect return value for publish")

    def test_new_cart(self):
        self.assertEqual(self.marketplace.new_cart(), 1, "Error in new cart!")
        self.assertEqual(self.marketplace.new_cart(), 2, "Error in new cart!")
        self.assertEqual(self.marketplace.new_cart(), 3, "Error in new cart!")

    def test_add_to_cart(self):
        self.marketplace.register_producer()
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.product1)
        self.marketplace.publish(2, self.product2)
        self.marketplace.new_cart()
        self.marketplace.new_cart()
        self.marketplace.add_to_cart(1, self.product1)
        self.assertEqual(len(self.marketplace.carts_dict[1]), 1, "Error in add to cart!")
        self.marketplace.add_to_cart(2, self.product1)
        self.assertEqual(len(self.marketplace.carts_dict[2]), 0, "Error in add to cart!")
        self.marketplace.add_to_cart(2, self.product2)
        self.assertEqual(len(self.marketplace.carts_dict[2]), 1, "Error in add to cart!")

    def test_remove_from_cart(self):
        self.marketplace.register_producer()
        self.marketplace.register_producer()


        self.marketplace.publish(1, self.product1)
        self.marketplace.publish(2, self.product2)
        self.marketplace.new_cart()
        self.marketplace.add_to_cart(1, self.product1)
        self.marketplace.add_to_cart(1, self.product2)
        self.marketplace.remove_from_cart(1, self.product2)
        self.assertEqual(len(self.marketplace.carts_dict[1]), 1, "Error in add to cart!")
        self.marketplace.remove_from_cart(1, self.product1)
        self.assertEqual(len(self.marketplace.carts_dict[1]), 0, "Error in add to cart!")

    def place_order(self):
        self.marketplace.register_producer()
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.product1)
        self.marketplace.publish(2, self.product2)
        self.marketplace.new_cart()
        self.marketplace.add_to_cart(1, self.product1)
        self.marketplace.add_to_cart(1, self.product2)
        self.marketplace.place_order(1)
        self.assertEqual(len(self.marketplace.producers_dict[1]),
                         0, "Error in add to cart!")
        self.assertEqual(len(self.marketplace.producers_dict[2]),
                         0, "Error in add to cart!")


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Functional Utility: Represent a production agent that continuously supplies the marketplace.
    Logic: Iterates through its production schedule, attempting to publish products. 
    It incorporates simulated manufacturing time and congestion handling.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Constructor: Registers the producer and initializes its schedule.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.id = self.marketplace.register_producer()

    def run(self):
        """
        Execution Logic: Infinite production loop.
        """
        while True:
            for product in self.products:
                quantity = product[1]
                while quantity > 0:
                    res = self.marketplace.publish(self.id, product[0])
                    
                    # Block Logic: Marketplace congestion backoff.
                    while not res:
                        sleep(self.republish_wait_time)
                        res = self.marketplace.publish(self.id, product[0])
                    
                    # Inline: Simulated production delay.
                    sleep(product[2])
                    quantity -= 1





from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Functional Utility: Immutable data carrier for generic products.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Functional Utility: Specialized data carrier for tea varieties.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Functional Utility: Specialized data carrier for coffee varieties.
    """
    acidity: str
    roast_level: str
