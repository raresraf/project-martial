
"""
@702ddbd5-41dd-4906-9947-a3f05a074844/consumer.py
@brief Synchronized marketplace simulation with booking-based state management.
This module implements a threaded multi-agent system where producers publish 
products and consumers acquire them via shopping carts. It utilizes a 
"booking" (mark_booked_products) pattern to manage item availability 
within producer queues and ensures thread safety through granular locking. 
A rotating logging system tracks the lifecycle of transactions.

Domain: Concurrent Programming, Synchronization, System Auditing.
"""


from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Functional Utility: Represent a consumer thread that executes a shopping schedule.
    Logic: For each assigned cart, it performs 'add' or 'remove' operations 
    sequentially. Acquisitions include a retry loop with backoff for stock 
    depletion scenarios.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Constructor: Binds the consumer to its schedules and the shared broker.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        Execution Logic: Orchestrates the processing of multiple shopping carts.
        """
        for i in self.carts:
            cart_id = self.marketplace.new_cart()


            for j in i:
                if j['type'] == "add":
                    /**
                     * Block Logic: Product acquisition loop.
                     * Logic: Attempts to add products to the cart. If the marketplace 
                     * is empty, it sleeps before re-trying.
                     */
                    for elem in range(j['quantity']):
                        add_val = self.marketplace.add_to_cart(cart_id, j['product'])
                        while add_val is False:
                            sleep(self.retry_wait_time)
                            add_val = self.marketplace.add_to_cart(cart_id, j['product'])
                elif j['type'] == "remove":
                    /**
                     * Block Logic: Product return.
                     */
                    for elem in range(j['quantity']):
                        self.marketplace.remove_from_cart(cart_id, j['product'])

            self.marketplace.place_order(cart_id)

import logging
from logging.handlers import RotatingFileHandler
import threading
import time
import unittest
from threading import Lock


logging.basicConfig(filename="marketplace.log", level=logging.INFO)

# Functional Utility: Logger initialization with rotation policies.
LOGGER = logging.getLogger('mktplace_logger')
LOGGER.setLevel(logging.INFO)

HANDLER = RotatingFileHandler('marketplace.log', maxBytes=7500000, backupCount=10)
LOGGER.addHandler(HANDLER)

logging.Formatter.converter = time.gmtime()


class Marketplace:
    """
    Functional Utility: Central coordinator for thread-safe product transactions.
    Logic: Tracks producers, their inventory, and consumer carts. It uses a 
    dual-state mapping (products_queue and mark_booked_products) to distinguish 
    between available, booked (in cart), and consumed products. Thread safety 
    is maintained via multiple specialized locks.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Constructor: Initializes storage structures and synchronization primitives.
        """
        self.queue_size_per_producer = queue_size_per_producer

        self.id_for_producer = 0
        self.id_for_cart = 0
        self.products_queue = []
        self.mark_booked_products = []
        self.list_of_carts = []

        self.lock_for_id = Lock()
        self.lock_for_cart_id = Lock()
        self.lock_for_add_cart = Lock()
        self.lock_for_publish = Lock()

        LOGGER.info("Constructor with queue_size_per_producer %s", queue_size_per_producer)

    def register_producer(self):
        """
        Functional Utility: Issues a unique ID to a new producer.
        """
        LOGGER.info("Register producer, asking for an id")
        self.lock_for_id.acquire()

        temp_id = self.id_for_producer
        
        self.products_queue.append([])
        
        self.mark_booked_products.append([])
        
        self.id_for_producer = self.id_for_producer + 1

        self.lock_for_id.release()
        LOGGER.info("Register producer, he will get the id %s", self.id_for_producer)

        return temp_id

    def publish(self, producer_id, product):
        """
        Functional Utility: Adds a product to a producer's queue.
        Logic: Enforces per-producer limits and initializes the item 
        as 'not booked' (0).
        """
        LOGGER.info("Producer with id %s wants to publish the following product: %s"
                    , producer_id, product)

        self.lock_for_publish.acquire()
        
        if (len(self.products_queue[producer_id]) + 1) <= self.queue_size_per_producer:
            self.products_queue[producer_id].append(product)
            self.mark_booked_products[producer_id].append(0)
            self.lock_for_publish.release()
            LOGGER.info("Producer published the product!")
            return True
        else:
            self.lock_for_publish.release()
            LOGGER.info("Producer did not publish the product!")
            return False


    def new_cart(self):
        """
        Functional Utility: Creates a new shopping cart for a consumer.
        """
        LOGGER.info("A new cart is created for a consumer")
        self.lock_for_cart_id.acquire()

        temp_id = self.id_for_cart
        
        self.list_of_carts.append(list([]))
        
        self.id_for_cart = self.id_for_cart + 1

        self.lock_for_cart_id.release()
        LOGGER.info("The cart was created and has the id %s", temp_id)

        return temp_id

    def add_to_cart(self, cart_id, product):
        """
        Functional Utility: Reserves a product instance for a cart.
        Logic: Scans producer queues for the product. If an unbooked instance 
        is found, it is added to the cart and marked as booked (1).
        """
        LOGGER.info("A cosumer wants to add product %s to the cart with id %s", product, cart_id)
        found_product = False


        self.lock_for_add_cart.acquire()
        
        # Block Logic: Nested scan for available product instances.
        for l_elem in range(len(self.products_queue)):
            for p_elem in range(len(self.products_queue[l_elem])):
                if (self.products_queue[l_elem][p_elem] == product and
                        self.mark_booked_products[l_elem][p_elem] == 0):
                    
                    self.list_of_carts[cart_id].append(self.products_queue[l_elem][p_elem])
                    self.mark_booked_products[l_elem][p_elem] = 1
                    found_product = True

                    break
            if found_product:
                break

        self.lock_for_add_cart.release()
        if found_product:
            LOGGER.info("The consumer added the product to the cart")
        else:
            LOGGER.info("The consumer did not add the product to the cart")
        return found_product


    def remove_from_cart(self, cart_id, product):
        """
        Functional Utility: Returns an item from a cart to available stock.
        Logic: Finds the booked instance in the producer queue and resets 
        its status to unbooked (0).
        """
        LOGGER.info("A cosumer wants to remove product %s from the cart %s", product, cart_id)
        self.lock_for_add_cart.acquire()
        removed_product = False
        for l_elem in range(len(self.products_queue)):
            for p_elem in range(len(self.products_queue[l_elem])):
                if (self.products_queue[l_elem][p_elem] == product and
                        self.mark_booked_products[l_elem][p_elem] == 1):
                    
                    self.list_of_carts[cart_id].remove(product)
                    self.mark_booked_products[l_elem][p_elem] = 0
                    removed_product = True
                    break

            if removed_product:
                break

        self.lock_for_add_cart.release()
        LOGGER.info("The consumer removed the product from the cart")


    def place_order(self, cart_id):
        """
        Functional Utility: Finalizes the order and removes booked items from stock.
        Logic: Consumes booked items from producer queues permanently.
        """
        LOGGER.info("A consumer wants to place the order for the cart with id %s", cart_id)
        self.lock_for_publish.acquire()
        elem_list = []
        found = False
        for i in self.list_of_carts[cart_id]:
            print(threading.current_thread().name, "bought", i)
            elem_list.append(i)
            found = False
            for l_elem in range(len(self.products_queue)):
                for p_elem in range(len(self.products_queue[l_elem])):
                    if (self.products_queue[l_elem][p_elem] == i and
                            self.mark_booked_products[l_elem][p_elem] == 1):
                        self.products_queue[l_elem].pop(p_elem)
                        self.mark_booked_products[l_elem].pop(p_elem)
                        found = True
                    if found:
                        break
                if found:
                    break

        self.lock_for_publish.release()
        LOGGER.info("The consumer placed the order")
        return elem_list


class TestMarketplace(unittest.TestCase):
    """
    Functional Utility: Integrity validation suite for Marketplace transaction logic.
    """
    
    def setUp(self):
        self.marketplace = Marketplace(15)
        self.carts = [["add", "id2", 1], ["add", "id1", 3], ["remove", "id1", 1]]
        self.products = [["id2", 2, 0.18], ["id1", 1, 0.23]]
        self.consumer = Consumer(self.carts, self.marketplace, 0.31)
        self.producer = Producer(self.products, self.marketplace, 0.15)

    def test_register_producer(self):
        
        expected_register_id = 0
        self.assertEqual(expected_register_id, self.producer.marketplace.register_producer())

    def test_publish(self):
        
        expected_queue = [["id2", "id2", "id1", "id2", "id2", "id1", "id2", "id2",
                           "id1", "id2", "id2", "id1", "id2", "id2", "id1"]]
        self.marketplace.products_queue.append([])
        self.marketplace.mark_booked_products.append([])
        pos = 0
        while pos < self.marketplace.queue_size_per_producer:
            if pos % 3 == 2:
                self.producer.marketplace.publish(0, "id1")
            else:
                self.producer.marketplace.publish(0, "id2")
            pos += 1
        self.assertEqual(expected_queue, self.marketplace.products_queue)

    def test_new_cart(self):
        
        expected_cart_id = 0
        self.assertEqual(expected_cart_id, self.consumer.marketplace.new_cart())

    def test_add_to_cart(self):
        
        self.marketplace.products_queue.append([])
        self.marketplace.mark_booked_products.append([])
        pos = 0
        while pos < self.marketplace.queue_size_per_producer:
            if pos % 3 == 2:
                self.producer.marketplace.publish(0, "id1")
            else:
                self.producer.marketplace.publish(0, "id2")
            pos += 1
        self.consumer.marketplace.list_of_carts.append([])
        expected_cart = ["id2", "id1", "id1", "id1"]
        for i in self.carts:
            if i[0] == "add":
                for j in range(i[2]):
                    self.consumer.marketplace.add_to_cart(0, i[1])
        self.assertEqual(expected_cart, self.consumer.marketplace.list_of_carts[0])

    def test_remove_from_cart(self):
        
        expected_cart = ["id2", "id1", "id1"]
        self.marketplace.products_queue.append([])
        self.marketplace.mark_booked_products.append([])
        pos = 0
        while pos < self.marketplace.queue_size_per_producer:
            if pos % 3 == 2:
                self.producer.marketplace.publish(0, "id1")
            else:
                self.producer.marketplace.publish(0, "id2")
            pos += 1
        self.consumer.marketplace.list_of_carts.append([])
        for i in self.carts:
            if i[0] == "add":
                for j in range(i[2]):
                    self.consumer.marketplace.add_to_cart(0, i[1])
        for i in self.carts:
            if i[0] == "remove":
                for j in range(i[2]):
                    self.consumer.marketplace.remove_from_cart(0, i[1])
        self.assertEqual(expected_cart, self.consumer.marketplace.list_of_carts[0])

    def test_place_order(self):
        
        expected_output = ["id2", "id1", "id1"]
        self.marketplace.products_queue.append([])
        self.marketplace.mark_booked_products.append([])
        pos = 0
        while pos < self.marketplace.queue_size_per_producer:
            if pos % 3 == 2:
                self.producer.marketplace.publish(0, "id1")
            else:
                self.producer.marketplace.publish(0, "id2")
            pos += 1
        self.consumer.marketplace.list_of_carts.append([])
        for i in self.carts:
            if i[0] == "add":
                for j in range(i[2]):
                    self.consumer.marketplace.add_to_cart(0, i[1])
        for i in self.carts:
            if i[0] == "remove":
                for j in range(i[2]):
                    self.consumer.marketplace.remove_from_cart(0, i[1])
        self.assertEqual(expected_output, self.consumer.marketplace.place_order(0))


from threading import Thread
import time


class Producer(Thread):
    """
    Functional Utility: Represent a production agent that continuously supplies items.
    Logic: Iterates through its product list, publishing items and observing 
    simulated manufacturing times. It implements a retry mechanism for 
    congested marketplaces.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Constructor: Registers the producer and initializes production parameters.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace


        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        Execution Logic: Infinite production loop.
        """
        register_id = self.marketplace.register_producer()
        while 1:
            for i in self.products:
                for j in range(i[1]):
                    value = self.marketplace.publish(register_id, i[0])
                    
                    # Block Logic: Congestion backoff.
                    while value is False:
                        time.sleep(self.republish_wait_time)
                        value = self.marketplace.publish(register_id, i[0])
                    
                    # Inline: Simulated production delay.
                    time.sleep(i[2])
