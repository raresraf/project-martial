
/**
 * @file consumer.py
 * @brief Thread-safe marketplace simulation with concurrent inventory tracking and logging.
 * 
 * Functional Intent: Orchestrates a multi-threaded producer-consumer system where 
 * multiple Producers publish items and multiple Consumers manage virtual shopping 
 * carts. It employs a distributed locking strategy (per-producer and per-product) 
 * to ensure atomicity during stock updates, while providing a detailed audit 
 * trail via a rotating log file.
 * 
 * Domain: Production Systems, Concurrency, Synchronized State Management.
 */

from threading import Thread
from time import sleep


class Consumer(Thread):
    /**
     * @class Consumer
     * @brief Customer thread that executes a scripted sequence of shopping cart operations.
     * 
     * Logic: For each cart in its assignment, it initializes a new marketplace 
     * session and performs 'add' or 'remove' operations. It implements a blocking 
     * polling loop for stock reservation, ensuring all items are eventually 
     * satisfied before committing the order.
     */

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs["name"]

    def run(self):
        /**
         * Block Logic: Shopping lifecycle management.
         * Invariant: Each cart is processed as an independent transactional unit.
         */
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            
            for operation in cart:
                if operation["type"] == "add":
                    # Block Logic: Transactional reservation with backoff retry.
                    for _ in range(operation["quantity"]):
                        while not self.marketplace.add_to_cart(cart_id, operation["product"]):
                            # Logic: Stall if the product is temporarily out of stock.
                            sleep(self.retry_wait_time)
                elif operation["type"] == "remove":
                    # Logic: Reverses a product reservation within the current session.
                    for _ in range(operation["quantity"]):
                        self.marketplace.remove_from_cart(cart_id, operation["product"])
            
            # Functional Utility: Finalizes the cart and outputs the purchased items.
            order = self.marketplace.place_order(cart_id)
            for product in order:
                print("{0} bought {1}".format(self.name, product))

import time
from threading import Lock
import unittest
import logging
from logging.handlers import RotatingFileHandler
from tema.product import Coffee, Tea


class Marketplace:
    /**
     * @class Marketplace
     * @brief Centralized coordinator for thread-safe inventory and transaction management.
     * 
     * Logic: Uses independent mutexes for identity counters and individual 
     * producer/product buckets to minimize global contention while guaranteeing 
     * consistency during concurrent access.
     */

    def __init__(self, queue_size_per_producer):
        self.queue_size_per_producer = queue_size_per_producer
        
        # Invariant: Maps producer IDs to their current active publication count.
        self.producers_queue = {}
        
        # Invariant: Stores active consumer carts.
        self.carts = {}
        
        # Synchronization: Mutexes for unique ID provisioning.
        self.producer_id = 0
        self.cart_id = 0
        self.producer_id_lock = Lock()
        self.cart_id_lock = Lock()
        
        # Invariant: Map of per-producer mutexes to protect local stock counts.
        self.producers_locks = {}
        
        # Invariant: Map of product instances to the list of producer IDs who supplied them.
        self.products_producers = {}
        
        # Invariant: Fine-grained mutexes for each product type to ensure atomic reservation.
        self.products_locks = {}
        
        # Logging: Initializing production audit trail with rotation policies.
        self.logger = logging.getLogger('my_logger')
        self.logger.setLevel(logging.INFO)
        self.handler = RotatingFileHandler("marketplace.log", maxBytes=1024 * 512, backupCount=20)
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(self.formatter)
        self.formatter.converter = time.gmtime
        self.logger.addHandler(self.handler)

    def register_producer(self):
        /**
         * register_producer - Assigns an identity and bootstraps state for a new provider.
         */
        self.logger.info("Entered register_producer()!")
        with self.producer_id_lock:
            producer_id_string = "prod{0}".format(self.producer_id)
            self.producers_queue[producer_id_string] = 0
            self.producers_locks[producer_id_string] = Lock()
            self.producer_id += 1
        
        self.logger.info("Finished register_producer(): returned producer_id: %s!",
                         producer_id_string)
        return producer_id_string

    def publish(self, producer_id, product):
        /**
         * publish - Adds a new unit to the global inventory pool.
         * 
         * Logic: Enforces the producer's publication quota and updates the 
         * per-product supplier list. Uses fine-grained locking to prevent 
         * race conditions during supplier list mutations.
         */
        self.logger.info("Entered publish(%s, %s)!", producer_id, product)
        
        with self.producers_locks[producer_id]:
            queue_size = self.producers_queue[producer_id]
            if queue_size == self.queue_size_per_producer:
                self.logger.info("Finished publish(%s, %s): Queue is Full!",
                                 producer_id, product)
                return False
            
            # Synchronization: Ensures atomic initialization of product-specific locks.
            if product not in self.products_producers:
                self.products_locks[product] = Lock()
                with self.products_locks[product]:
                    self.products_producers[product] = []
                    self.products_producers[product].append(producer_id)
            else:
                with self.products_locks[product]:
                    self.products_producers[product].append(producer_id)
            
            self.producers_queue[producer_id] += 1
        
        self.logger.info("Finished publish(%s, %s): Published product!",
                         producer_id, product)
        return True

    def new_cart(self):
        self.logger.info("Entered new_cart()!")
        with self.cart_id_lock:
            cart_id = self.cart_id
            self.carts[cart_id] = []
            self.cart_id += 1
        self.logger.info("Finished new_cart(): New cart: %d!", cart_id)
        return cart_id

    def add_to_cart(self, cart_id, product):
        /**
         * add_to_cart - Atomically reserves a product for a consumer session.
         * 
         * Algorithm: Queue-based FIFO reservation.
         * 1. Validates cart and product existence.
         * 2. Acquires the product-specific lock.
         * 3. Pops the first available producer ID from the supply list.
         * 4. Appends the unit metadata to the consumer's cart.
         */
        self.logger.info("Entered add_to_cart(%d, %s)!", cart_id, product)
        
        if cart_id not in self.carts or product not in self.products_producers:
            return False

        with self.products_locks[product]:
            if not self.products_producers[product]:
                self.logger.info("Finished add_to_cart(%d, %s): Product is not available!",
                                 cart_id, product)
                return False
            
            # Logic: Extracts supply reference (reservation).
            producer_id = self.products_producers[product].pop(0)
        
        self.carts[cart_id].append({"product": product,
                                    "producer_id": producer_id})
        self.logger.info("Finished add_to_cart(%d, %s): Product added to cart!",
                         cart_id, product)
        return True

    def remove_from_cart(self, cart_id, product):
        /**
         * remove_from_cart - Discards a reservation and returns the unit to the supply list.
         */
        self.logger.info("Entered remove_from_cart(%d, %s)!", cart_id, product)
        
        if cart_id not in self.carts:
            return False
        
        cart_list = self.carts[cart_id]
        for cart_element in cart_list:
            if cart_element["product"] == product:
                producer_id = cart_element["producer_id"]
                
                # Logic: Restores availability for other consumers.
                with self.products_locks[product]:
                    self.products_producers[product].append(producer_id)
                
                self.carts[cart_id].remove(cart_element)
                self.logger.info("Finished remove_from_cart(%d, %s): Product removed from cart!",
                                 cart_id, product)
                return True
        return False

    def place_order(self, cart_id):
        /**
         * place_order - Converts all reservations in a cart into finalized sales.
         * 
         * Logic: For every unit sold, it decrements the producer's active 
         * count, effectively reclaiming quota space for new publications.
         */
        self.logger.info("Entered place_order(%d)!", cart_id)
        result = []
        
        if cart_id not in self.carts:
            return None
        
        cart_list = self.carts[cart_id]
        for cart_element in cart_list:
            product = cart_element["product"]
            result.append(product)
            producer_id = cart_element["producer_id"]
            
            # Synchronization: Reclaiming producer publication slots.
            with self.producers_locks[producer_id]:
                self.producers_queue[producer_id] -= 1
        
        # Invariant: Cart is cleared upon successful order finalization.
        self.carts[cart_id] = []
        self.logger.info("Finished place_order(%d): Order placed: %s!", cart_id, result)
        return result


class TestMarketplace(unittest.TestCase):
    /**
     * @class TestMarketplace
     * @brief Validation suite for marketplace inventory consistency and capacity management.
     */
    
    def setUp(self):
        self.marketplace = Marketplace(5)
        self.product0 = Coffee(name="Indonezia", acidity="5.05", roast_level="MEDIUM", price=1)
        self.product1 = Tea(name="Linden", type="Herbal", price=9)
        self.product2 = Coffee(name="Ethiopia", acidity="5.09", roast_level="MEDIUM", price=10)
        self.product3 = Coffee(name="Arabica", acidity="5.02", roast_level="MEDIUM", price=9)

    def test_register_producer(self):
        self.assertEqual(self.marketplace.register_producer(), 'prod0')
        self.assertEqual(self.marketplace.register_producer(), 'prod1')

    def test_publish(self):
        self.test_register_producer()
        for _ in range(3):
            self.assertTrue(self.marketplace.publish('prod0', self.product0))
        
        # Block Logic: Quota enforcement check.
        self.assertEqual(self.marketplace.producers_queue['prod0'], 3)

    def test_add_to_cart(self):
        self.test_publish()
        id_cart = self.marketplace.new_cart()
        self.assertTrue(self.marketplace.add_to_cart(id_cart, self.product0))
        self.assertEqual(len(self.marketplace.carts[id_cart]), 1)

    def test_place_order(self):
        self.test_add_to_cart()
        order = self.marketplace.place_order(0)
        self.assertEqual(len(order), 1)
        self.assertEqual(self.marketplace.producers_queue['prod0'], 2)


from threading import Thread
from time import sleep


class Producer(Thread):
    /**
     * @class Producer
     * @brief Background thread responsible for replenishing marketplace stock.
     * 
     * Logic: Periodically publishes product units to its assigned marketplace slot. 
     * Implements a retry loop when the marketplace is at capacity for its specific ID.
     */

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, daemon=True)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.name = kwargs["name"]

    def run(self):
        # Invariant: Each producer obtains a unique supplier identity upon startup.
        producer_id = self.marketplace.register_producer()
        
        while True:
            for element in self.products:
                product, quantity, production_time = element
                
                # Logic: Simulated manufacturing delay.
                sleep(production_time)
                
                for _ in range(quantity):
                    # Block Logic: Publication retry.
                    while not self.marketplace.publish(producer_id, product):
                        sleep(self.republish_wait_time)


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
