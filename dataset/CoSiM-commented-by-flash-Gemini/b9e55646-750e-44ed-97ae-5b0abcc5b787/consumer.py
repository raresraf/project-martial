
"""
@file consumer.py
@brief Concurrent marketplace simulation implementing the Producer-Consumer pattern.

Functional Intent: Provides a thread-safe environment for multiple Producers to 
publish products and Consumers to acquire them via virtual shopping carts. 
Features multi-level locking for independent shared resources (Producers, 
Carts, and Inventory) and utilizes a custom `ProductDict` to manage concurrent 
inventory state transitions atomically.

Domain: Production Systems, Concurrency and Synchronization.
"""

from threading import Thread
import time


class Consumer(Thread):
    """
    @brief Represents a consumer entity that operates in its own execution thread.
    
    Logic: Sequentially processes assigned carts, attempting to add or remove 
    products until all target quantities are met, then finalizes the order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes the consumer with specific shopping tasks.
        @param carts List of carts containing operation requests.
        @param marketplace Reference to the central Marketplace instance.
        @param retry_wait_time Interval to wait when a requested item is out of stock.
        """
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def run(self):
        """
        @brief Core execution loop for the consumer thread.
        
        Algorithm: Iterative cart processing with nested retry polling.
        """
        market = self.marketplace
        for cart in self.carts:
            # Block Logic: Registration of a new shopping session.
            cart_id = self.marketplace.new_cart()
            for cart_ops in cart:
                
                # Block Logic: Batch acquisition loop.
                for _ in range(0, cart_ops['quantity']):
                    if cart_ops['type'] == 'add':
                        is_product_in_market = market.add_to_cart(cart_id, cart_ops['product'])
                        
                        # Invariant: Continues polling until the marketplace grants the reservation.
                        while not is_product_in_market:
                            # Optimization: Back-off wait to reduce CPU spin during contention.
                            time.sleep(self.retry_wait_time)
                            is_product_in_market = market.add_to_cart(cart_id, cart_ops['product'])
                    else:
                        # Block Logic: Batch restoration pass.
                        self.marketplace.remove_from_cart(cart_id, cart_ops['product'])

            # Functional Intent: Commits the transaction and logs purchase results.
            product_list = self.marketplace.place_order(cart_id)
            for product in product_list:
                print(self.name, "bought", product)

import logging
import time
import unittest


from threading import Lock
from logging.handlers import RotatingFileHandler
from tema.product_dict import ProductDict
from tema.product import Tea
from tema.product import Coffee


class Marketplace:
    """
    @brief Central broker for thread-safe item publishing and purchase fulfillment.
    
    Functional Utility: Manages global inventory via `ProductDict` and maintains 
    per-producer quotas. Utilizes multiple discrete locks to maximize concurrency 
    while preventing race conditions in ID generation and session management.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the marketplace with capacity limits and diagnostic logging.
        """
        self.queue_size_per_producer = queue_size_per_producer

        # Synchronization: Atomic ID generators and registry locks.
        self.next_producer_id = 1
        self.next_producer_id_lock = Lock()

        self.next_cart_id = 1
        self.next_cart_id_lock = Lock()

        # Logic: Dedicated thread-safe container for global product availability.
        self.market_products = ProductDict()

        self.producer_queue_sizes = {}
        self.producer_queue_sizes_lock = Lock()

        self.consumer_carts = {}
        self.consumer_carts_lock = Lock()
        
        # Configuration: Standardized rotating log handling.
        handler = RotatingFileHandler(
            'marketplace.log',
            mode='w',
            maxBytes=1000000,
            backupCount=1000,
            delay=True
        )

        logging.basicConfig(
            handlers=[handler],
            level=logging.INFO,
            format='%(asctime)s %(levelname)s : %(message)s'
        )

        logging.Formatter.converter = time.gmtime

    def register_producer(self):
        """
        @brief Onboards a new producer and initializes its inventory tracking context.
        """
        logging.info('Entering register_producer')
        with self.next_producer_id_lock:
            curr_producer_id = self.next_producer_id
            self.next_producer_id += 1

        with self.producer_queue_sizes_lock:
            self.producer_queue_sizes[curr_producer_id] = 0

        logging.info('Leaving register_producer')
        return curr_producer_id

    def publish(self, producer_id, product):
        """
        @brief Exposes a product to the marketplace if the producer's quota allows.
        """
        logging.info('Entering publish with producer_id=%d product=%s', producer_id, repr(product))

        # Block Logic: Quota verification.
        with self.producer_queue_sizes_lock:
            if self.producer_queue_sizes[producer_id] >= self.queue_size_per_producer:
                logging.info('Leaving publish')
                return False

        # Synchronization: Atomic insertion into the shared inventory.
        self.market_products.put(product, producer_id)

        with self.producer_queue_sizes_lock:
            self.producer_queue_sizes[producer_id] += 1

        logging.info('Leaving publish')
        return True

    def new_cart(self):
        """
        @brief Spawns a new shopping session for a consumer.
        """
        logging.info('Entering new_cart')
        with self.next_cart_id_lock:
            curr_cart_id = self.next_cart_id
            self.next_cart_id += 1


        # Initialization: Creates a private inventory map for the cart.
        with self.consumer_carts_lock:
            self.consumer_carts[curr_cart_id] = ProductDict()

        logging.info('Leaving new_cart')
        return curr_cart_id

    def get_cart(self, cart_id) -> ProductDict:
        """
        @brief Retrieves the thread-safe inventory handle for a specific cart session.
        """
        logging.info('Entering get_cart with cart_id=%d', cart_id)
        with self.consumer_carts_lock:
            logging.info('Leaving get_cart')
            return self.consumer_carts[cart_id]

    def add_to_cart(self, cart_id, product):
        """
        @brief Atomically transfers an item from the global pool to a specific cart.
        """
        logging.info('Entering add_to_cart with cart_id=%d product=%s', cart_id, repr(product))
        
        # Logic: Attempts to pop an instance from global inventory.
        producer_id = self.market_products.remove(product)

        if not producer_id:
            logging.info('Leaving add_to_cart')
            return False

        # Invariant: If removed from global, it must be added to the target cart.
        consumer_cart = self.get_cart(cart_id)
        consumer_cart.put(product, producer_id)

        logging.info('Leaving add_to_cart')
        return True

    def remove_from_cart(self, cart_id, product):
        """
        @brief Restores an item from a cart back to the general marketplace inventory.
        """
        log_message = 'Entering remove_from_cart with cart_id=%d product=%s'
        logging.info(log_message, cart_id, repr(product))
        consumer_cart = self.get_cart(cart_id)
        
        # Logic: Atomic transfer between local cart and global pool.
        producer_id = consumer_cart.remove(product)

        self.market_products.put(product, producer_id)
        logging.info('Leaving remove_from_cart')

    def place_order(self, cart_id):
        """
        @brief Finalizes the transaction, releasing producer quotas and clearing session.
        """
        logging.info('Entering place_order with cart_id=%d', cart_id)
        consumer_cart = self.get_cart(cart_id)
        product_list = []
        
        # Block Logic: Final checkout pass.
        for product in consumer_cart.dict:
            quantity_dict = consumer_cart.dict[product]

            # Logic: Reconciles producer inventory counters for every item sold.
            for producer_id in quantity_dict:
                quantity = quantity_dict[producer_id]
                for _ in range(0, quantity):
                    product_list.append(product)

                # Synchronization: Safe update of producer availability slots.
                with self.producer_queue_sizes_lock:
                    self.producer_queue_sizes[producer_id] -= quantity

        logging.info('Leaving place_order')
        return product_list


class TestMarketplace(unittest.TestCase):
    """
    @brief Unit tests for validating Marketplace state consistency and thread safety.
    """
    
    def setUp(self) -> None:
        self.marketplace = Marketplace(3)
        self.product1 = Tea("Linden", 9, "Linden")
        self.product2 = Coffee("Indonezia", 1, 5.05, 'MEDIUM')

    def test_register_producer(self):
        for i in range(1, 100):
            self.assertEqual(self.marketplace.register_producer(), i)

    def test_publish(self):
        self.marketplace.register_producer()

        self.assertEqual(self.marketplace.publish(1, self.product1), True)
        self.assertEqual(self.marketplace.publish(1, self.product1), True)
        self.assertEqual(self.marketplace.publish(1, self.product2), True)
        self.assertEqual(self.marketplace.producer_queue_sizes[1], 3)

        self.assertEqual(self.marketplace.publish(1, self.product2), False)
        self.assertEqual(self.marketplace.producer_queue_sizes[1], 3)
        market_products = {self.product1: {1: 2}, self.product2: {1: 1}}
        self.assertEqual(self.marketplace.market_products.dict, market_products)

        self.marketplace.register_producer()
        for _ in range(0, 10):
            self.marketplace.publish(2, self.product2)

        market_products[self.product2][2] = 3
        self.assertEqual(self.marketplace.market_products.dict, market_products)

    def test_new_cart(self):
        for i in range(1, 100):
            self.assertEqual(self.marketplace.new_cart(), i)

    def test_get_cart(self):
        self.marketplace.new_cart()
        self.assertEqual(self.marketplace.get_cart(1).dict, {})
        self.marketplace.register_producer()
        for i in range(1, 4):
            self.marketplace.publish(1, self.product1)
            self.marketplace.add_to_cart(1, self.product1)
            cart = {self.product1: {1: i}}
            self.assertEqual(self.marketplace.get_cart(1).dict, cart)

    def test_add_to_cart(self):
        self.marketplace.register_producer()
        self.marketplace.register_producer()
        self.marketplace.publish(1, self.product1)
        self.marketplace.publish(2, self.product1)
        self.marketplace.publish(2, self.product2)

        self.marketplace.new_cart()
        self.marketplace.add_to_cart(1, self.product1)
        self.marketplace.add_to_cart(1, self.product1)
        cart = {self.product1: {1: 1, 2: 1}}
        self.assertEqual(self.marketplace.get_cart(1).dict, cart)

        self.marketplace.add_to_cart(1, self.product2)
        self.assertEqual(self.marketplace.market_products.dict, {})
        cart = {self.product1: {1: 1, 2: 1}, self.product2: {2: 1}}
        self.assertEqual(self.marketplace.get_cart(1).dict, cart)

    def fill_cart(self):
        self.marketplace.register_producer()
        self.marketplace.register_producer()
        self.marketplace.new_cart()
        for _ in range(1, 4):
            self.marketplace.publish(1, self.product1)
            self.marketplace.publish(2, self.product2)
            self.marketplace.add_to_cart(1, self.product2)
            self.marketplace.add_to_cart(1, self.product1)

    def test_remove_from_cart(self):
        self.fill_cart()
        for i in range(0, 3):
            cart = {self.product1: {1: 3 - i}, self.product2: {2: 3}}
            self.assertEqual(self.marketplace.get_cart(1).dict, cart)
            self.marketplace.remove_from_cart(1, self.product1)
            market_products = {self.product1: {1: i + 1}}
            self.assertEqual(self.marketplace.market_products.dict, market_products)

        for i in range(0, 3):
            cart = {self.product2: {2: 3 - i}}
            self.assertEqual(self.marketplace.get_cart(1).dict, cart)
            self.marketplace.remove_from_cart(1, self.product2)
            market_products = {self.product1: {1: 3}, self.product2: {2: i + 1}}
            self.assertEqual(self.marketplace.market_products.dict, market_products)

        self.assertEqual(self.marketplace.get_cart(1).dict, {})

    def test_place_order(self):
        self.fill_cart()
        self.marketplace.remove_from_cart(1, self.product1)
        self.marketplace.remove_from_cart(1, self.product2)
        products = self.marketplace.place_order(1)
        product1_count = 0
        product2_count = 0
        for product in products:
            if product == self.product1:
                product1_count += 1

            if product == self.product2:
                product2_count += 1

        self.assertEqual(product1_count, 2)
        self.assertEqual(product2_count, 2)
        market_products = {self.product1: {1: 1}, self.product2: {2: 1}}
        self.assertEqual(self.marketplace.market_products.dict, market_products)
        self.assertEqual(self.marketplace.producer_queue_sizes[1], 1)
        self.assertEqual(self.marketplace.producer_queue_sizes[2], 1)

from threading import Thread
import time


class Producer(Thread):
    """
    @brief Represents a producer entity that generates items for the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes the producer with catalog and manufacturing schedule.
        """
        Thread.__init__(self)
        self.setDaemon(kwargs['daemon'])
        self.product_infos = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()
        self.name = kwargs['name']

    def run(self):
        """
        @brief Core production cycle for the thread.
        
        Logic: Continuously cycles through its catalog, manufacturing items with 
        defined delays and publishing them. Implements back-off polling if the 
        marketplace is full.
        """
        while True:
            for product_info in self.product_infos:
                (product, quantity, processing_time) = product_info
                
                # Block Logic: Manufacturing batch.
                for _ in range(0, quantity):
                    can_i_republish = self.marketplace.publish(self.producer_id, product)
                    # Optimization: Simulated manufacturing latency.
                    time.sleep(processing_time)
                    
                    if not can_i_republish:
                        # Block Logic: Quota-driven wait state.
                        time.sleep(self.republish_wait_time)

            time.sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Base immutable representation of a market item.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Specialized product type for tea.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Specialized product type for coffee.
    """
    acidity: float
    roast_level: str


from threading import Lock
from threading import Thread
import unittest
from tema.product import Tea
from tema.product import Coffee

class ProductDict:
    """
    @brief Thread-safe hierarchical mapping for concurrent inventory management.
    
    Functional Utility: Manages a nested dictionary (Product -> ProducerID -> Count). 
    Encapsulates mutex-protected logic to ensure that increments and decrements 
    remain consistent during multi-threaded access.
    """
    
    def __init__(self):
        self.dict = {}
        # Synchronization: Critical section protection for the internal data map.
        self.dict_lock = Lock()

    def put(self, product, producer_id):
        """
        @brief Atomically registers a product instance in the inventory.
        """
        with self.dict_lock:
            if product in self.dict:
                quantity_dict = self.dict[product]

                if producer_id in quantity_dict:
                    quantity_dict[producer_id] += 1
                else:
                    quantity_dict[producer_id] = 1
            else:
                self.dict[product] = {producer_id: 1}

    def remove(self, product):
        """
        @brief Atomically claims a product instance from any available producer.
        @return The ID of the producer from whom the item was taken, or None.
        """
        with self.dict_lock:
            if product not in self.dict:
                return None

            # Block Logic: First-available extraction.
            quantity_dict = self.dict[product]
            for producer_id in quantity_dict:
                quantity_dict[producer_id] -= 1
                producer_id_return = producer_id
                break

            # Garbage Collection: Prunes empty branches from the mapping tree.
            if quantity_dict[producer_id_return] == 0:
                quantity_dict.pop(producer_id_return)

            if not quantity_dict:
                self.dict.pop(product)

            return producer_id_return


class TestProductDict(unittest.TestCase):
    """
    @brief Stress tests for verifying thread-safety of the ProductDict container.
    """
    
    def setUp(self) -> None:
        self.product_dict = ProductDict()
        self.product1 = Tea("Linden", 9, "Herbal")
        self.product2 = Coffee("Indonezia", 1, 5.05, 'MEDIUM')

        def thread_run():
            for _ in range(0, 5):
                for j in range(1, 6):
                    self.product_dict.put(self.product1, j)
                    self.product_dict.put(self.product2, j + 1)

        threads = []
        for _ in range(0, 10):
            thread = Thread(target=thread_run)
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    def test_put(self):
        quantity_dict1 = {1: 50, 2: 50, 3: 50, 4: 50, 5: 50}
        quantity_dict2 = {2: 50, 3: 50, 4: 50, 5: 50, 6: 50}
        product_dict = {self.product1: quantity_dict1, self.product2: quantity_dict2}
        self.assertEqual(self.product_dict.dict, product_dict)

    def test_remove(self):
        product_ids1 = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        product_ids2 = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

        def thread_run():
            for _ in range(0, 5):
                for _ in range(0, 5):
                    product_id1 = self.product_dict.remove(self.product1)
                    product_ids1[product_id1] += 1
                    product_id2 = self.product_dict.remove(self.product2)
                    product_ids2[product_id2] += 1

        threads = []
        for _ in range(0, 10):
            thread = Thread(target=thread_run)
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        self.assertEqual(self.product_dict.dict, {})
        product_ids1_correct = {1: 50, 2: 50, 3: 50, 4: 50, 5: 50}
        product_ids2_correct = {2: 50, 3: 50, 4: 50, 5: 50, 6: 50}
        self.assertEqual(product_ids1, product_ids1_correct)
        self.assertEqual(product_ids2, product_ids2_correct)
        self.assertEqual(self.product_dict.remove(self.product1), None)
