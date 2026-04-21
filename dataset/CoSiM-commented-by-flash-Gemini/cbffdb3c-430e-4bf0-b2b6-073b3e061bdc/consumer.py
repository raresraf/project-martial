
/**
 * @file consumer.py
 * @brief Thread-safe marketplace simulation with inventory tracking and transactional order placement.
 * 
 * Functional Intent: Implements a concurrent marketplace where Producers publish 
 * items and Consumers purchase them via virtual carts. The system uses a 
 * specialized `ProductDict` to manage stock counts per producer with fine-grained 
 * locking, ensuring atomic transitions between available, reserved, and sold states.
 * 
 * Domain: Production Systems, Concurrency, Inventory Management.
 */

from threading import Thread
import time


class Consumer(Thread):
    /**
     * @class Consumer
     * @brief Customer thread that executes a series of shopping operations across multiple carts.
     * 
     * Logic: For each cart, the consumer attempts to 'add' products to their virtual 
     * reservation. If a product is unavailable, the thread enters a retry loop with 
     * a configured delay. Once all operations for a cart are satisfied, the order 
     * is finalized (place_order).
     */

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def run(self):
        /**
         * Block Logic: Shopping lifecycle management.
         * Invariant: Every product added to a cart is either purchased 
         * or explicitly returned to the market.
         */
        market = self.marketplace
        for cart in self.carts:
            # Logic: Acquires a fresh cart ID for each set of operations.
            cart_id = self.marketplace.new_cart()
            for cart_ops in cart:
                for _ in range(0, cart_ops['quantity']):
                    if cart_ops['type'] == 'add':
                        # Block Logic: Transactional add with retry.
                        is_product_in_market = market.add_to_cart(cart_id, cart_ops['product'])
                        while not is_product_in_market:
                            time.sleep(self.retry_wait_time)
                            is_product_in_market = market.add_to_cart(cart_id, cart_ops['product'])
                    else:
                        # Logic: Explicit return of a reserved item.
                        self.marketplace.remove_from_cart(cart_id, cart_ops['product'])

            # Functional Utility: Commits all cart reservations to the final sale.
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
    /**
     * @class Marketplace
     * @brief Central coordinator for multi-producer inventory and multi-consumer sales.
     * 
     * Logic: Orchestrates shared state across multiple locks to prevent contention. 
     * Tracks producer quotas to prevent overflow of the marketplace buffer.
     */
    
    def __init__(self, queue_size_per_producer):
        self.queue_size_per_producer = queue_size_per_producer

        self.next_producer_id = 1
        self.next_producer_id_lock = Lock()

        self.next_cart_id = 1
        self.next_cart_id_lock = Lock()

        # Invariant: Central inventory stored in a thread-safe custom dictionary.
        self.market_products = ProductDict()

        self.producer_queue_sizes = {}
        self.producer_queue_sizes_lock = Lock()

        self.consumer_carts = {}
        self.consumer_carts_lock = Lock()
        
        # Logging: Initializing audit trail for all marketplace transactions.
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
        /**
         * register_producer - Bootstraps a new producer identity and tracking state.
         */
        logging.info('Entering register_producer')
        with self.next_producer_id_lock:
            curr_producer_id = self.next_producer_id
            self.next_producer_id += 1

        with self.producer_queue_sizes_lock:
            self.producer_queue_sizes[curr_producer_id] = 0

        logging.info('Leaving register_producer')
        return curr_producer_id

    def publish(self, producer_id, product):
        /**
         * publish - Injects a new product unit into the available pool.
         * 
         * Logic: Rejects publications that would exceed the producer's allotted 
         * quota in the marketplace.
         */
        logging.info('Entering publish with producer_id=%d product=%s', producer_id, repr(product))

        with self.producer_queue_sizes_lock:
            if self.producer_queue_sizes[producer_id] >= self.queue_size_per_producer:
                logging.info('Leaving publish')
                return False

        # Synchronization: Atomic insertion into shared inventory.
        self.market_products.put(product, producer_id)

        with self.producer_queue_sizes_lock:
            self.producer_queue_sizes[producer_id] += 1

        logging.info('Leaving publish')
        return True

    def new_cart(self):
        logging.info('Entering new_cart')
        with self.next_cart_id_lock:
            curr_cart_id = self.next_cart_id
            self.next_cart_id += 1

        with self.consumer_carts_lock:
            # Invariant: Each cart is backed by its own ProductDict for thread-safe reservations.
            self.consumer_carts[curr_cart_id] = ProductDict()

        logging.info('Leaving new_cart')
        return curr_cart_id

    def get_cart(self, cart_id) -> ProductDict:
        logging.info('Entering get_cart with cart_id=%d', cart_id)
        with self.consumer_carts_lock:
            logging.info('Leaving get_cart')
            return self.consumer_carts[cart_id]

    def add_to_cart(self, cart_id, product):
        /**
         * add_to_cart - Atomically moves a product from the market to a specific cart.
         * 
         * Logic: Removes the item from global inventory (blocking others) and 
         * places it into the consumer's private reservation list.
         */
        logging.info('Entering add_to_cart with cart_id=%d product=%s', cart_id, repr(product))
        producer_id = self.market_products.remove(product)

        if not producer_id:
            logging.info('Leaving add_to_cart')
            return False

        consumer_cart = self.get_cart(cart_id)
        consumer_cart.put(product, producer_id)

        logging.info('Leaving add_to_cart')
        return True

    def remove_from_cart(self, cart_id, product):
        /**
         * remove_from_cart - Discards a reservation and returns the item to global stock.
         */
        log_message = 'Entering remove_from_cart with cart_id=%d product=%s'
        logging.info(log_message, cart_id, repr(product))
        consumer_cart = self.get_cart(cart_id)
        producer_id = consumer_cart.remove(product)

        self.market_products.put(product, producer_id)
        logging.info('Leaving remove_from_cart')

    def place_order(self, cart_id):
        /**
         * place_order - Converts all reservations in a cart into permanent sales.
         * 
         * Logic: For each item bought, it decrements the producer's active item 
         * count in the marketplace, freeing up space for new publications.
         */
        logging.info('Entering place_order with cart_id=%d', cart_id)
        consumer_cart = self.get_cart(cart_id)
        product_list = []
        for product in consumer_cart.dict:
            quantity_dict = consumer_cart.dict[product]
            for producer_id in quantity_dict:
                quantity = quantity_dict[producer_id]
                for _ in range(0, quantity):
                    product_list.append(product)

                # Logic: Quota reclamation after sale finalization.
                with self.producer_queue_sizes_lock:
                    self.producer_queue_sizes[producer_id] -= quantity

        logging.info('Leaving place_order')
        return product_list


class TestMarketplace(unittest.TestCase):
    /**
     * @class TestMarketplace
     * @brief Validation suite for marketplace inventory consistency and thread safety.
     */
    
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
    /**
     * @class Producer
     * @brief Background worker thread that populates the marketplace with new stock.
     */

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self)
        self.setDaemon(kwargs['daemon'])
        self.product_infos = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # Invariant: Each producer registers once to obtain a unique ID for publication.
        self.producer_id = self.marketplace.register_producer()
        self.name = kwargs['name']

    def run(self):
        /**
         * Block Logic: Production cycle.
         * Invariant: Continuously attempts to publish items, sleeping for 
         * 'processing_time' per success and 'republish_wait_time' per failure.
         */
        while True:
            for product_info in self.product_infos:
                (product, quantity, processing_time) = product_info
                for _ in range(0, quantity):
                    can_i_republish = self.marketplace.publish(self.producer_id, product)
                    time.sleep(processing_time)
                    if not can_i_republish:
                        # Logic: Wait if marketplace is currently at capacity.
                        time.sleep(self.republish_wait_time)

            time.sleep(self.republish_wait_time)


from dataclasses import dataclass

/**
 * @dataclass Product
 * @brief Base immutable descriptor for marketplace items.
 */
@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    acidity: float
    roast_level: str


from threading import Lock
from threading import Thread
import unittest
from tema.product import Tea
from tema.product import Coffee

class ProductDict:
    /**
     * @class ProductDict
     * @brief Specialized thread-safe container for mapping products to their source producers.
     * 
     * Logic: Maintains a nested dictionary structure [Product][ProducerID] -> Count. 
     * Uses internal mutex for all mutation operations to ensure atomicity in a 
     * concurrent environment.
     */
    
    def __init__(self):
        self.dict = {}
        self.dict_lock = Lock()

    def put(self, product, producer_id):
        /**
         * put - Increments the stock count for a specific product/producer pair.
         */
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
        /**
         * remove - Atomically extracts a single unit of a product.
         * 
         * Logic: Identifies the first available producer for the product, 
         * decrements the count, and prunes empty dictionary nodes to maintain 
         * minimal memory footprint.
         */
        with self.dict_lock:
            if product not in self.dict:
                return None

            quantity_dict = self.dict[product]
            for producer_id in quantity_dict:
                quantity_dict[producer_id] -= 1
                producer_id_return = producer_id
                break

            # Block Logic: State pruning.
            if quantity_dict[producer_id_return] == 0:
                quantity_dict.pop(producer_id_return)

            if not quantity_dict:
                self.dict.pop(product)

            return producer_id_return


class TestProductDict(unittest.TestCase):
    /**
     * @class TestProductDict
     * @brief Stress test for the ProductDict concurrent insertion and removal logic.
     */
    
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
