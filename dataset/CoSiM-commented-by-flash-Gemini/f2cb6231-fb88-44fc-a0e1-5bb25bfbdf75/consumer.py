
/**
 * @file consumer.py
 * @brief Concurrent marketplace simulation with centralized resource management and logging.
 * 
 * Functional Intent: Provides a multi-threaded framework for simulating a marketplace 
 * where Producers inject items into a synchronized pool and Consumers perform 
 * transactional shopping operations. It leverages mutex-based synchronization 
 * to ensure that product reservations and sales are atomic across the entire 
 * distributed state (Producers and Carts).
 * 
 * Domain: Production Systems, Concurrency, Synchronized State Management.
 */

from threading import Thread
from time import sleep

class Consumer(Thread):
    /**
     * @class Consumer
     * @brief Customer thread that executes a series of scripted shopping carts.
     * 
     * Logic: For each cart, the consumer initializes a session with the 
     * marketplace and performs 'add' or 'remove' operations. It implements 
     * a blocking polling loop for unavailable stock, ensuring that all 
     * requested quantities are eventually satisfied before finalizing the order.
     */

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs
        self.name = self.kwargs['name']

    def run(self):
        /**
         * Block Logic: Shopping lifecycle loop.
         * Invariant: Each cart is processed as a separate transaction session 
         * in the marketplace.
         */
        for cart in self.carts:
            new_cart_id = self.marketplace.new_cart()
            self.marketplace.add_empty_cart(self.name, new_cart_id)

            for operation in cart:
                if operation['type'] == 'add':
                    # Block Logic: Atomic addition with busy-wait retry.
                    for _ in range(operation['quantity']):
                        is_available = self.marketplace \
                            .add_to_cart(new_cart_id, operation['product'])

                        while is_available is False:
                            # Logic: Stall if stock is exhausted; retry after backoff.
                            sleep(self.retry_wait_time)
                            is_available = self.marketplace \
                                .add_to_cart(new_cart_id, operation['product'])
                elif operation['type'] == 'remove':
                    # Logic: Reverses a reservation in the current cart session.
                    for _ in range(operation['quantity']):
                        self.marketplace.remove_from_cart(new_cart_id, operation['product'])

            # Functional Utility: Finalizes the transaction and commits items to the consumer.
            _ = self.marketplace.place_order(new_cart_id)


from threading import Lock
import unittest
import time
import logging
from logging.handlers import RotatingFileHandler
from tema.product import Coffee, Tea

class Marketplace:
    /**
     * @class Marketplace
     * @brief Thread-safe central registry for inventory, carts, and transactions.
     * 
     * Logic: Manages shared state dictionaries for producers and consumers. 
     * Uses independent locks for ID generation and a global 'cart_lock' to 
     * serialize concurrent access to the complex nested inventory structures.
     */
    
    def __init__(self, queue_size_per_producer):
        self.queue_size_per_producer = queue_size_per_producer

        # Synchronization: Mutexes for provisioning unique identifiers.
        self.producer_ids = 0
        self.cart_ids = 0
        self.producer_ids_lock = Lock()
        self.cart_ids_lock = Lock()
        
        # Invariant: Global lock for all inventory/cart mutation operations.
        self.cart_lock = Lock()
        
        self.carts_dict = {}
        self.producers_dict = {}
        self.prod_id = 0

        # Logging: Initializing production-ready audit trails via rotating files.
        logging.basicConfig(
            handlers=[RotatingFileHandler('marketplace.log', maxBytes=200000, backupCount=10)],
                        level=logging.INFO)

    @staticmethod
    def gmt_time():
        gmt = time.gmtime(time.time())
        gmt_time = str(gmt.tm_hour) + ':' + str(gmt.tm_min) + ':' + str(gmt.tm_sec)
        return gmt_time

    def register_producer(self):
        /**
         * register_producer - Assigns an identity and bootstraps tracking for a new provider.
         */
        logging.info('GMT %s --> New producer register', self.gmt_time())
        with self.producer_ids_lock:
            self.producer_ids += 1
        
        self.producers_dict[self.producer_ids] = {'products' : [], 'products_count' : 0}
        logging.info('GMT %s --> Producer was registered', self.gmt_time())
        return self.producer_ids

    def publish(self, producer_id, product):
        /**
         * publish - Adds a new unit to the provider's stock.
         * 
         * Logic: Rejects units if the per-producer capacity is reached. 
         * Stores units with an 'available' flag for future reservation.
         */
        logging.info('GMT %s --> Producer %s wants to publish the product: %s',
                     self.gmt_time(), producer_id, product)

        if self.producers_dict[producer_id]['products_count'] == self.queue_size_per_producer:
            logging.info('GMT %s --> Product couldn\'t be published. \
                        The producer\'s queue is full.', self.gmt_time())
            return False

        # Invariant: Products are tracked as objects with a boolean availability status.
        self.producers_dict[producer_id]['products_count'] += 1
        self.producers_dict[producer_id]['products'] \
            .append({'available': True, 'product': product})

        logging.info('GMT %s --> Product was published successfully', self.gmt_time())
        return True

    def new_cart(self):
        logging.info('GMT %s --> New cart to register.', self.gmt_time())
        with self.cart_ids_lock:
            self.cart_ids += 1
        logging.info('GMT %s --> The cart was registered successfully.', self.gmt_time())
        return self.cart_ids

    def add_empty_cart(self, consumer_name, new_cart_id):
        /**
         * add_empty_cart - Bootstraps a fresh shopping session for a consumer.
         */
        logging.info('GMT %s --> Consumer %s wants to receive an empty cart with id %s.',
                     self.gmt_time(), consumer_name, new_cart_id)
        self.carts_dict[new_cart_id] = {'consumer_name' : consumer_name, 'products' : []}
        logging.info('GMT %s --> Cart was assigned successfully.', self.gmt_time())

    def add_to_cart(self, cart_id, product):
        /**
         * add_to_cart - Atomically reserves a product unit for a consumer session.
         * 
         * Algorithm: Exhaustive search across all producers.
         * 1. Iterates through the producers_dict.
         * 2. Inspects available units for the target product.
         * 3. On match: flips 'available' to False and appends unit metadata to the cart.
         */
        logging.info('GMT %s --> Someone wants to add %s in the cart with id %s.',
                     self.gmt_time(), product, cart_id)

        with self.cart_lock:
            found_product = 0
            for producer_id, producer_info in self.producers_dict.items():
                if found_product == 1:
                    break

                products_list = producer_info['products']
                for product_info in products_list:
                    if product_info['product'] == product and product_info['available'] is True:
                        # Synchronization: Transition to 'reserved' state.
                        product_info['available'] = False
                        self.prod_id = producer_id
                        self.carts_dict[cart_id]['products'] \
                            .append({'producer_id': self.prod_id, 'product':product})
                        found_product = 1
                        break

        if found_product == 1:
            logging.info('GMT %s --> The product was successfully added to cart.',
                         self.gmt_time())
            return True

        logging.info('GMT %s --> The product couldn\' be added to cart.', self.gmt_time())
        return False

    def remove_from_cart(self, cart_id, product):
        /**
         * remove_from_cart - Discards a reservation and restores unit availability.
         */
        logging.info('GMT %s --> Someone wants to remove %s from the cart with id %s.',
                     self.gmt_time(), product, cart_id)

        with self.cart_lock:
            found_producer_id = 0
            consumer_products_list = self.carts_dict[cart_id]['products']

            # Block Logic: Reservation extraction.
            for product_info in consumer_products_list:
                if product_info['product'] == product:
                    consumer_products_list.remove(product_info)
                    found_producer_id = product_info['producer_id']
                    break

            # Block Logic: State restoration in source producer.
            producer_products_list = self.producers_dict[found_producer_id]['products']
            for product_info in producer_products_list:
                if product_info['product'] == product and product_info['available'] is False:
                    product_info['available'] = True
                    break

            logging.info('GMT %s --> The product was successfully removed from the cart.',
                         self.gmt_time())

    def place_order(self, cart_id):
        /**
         * place_order - Finalizes a transaction by committing all reserved items.
         * 
         * Logic: For every item in the cart, it removes the corresponding 
         * unit permanently from the producer's inventory and decrements the 
         * active producer quota.
         */
        logging.info('GMT %s --> Someone wants to place an order for cart with id %s.',
                     self.gmt_time(), cart_id)
        with self.cart_lock:
            # Block Logic: Permanent inventory removal.
            for cart_product_info in self.carts_dict[cart_id]['products']:
                producer_id = cart_product_info['producer_id']
                producer_list = self.producers_dict[producer_id]['products']

                for producer_product_info in producer_list:
                    if producer_product_info['product'] == cart_product_info['product'] \
                    and producer_product_info['available'] is False:
                        producer_list.remove(producer_product_info)
                        self.producers_dict[producer_id]['products_count'] -= 1
                        break

            # Output: Print summary of purchased items.
            for product in self.carts_dict[cart_id]['products']:
                print(f"{self.carts_dict[cart_id]['consumer_name']} bought {product['product']}")

            logging.info('GMT %s --> The order was successfully placed.', self.gmt_time())
            return self.carts_dict[cart_id]['products']

class MarketplaceTestCase(unittest.TestCase):
    /**
     * @class MarketplaceTestCase
     * @brief Automated validation suite for marketplace atomicity and capacity limits.
     */
    
    product1 = Tea(name='Linden', price=9, type='Herbal')
    product2 = Coffee(name='Indonezia', price=1, acidity=5.05, roast_level='MEDIUM')
    product1_info = (product1, 2, 0.18)
    product2_info = (product2, 1, 0.23)
    republish_wait_time = 0.15
    kwargs = {'name': 'prod1', 'daemon': True}

    def setUp(self):
        self.marketplace = Marketplace(queue_size_per_producer=3)

    def test_register_producer(self):
        id_producer = self.marketplace.register_producer()
        self.assertEqual(id_producer, 1)
        self.assertEqual(len(self.marketplace.producers_dict), 1)

        id_producer = self.marketplace.register_producer()
        self.assertEqual(id_producer, 2)
        self.assertEqual(len(self.marketplace.producers_dict), 2)

        id_producer = self.marketplace.register_producer()
        self.assertEqual(id_producer, 3)
        self.assertEqual(len(self.marketplace.producers_dict), 3)

    def test_publish(self):
        _ = self.marketplace.register_producer()
        _ = self.marketplace.register_producer()
        _ = self.marketplace.register_producer()

        mock_mp = Marketplace(queue_size_per_producer=3)
        mock_mp.producers_dict = \
            {1: {'products': [], 'products_count': 0}, \
            2: {'products': [], 'products_count': 0}, \
            3: {'products': [], 'products_count': 0}}

        mock_mp.producers_dict[1]['products'].append({'available': True, 'product':self.product1})
        mock_mp.producers_dict[1]['products_count'] = 1
        published = self.marketplace.publish(1, self.product1)
        self.assertTrue(published)
        self.assertEqual(self.marketplace.producers_dict, mock_mp.producers_dict)

        mock_mp.producers_dict[1]['products'].append({'available': True, 'product':self.product1})
        mock_mp.producers_dict[1]['products_count'] = 2
        published = self.marketplace.publish(1, self.product1)
        self.assertTrue(published)
        self.assertEqual(self.marketplace.producers_dict, mock_mp.producers_dict)

        mock_mp.producers_dict[1]['products'].append({'available': True, 'product':self.product2})
        mock_mp.producers_dict[1]['products_count'] = 3
        published = self.marketplace.publish(1, self.product2)
        self.assertTrue(published)
        self.assertEqual(self.marketplace.producers_dict, mock_mp.producers_dict)

        published = self.marketplace.publish(1, self.product2)
        self.assertFalse(published)
        self.assertEqual(self.marketplace.producers_dict, mock_mp.producers_dict)

        mock_mp.producers_dict[3]['products'].append({'available': True, 'product':self.product2})
        mock_mp.producers_dict[3]['products_count'] = 1
        published = self.marketplace.publish(3, self.product2)
        self.assertTrue(published)
        self.assertEqual(self.marketplace.producers_dict, mock_mp.producers_dict)

    def test_new_cart(self):
        id_cart = self.marketplace.new_cart()
        self.assertEqual(id_cart, 1)

        id_cart = self.marketplace.new_cart()
        self.assertEqual(id_cart, 2)

        id_cart = self.marketplace.new_cart()
        self.assertEqual(id_cart, 3)

    def test_add_empty_cart(self):
        mock_mp = Marketplace(queue_size_per_producer=3)
        mock_mp.carts_dict = {}

        mock_mp.carts_dict[1] = {'consumer_name':'cons1', 'products':[]}
        self.marketplace.add_empty_cart('cons1', 1)
        self.assertEqual(self.marketplace.carts_dict, mock_mp.carts_dict)

        mock_mp.carts_dict[2] = {'consumer_name':'cons1', 'products':[]}
        self.marketplace.add_empty_cart('cons1', 2)
        self.assertEqual(self.marketplace.carts_dict, mock_mp.carts_dict)

        mock_mp.carts_dict[3] = {'consumer_name':'cons2', 'products':[]}
        self.marketplace.add_empty_cart('cons2', 3)
        self.assertEqual(self.marketplace.carts_dict, mock_mp.carts_dict)

    def test_add_to_cart(self):
        mock_mp = Marketplace(queue_size_per_producer=3)
        mock_producers_dict = \
            {1: {'products': [{'available': True, 'product':self.product1}, \
                            {'available': True, 'product':self.product2}], \
                'products_count': 2}, \
            2: {'products': [{'available': True, 'product':self.product1}], 'products_count': 1}, \
            3: {'products': [{'available': True, 'product':self.product2}], 'products_count': 1}}
        mock_carts_dict = \
            {1: {'consumer_name': 'cons1', 'products': []}, \
            2: {'consumer_name': 'cons1', 'products': []}, \
            3: {'consumer_name': 'cons2', 'products': []}}
        mock_mp.producers_dict = mock_producers_dict
        mock_mp.carts_dict = mock_carts_dict
        self.marketplace.producers_dict = mock_producers_dict
        self.marketplace.carts_dict = mock_carts_dict

        mock_mp.producers_dict[1]['products'] =  \
            [{'available': False, 'product':self.product1}, \
            {'available': True, 'product':self.product2}]
        mock_mp.carts_dict[1]['products'].append({'producer_id': 1, 'product':self.product1})
        added_to_cart = self.marketplace.add_to_cart(1, self.product1)
        self.assertTrue(added_to_cart)
        self.assertEqual(self.marketplace.producers_dict, mock_mp.producers_dict)

    def test_place_order(self):
        mock_mp = Marketplace(queue_size_per_producer=3)
        mock_producers_dict = \
            {1: {'products': [{'available': False, 'product':self.product1}, \
                            {'available': False, 'product':self.product2}], \
                'products_count': 2}}
        mock_carts_dict = \
            {1: {'consumer_name': 'cons1', \
                'products': [{'producer_id' : 1, 'product' : self.product1}, \
                            {'producer_id' : 1, 'product' : self.product2}]}}
        mock_mp.producers_dict = mock_producers_dict
        mock_mp.carts_dict = mock_carts_dict
        self.marketplace.producers_dict = mock_producers_dict
        self.marketplace.carts_dict = mock_carts_dict

        order = self.marketplace.place_order(1)
        order_result = []
        for prod in order:
            order_result.append(prod['product'])
        expected_order = [self.product1, self.product2]
        self.assertEqual(order_result, expected_order)


from threading import Thread
from time import sleep

class Producer(Thread):
    /**
     * @class Producer
     * @brief Background thread that periodically populates the marketplace inventory.
     * 
     * Logic: For every product type in its assignment, it attempts to publish units 
     * until the quantity is satisfied. It implements an exponential backoff 
     * potential via a wait loop when the marketplace is full.
     */

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs
        self.name = self.kwargs['name']
        self.id_producer = self.marketplace.register_producer()

    def run(self):
        while True:
            for (product_info, quantity, production_time) in self.products:
                for _ in range(quantity):
                    # Block Logic: Publication retry.
                    while True:
                        can_publish = self.marketplace.publish(self.id_producer, product_info)
                        if can_publish:
                            # Logic: Simulated production delay.
                            sleep(production_time)
                            break
                        # Logic: Wait for capacity to become available in the market.
                        sleep(self.republish_wait_time)
