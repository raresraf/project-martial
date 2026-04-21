
/**
 * @file consumer.py
 * @brief Multi-threaded marketplace simulation with concurrent inventory and transaction management.
 * 
 * Functional Intent: Orchestrates a producer-consumer marketplace where multiple 
 * entities (Producers) populate a central inventory and multiple customers 
 * (Consumers) perform transactional cart operations (add/remove) and final 
 * order placement. It employs mutex-based synchronization to ensure atomic updates 
 * to shared state and implements a retry mechanism for saturated producer queues.
 * 
 * Domain: Production Systems, Concurrency, Producer-Consumer Pattern.
 */

from threading import Thread
from time import sleep


class Consumer(Thread):
    /**
     * @class Consumer
     * @brief Individual customer thread that processes a list of shopping carts.
     * 
     * Logic: Iteratively performs 'add' or 'remove' operations against the 
     * central marketplace. Employs a retry loop with exponential backoff (sleep) 
     * when the marketplace is temporarily unable to fulfill a request (e.g., stock 
     * unavailable).
     */

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        super().__init__(**kwargs)
        self.marketplace = marketplace
        self.carts = carts
        self.retry_wait_time = retry_wait_time
        # Invariant: Each consumer receives a unique cart identifier upon registration.
        self.cart_id = self.marketplace.new_cart()

    def run(self):
        /**
         * Block Logic: Shopping execution loop.
         * Invariant: For every operation in every cart, the consumer persists 
         * until the requested quantity is successfully fulfilled.
         */
        for cart in self.carts:
            for operation in cart:
                op_type = operation['type']
                product = operation['product']
                quantity = operation['quantity']
                while True:
                    # Logic: Dispatch to marketplace based on operation intent.
                    op_res = self.marketplace.add_to_cart(self.cart_id, product) \
                                if op_type == 'add' \
                                else self.marketplace.remove_from_cart(self.cart_id, product)

                    if op_res:
                        quantity -= 1
                    else:
                        # Logic: Wait before retrying if marketplace operation failed.
                        sleep(self.retry_wait_time)

                    if quantity == 0:
                        break

            # Functional Utility: Commits all cart reservations into a finalized order.
            items_bought = self.marketplace.place_order(self.cart_id)
            if len(items_bought) > 0:
                with self.marketplace.print_lock:
                    print('\n'.join(items_bought))

import time
from threading import Lock
from unittest import TestCase
import logging
import logging.handlers

from tema.product import Tea, Coffee


class Marketplace:
    /**
     * @class Marketplace
     * @brief Thread-safe central authority for product registration, inventory tracking, and sales.
     * 
     * Logic: Manages shared dictionaries of products, producer limits, and consumer carts. 
     * Uses 'market_lock' to prevent race conditions during inventory updates and 
     * 'print_lock' to synchronize console output.
     */
    
    def __init__(self, queue_size_per_producer):
        # Logging: Initializing rotating log handlers for audit trails.
        self.logger = logging.getLogger('marketplace_logger')
        self.logger.setLevel(logging.INFO)
        rotating_handler = logging.handlers.RotatingFileHandler('marketplace.log',
                                                                maxBytes=10000,
                                                                backupCount=10)

        formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%dT%H:%M:%S')
        formatter.converter = time.gmtime
        rotating_handler.setFormatter(formatter)
        self.logger.addHandler(rotating_handler)

        self.market_lock = Lock()
        self.print_lock = Lock()

        self.queue_size = queue_size_per_producer
        self.producer_items_count = {}
        self.consumer_id_count = 1
        self.producer_id_count = 1

        self.products = {}
        self.carts = {}

        self.all_products = {}

    def register_producer(self):
        /**
         * register_producer - Assigns a unique ID to a new inventory provider.
         */
        with self.market_lock:
            self.logger.info('entering register_producer')
            producer_id = f'producer{self.producer_id_count}'
            self.producer_items_count[producer_id] = 0
            self.products[producer_id] = {}
            self.producer_id_count += 1

            self.logger.info('leaving register_producer')
            return producer_id

    def publish(self, producer_id, product):
        /**
         * publish - Adds a new product unit to a producer's available stock.
         * 
         * Logic: Enforces the 'queue_size' per-producer limit to prevent 
         * inventory overflows.
         */
        self.logger.info('entering publish with args: {%s}, {%s}', producer_id, str(product))
        if self.producer_items_count[producer_id] == self.queue_size:
            self.logger.info('leaving publish')
            return False

        if product.name not in self.all_products:
            self.all_products[product.name] = product

        with self.market_lock:
            self.producer_items_count[producer_id] += 1
            if product.name not in self.products[producer_id]:
                # Invariant: Value is (available_units, reserved_units).
                self.products[producer_id][product.name] = (1, 0)
            else:
                num_items, reserved_items = self.products[producer_id][product.name]
                self.products[producer_id][product.name] = num_items + 1, reserved_items

        self.logger.info('leaving publish')
        return True

    def new_cart(self):
        with self.market_lock:
            self.logger.info('entering new_cart')
            cart_id = f'cons{self.consumer_id_count}'
            self.carts[cart_id] = {}
            self.consumer_id_count += 1

            self.logger.info('leaving new_cart')
            return cart_id

    def add_to_cart(self, cart_id, product):
        /**
         * add_to_cart - Transitions a product unit from 'available' to 'reserved' status.
         * 
         * Logic: Performs a sweep across all producers to find stock. If found, 
         * it increments the reservation count in the producer's inventory and 
         * adds the producer/product pair to the consumer's cart.
         */
        self.logger.info('entering add_to_cart with args: {%s}, {%s}', cart_id, str(product))
        for producer_id, producer_products in self.products.items():
            if product.name in producer_products:
                num_items, reserved_items = producer_products[product.name]
                with self.market_lock:
                    if reserved_items < num_items:
                        # Synchronization: Atomic transition to 'reserved' state.
                        producer_products[product.name] = (num_items, reserved_items + 1)

                        if product.name not in self.carts[cart_id]:
                            self.carts[cart_id][product.name] = {}

                        if producer_id not in self.carts[cart_id][product.name]:
                            self.carts[cart_id][product.name][producer_id] = 1
                        else:
                            self.carts[cart_id][product.name][producer_id] += 1

                        self.logger.info('leaving add_to_cart')
                        return True

        self.logger.info('leaving add_to_cart')
        return False

    def remove_from_cart(self, cart_id, product):
        /**
         * remove_from_cart - Reverses a reservation, making the unit available for others.
         */
        self.logger.info('entering remove_from_cart with args: {%s}, {%s}',
                         cart_id, str(product))
        deleted_producer_id = None
        for producer_id in self.carts[cart_id][product.name]:
            with self.market_lock:
                if self.carts[cart_id][product.name][producer_id] > 0:
                    deleted_producer_id = producer_id
                    self.carts[cart_id][product.name][producer_id] -= 1
                    if self.carts[cart_id][product.name][producer_id] == 0:
                        del self.carts[cart_id][product.name][producer_id]
                    if len(self.carts[cart_id][product.name]) == 0:
                        del self.carts[cart_id][product.name]

                    break

        if deleted_producer_id is None:
            self.logger.info('leaving remove_from_cart')
            return False

        with self.market_lock:
            num_items, reserved_items = self.products[deleted_producer_id][product.name]
            self.products[deleted_producer_id][product.name] = num_items, reserved_items - 1

        self.logger.info('leaving remove_from_cart')
        return True

    def place_order(self, cart_id):
        /**
         * place_order - Finalizes a transaction by removing reserved items from global counts.
         * 
         * Logic: Converts all reservations in a cart into a permanent purchase, 
         * updating the total item counts for each source producer.
         */
        self.logger.info('entering place_order with args: {%s}', cart_id)
        items_bought = []
        for product_name in self.carts[cart_id]:
            for producer_id, num_reserved in self.carts[cart_id][product_name].items():
                with self.market_lock:
                    num_items, reserved_items = self.products[producer_id][product_name]
                    # Logic: Decrements both physical stock and reservation count.
                    self.products[producer_id][product_name] = \
                        (num_items - num_reserved, reserved_items - num_reserved)
                    self.producer_items_count[producer_id] -= num_reserved
                    for _ in range(num_reserved):
                        items_bought.append(f'{cart_id} bought {self.all_products[product_name]}')

        # Invariant: Clears the cart after order fulfillment.
        self.carts[cart_id] = {}

        self.logger.info('leaving place_order')
        return items_bought


class TestMarketplace(TestCase):
    /**
     * @class TestMarketplace
     * @brief Unit testing suite to verify marketplace atomicity and inventory limits.
     */
    
    def setUp(self):
        self.marketplace = Marketplace(2)
        self.first_prd_id = self.marketplace.register_producer()
        self.second_prd_id = self.marketplace.register_producer()

        self.first_cart_id = self.marketplace.new_cart()
        self.second_cart_id = self.marketplace.new_cart()

        self.fake_products = {'first_tea': Tea('Green', 2, 'Good'),
                              'second_tea': Tea('Black', 3, 'Bad'),
                              'first_coffee': Coffee('Brazilian', 5, 'high', 'high')}

    def test_register_producer(self):
        first_producer_id = self.marketplace.register_producer()
        self.assertEqual(first_producer_id, 'producer3')
        self.assertEqual(self.marketplace.producer_items_count[first_producer_id], 0)
        self.assertTrue(first_producer_id in self.marketplace.products)
        self.assertEqual(self.marketplace.producer_id_count, 4)

        second_producer_id = self.marketplace.register_producer()
        self.assertEqual(second_producer_id, 'producer4')
        self.assertEqual(self.marketplace.producer_id_count, 5)

    def test_publish(self):
        first_product = self.fake_products['first_tea']
        result = self.marketplace.publish(self.first_prd_id, first_product)
        self.assertEqual(result, True)
        self.assertTrue(first_product.name in self.marketplace.all_products)
        self.assertEqual(self.marketplace.products[self.first_prd_id][first_product.name], (1, 0))

        result = self.marketplace.publish(self.first_prd_id, first_product)
        self.assertEqual(result, True)
        self.assertEqual(self.marketplace.products[self.first_prd_id][first_product.name], (2, 0))

        result = self.marketplace.publish(self.first_prd_id, first_product)
        self.assertEqual(result, False)

        second_product = self.fake_products['first_coffee']
        result = self.marketplace.publish(self.second_prd_id, second_product)
        self.assertEqual(result, True)
        self.assertEqual(self.marketplace.products[self.second_prd_id][second_product.name],
                         (1, 0))

    def test_new_cart(self):
        first_cart_id = self.marketplace.new_cart()
        self.assertEqual(first_cart_id, 'cons3')
        self.assertTrue(first_cart_id in self.marketplace.carts)
        self.assertEqual(self.marketplace.consumer_id_count, 4)

        second_producer_id = self.marketplace.new_cart()
        self.assertEqual(second_producer_id, 'cons4')
        self.assertEqual(self.marketplace.consumer_id_count, 5)

    def test_add_to_cart(self):
        self.marketplace.publish(self.first_prd_id, self.fake_products['first_tea'])
        found_res = self.marketplace.add_to_cart(self.first_cart_id,
                                                 self.fake_products['first_tea'])
        self.assertTrue(found_res)
        self.assertEqual(self.marketplace.products[self.first_prd_id]['Green'],
                         (1, 1))
        self.assertTrue('Green' in self.marketplace.carts[self.first_cart_id])
        self.assertEqual(self.marketplace.carts[self.first_cart_id]['Green'][self.first_prd_id],
                         1)

        add_again_res = self.marketplace.add_to_cart(self.first_cart_id,
                                                     self.fake_products['first_tea'])
        self.assertFalse(add_again_res)

        self.marketplace.publish(self.first_prd_id, self.fake_products['first_tea'])
        found_again = self.marketplace.add_to_cart(self.first_cart_id,
                                                   self.fake_products['first_tea'])
        self.assertTrue(found_again)
        self.assertEqual(self.marketplace.products[self.first_prd_id]['Green'],
                         (2, 2))
        self.assertEqual(self.marketplace.carts[self.first_cart_id]['Green'][self.first_prd_id],
                         2)

    def test_remove_from_cart(self):
        self.marketplace.publish(self.first_prd_id, self.fake_products['first_tea'])
        self.marketplace.add_to_cart(self.first_cart_id, self.fake_products['first_tea'])

        result = self.marketplace.remove_from_cart(self.first_cart_id,
                                                   self.fake_products['first_tea'])
        self.assertTrue(result)
        self.assertEqual(self.marketplace.products[self.first_prd_id]['Green'],
                         (1, 0))
        self.assertTrue('Green' not in self.marketplace.carts[self.first_cart_id])

    def test_place_order(self):
        self.marketplace.publish(self.first_prd_id, self.fake_products['first_tea'])
        self.marketplace.publish(self.first_prd_id, self.fake_products['first_coffee'])
        self.marketplace.add_to_cart(self.first_cart_id, self.fake_products['first_tea'])
        self.marketplace.add_to_cart(self.first_cart_id, self.fake_products['second_tea'])

        items_bought = self.marketplace.place_order(self.first_cart_id)
        self.assertTrue(str(self.fake_products['first_tea']) in items_bought[0])


from threading import Thread
from time import sleep


class Producer(Thread):
    /**
     * @class Producer
     * @brief Individual inventory provider thread that periodically populates the marketplace.
     */

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        super().__init__(**kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        /**
         * Block Logic: Production cycle.
         * Invariant: Continuously attempts to publish products until their 
         * respective quotas are reached.
         */
        while True:
            for product, quantity, sleep_time in self.products:
                produce_res = self.marketplace.publish(self.producer_id, product)
                if produce_res:
                    quantity -= 1
                    sleep(sleep_time)
                else:
                    # Logic: Wait if the producer's marketplace queue is full.
                    sleep(self.republish_wait_time)

                if quantity == 0:
                    break


from dataclasses import dataclass

/**
 * @dataclass Product
 * @brief Base immutable record for items in the marketplace.
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
    acidity: str
    roast_level: str
