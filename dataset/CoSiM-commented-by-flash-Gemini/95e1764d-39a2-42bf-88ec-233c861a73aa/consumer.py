"""
@95e1764d-39a2-42bf-88ec-233c861a73aa/consumer.py
@brief Event-driven simulation of a retail marketplace using concurrent Producer and Consumer threads.
Architecture: Multi-threaded actor-based model where a centralized Marketplace mediator manages shared state.
Functional Utility: Facilitates asynchronous inventory flow, transaction processing, and session-based order fulfillment.
Synchronization: Employs fine-grained locking (threading.Lock) to protect critical sections and polling retries for demand-supply flow control.
"""

import time
from threading import Thread


class Consumer(Thread):
    """
    @brief Consumer agent responsible for executing sequential shopping sessions.
    Logic: Orchestrates cart fulfillment by iteratively attempting to acquire requested commodities from the Marketplace.
    Error Handling: Implements a polling retry mechanism with yields (sleep) for handling temporary stock depletions.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @param carts List of product acquisition lists to be fulfilled.
        @param marketplace Shared resource mediator.
        @param retry_wait_time Duration to wait when the marketplace is depleted.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        @brief Lifecycle manager for the consumer thread.
        Logic: Allocates a new transactional session (cart_id) and fulfills all queued commands before finalization.
        """

        for cart in self.carts:
            # Initialization: establishes a unique inventory buffer in the marketplace.
            cart_id = self.marketplace.new_cart()
            for operation in cart:
                # Block Logic: Dispatcher for marketplace operations.
                if operation['type'] == 'add':
                    for _ in range(operation['quantity']):
                        # Synchronization: Fulfillment barrier. Ensures the thread blocks until units are reserved.
                        success = self.marketplace.add_to_cart(cart_id, operation['product'])
                        while not success:


                            time.sleep(self.retry_wait_time)
                            success = self.marketplace.add_to_cart(cart_id, operation['product'])
                else:
                    for _ in range(operation['quantity']):
                        # Logic: Returns reserved commodities to the marketplace inventory.
                        self.marketplace.remove_from_cart(cart_id, operation['product'])

            # Finalization: Executes the transaction and outputs acquisition results.
            orders = self.marketplace.place_order(cart_id)
            for order in orders:
                print(self.kwargs['name'], 'bought', order)

import logging
from logging.handlers import RotatingFileHandler
from threading import Lock
import unittest
from tema.product import Tea
from tema.product import Coffee

class Marketplace:
    """
    @brief Shared resource manager orchestrating inventory pools, cart registration, and thread synchronization.
    State Management: Maintains mappings for producer capacity (slots), global inventory (queues), and active consumer carts.
    Synchronization: Employs fine-grained locks per producer and per registry (slots_locks, queues_locks) to minimize global contention.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @param queue_size_per_producer Hard limit on inventory per supplier for backpressure control.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producers_slots = {} # Tracks available capacity per producer.
        self.producers_queues = {} # Maps producers to their current stock lists.
        self.carts = {} # Maps session IDs to reserved product lists.
        self.next_producer_id = 0
        self.next_cart_id = 0

        
        self.slots_locks = {}
        self.queues_locks = {}
        self.producer_id_lock = Lock() # Protects global producer enumeration.
        self.cart_id_lock = Lock() # Protects global session enumeration.

        # Observability: Structured transactional auditing.
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        handler = RotatingFileHandler('marketplace.log', maxBytes=20000, backupCount=5)
        self.logger.addHandler(handler)

    def register_producer(self):
        """
        @brief Onboards a new supplier and initializes its thread-safe inventory buffers.
        @return Unique producer identifier (string).
        """

        
        self.producer_id_lock.acquire()
        producer_id = str(self.next_producer_id)
        self.next_producer_id += 1
        self.producer_id_lock.release()

        # Initialization: Scaffolds the stateful tracking for the new producer.
        self.producers_slots[producer_id] = self.queue_size_per_producer
        self.producers_queues[producer_id] = []
        self.slots_locks[producer_id] = Lock()
        self.queues_locks[producer_id] = Lock()

        self.logger.info('Register producer: producer_id = %s', producer_id)
        return producer_id

    def publish(self, producer_id, product):
        """
        @brief Allows a producer to add commodities to the global marketplace pool.
        Constraint: Operation fails if the supplier's individual capacity (slots) is zero.
        """

        
        self.slots_locks[producer_id].acquire()
        if self.producers_slots[producer_id] == 0:
            # Logic: Capacity boundary reached.
            self.logger.info('Publish product: producer_id = %s, product = %s, \
                            return = False', producer_id, product)
            self.slots_locks[producer_id].release()
            return False
        self.slots_locks[producer_id].release()

        # Invariant: Item is added to the producer's specific queue under lock.
        self.queues_locks[producer_id].acquire()
        self.producers_queues[producer_id].append(product)
        self.queues_locks[producer_id].release()

        # State Transition: Decrements available capacity.
        self.slots_locks[producer_id].acquire()
        self.producers_slots[producer_id] -= 1
        self.slots_locks[producer_id].release()

        self.logger.info('Publish product: producer_id = %s, product = %s, \
                        return = True ', producer_id, product)
        return True

    def new_cart(self):
        """
        @brief Allocates a new transactional session for a consumer.
        @return Unique session identifier (integer).
        """

        
        self.cart_id_lock.acquire()
        cart_id = self.next_cart_id
        self.next_cart_id = self.next_cart_id + 1
        self.cart_id_lock.release()

        self.carts[cart_id] = {} # Logic: Map within map to track source producer for each unit.

        self.logger.info('New cart: cart_id = %d', cart_id)
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Atomically moves a commodity from any available producer pool to a consumer cart.
        Logic: Performs an exhaustive sweep across all producer queues. First-available fulfillment strategy.
        @return Boolean indicating if the acquisition was successful.
        """

        
        for producer in self.producers_queues:
            
            # Block Logic: Searches for requested stock in the current producer pool.
            self.queues_locks[producer].acquire()
            if product in self.producers_queues[producer]:
                # Invariant: Product must be removed from global inventory before being assigned to a session.
                self.producers_queues[producer].remove(product)
                self.queues_locks[producer].release()

                
                if not producer in self.carts[cart_id].keys():
                    self.carts[cart_id][producer] = []
                self.carts[cart_id][producer].append(product)

                self.logger.info('Add to cart: cart_id = %d, product = %s, \
                                return = True', cart_id, product)
                return True
            self.queues_locks[producer].release()

        
        self.logger.info('Add to cart: cart_id = %d, product = %s, \
                        return = False', cart_id, product)
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Reverts an acquisition, restoring the unit to its originating producer's stock.
        """
        
        for producer in self.carts[cart_id]:
            if product in self.carts[cart_id][producer]:
                
                self.carts[cart_id][producer].remove(product)
                
                # Logic: Uses the session-bound producer reference to route the return correctly.
                self.queues_locks[producer].acquire()
                self.producers_queues[producer].append(product)
                self.queues_locks[producer].release()

                self.logger.info('Remove from cart: cart_id = %d, product = %s', cart_id, product)
                break

    def place_order(self, cart_id):
        """
        @brief Finalizes the transaction and flushes results.
        Side Effect: Reclaims producer capacity (slots) for all fulfilled items.
        """
        
        returned_list = []

        
        for producer in self.carts[cart_id]:
            for product in self.carts[cart_id][producer]:
                returned_list.append(product)
                
                # State Sync: Signal to producers that a slot has been vacated.
                self.slots_locks[producer].acquire()
                self.producers_slots[producer] += 1
                self.slots_locks[producer].release()

        # Finalization: Deletes the session state.
        del self.carts[cart_id]

        self.logger.info('Place order: cart_id = %d,returned list = %s', cart_id, list)
        return returned_list


class TestMarketplace(unittest.TestCase):
    """
    @brief Unit test suite for verifying Marketplace transactional logic and concurrency integrity.
    """

    def setUp(self):
        self.marketplace = Marketplace(3)
        self.products = []
        self.products.append(Tea('Musetel', 1, 'Herbal'))
        self.products.append(Tea('Coada Soricelului', 3, 'Herbal'))
        self.products.append(Coffee('Espresso', 2, '10.0', 'HIGH'))
        self.products.append(Tea('Urechea boului', 4, 'Non-herbal'))

    def test_register_producer(self):
        
        for i in range(0, 10):
            self.assertEqual(self.marketplace.register_producer(), str(i))

    def test_new_cart(self):
        
        for i in range(0, 10):
            self.assertEqual(self.marketplace.new_cart(), i)

    def test_publish(self):
        
        for _ in range(0, 4):
            self.marketplace.register_producer()

        self.marketplace.publish('0', self.products[0])
        self.marketplace.publish('0', self.products[1])
        self.marketplace.publish('2', self.products[0])
        self.marketplace.publish('0', self.products[0])
        self.marketplace.publish('1', self.products[3])

        self.assertEqual(self.marketplace.producers_queues['0'],
                         [self.products[0], self.products[1], self.products[0]])
        self.assertEqual(self.marketplace.producers_queues['1'], [self.products[3]])
        self.assertEqual(self.marketplace.producers_queues['2'], [self.products[0]])
        self.assertEqual(self.marketplace.producers_queues['3'], [])

    def test_publish_fails(self):
        
        self.marketplace.register_producer()
        for i in range(0, 3):
            self.assertTrue(self.marketplace.publish(str(0), self.products[i]))
        self.assertFalse(self.marketplace.publish(str(0), self.products[0]))

    def test_add_to_cart(self):
        

        
        for i in range(0, 3):
            self.marketplace.new_cart()
            self.marketplace.register_producer()
        for i in range(0, 3):
            for _ in range(0, 3):
                self.marketplace.publish(str(i), self.products[i])

        
        for i in range(0, 3):
            for _ in range(0, 3):
                self.assertTrue(self.marketplace.add_to_cart(i, self.products[i]))

        
        for i in range(0, 3):
            self.assertFalse(self.marketplace.add_to_cart(i, self.products[i]))
        
        for i in range(0, 3):
            self.assertEqual(self.marketplace.carts[i][str(i)],
                             [self.products[i], self.products[i], self.products[i]])

    def test_remove_from_cart(self):
        
        id_cart = self.marketplace.new_cart()
        id_producer = self.marketplace.register_producer()
        for i in range(0, 3):
            self.marketplace.publish('0', self.products[i])
            self.marketplace.add_to_cart(0, self.products[i])

        self.marketplace.remove_from_cart(id_cart, self.products[1])
        self.assertEqual(self.marketplace.carts[id_cart][id_producer],
                         [self.products[0], self.products[2]])
        self.marketplace.remove_from_cart(id_cart, self.products[0])
        self.assertEqual(self.marketplace.carts[id_cart][id_producer], [self.products[2]])
        self.marketplace.remove_from_cart(id_cart, self.products[2])
        self.assertEqual(self.marketplace.carts[id_cart][id_producer], [])

    def test_place_order(self):
        

        
        cart_id = self.marketplace.new_cart()
        producer_id = self.marketplace.register_producer()
        for i in range(0, 3):
            self.marketplace.publish(producer_id, self.products[i])
            self.marketplace.add_to_cart(cart_id, self.products[i])

        returned_list = self.marketplace.place_order(cart_id)
        
        self.assertEqual(returned_list, [self.products[0], self.products[1], self.products[2]])
        
        self.assertEqual(self.marketplace.producers_slots[producer_id], 3)


import time
from threading import Thread



class Producer(Thread):
    """
    @brief Producer agent responsible for industrial resource generation and stock replenishment.
    Logic: Continually fulfills production quotas and publishes results to the Marketplace.
    Functional Utility: Models industrial processing latencies and handles supply-side flow control.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @param products List of (ProductID, Quantity, ManufactureTime) production metrics.
        @param marketplace Shared resource management interface.
        @param republish_wait_time Duration to wait when the marketplace is saturated.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        @brief Main manufacturing loop for the producer thread.
        """

        # Initialization: Registers once as a persistent supplier.
        producer_id = self.marketplace.register_producer()
        
        while True:
            for product in self.products:
                for _ in range(product[1]):
                    # Logic: Simulated duration required to create the commodity unit.
                    time.sleep(product[2])
                    
                    # Block Logic: Resource publication with backpressure handling.
                    success = self.marketplace.publish(producer_id, product[0])
                    while not success:

                        # Synchronization: Yield execution when the marketplace queue is full.
                        time.sleep(self.republish_wait_time)
                        success = self.marketplace.publish(producer_id, product[0])


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Immutable base representation of a marketable commodity unit.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Specialized beverage commodity.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Specialized beverage commodity with profile metrics.
    """
    acidity: str
    roast_level: str
