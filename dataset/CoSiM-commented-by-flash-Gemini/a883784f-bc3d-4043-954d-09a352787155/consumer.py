"""
@file consumer.py
@brief multi-threaded marketplace simulation with persistent logging and transactional safety.
@details Implements a synchronized marketplace environment where producers publish items 
and consumers perform batch purchases. Utilizes fine-grained locking and rotating logs 
for auditability.
"""

from threading import Thread
from time import sleep

class Consumer(Thread):
    """
    @brief Asynchronous agent performing structured shopping operations.
    Functional Utility: Manages a sequence of transaction lists (carts), handling 
    resource contention through exponential backoff retries.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @param carts Hierarchical list of intended transactions.
        @param marketplace Shared state controller.
        @param retry_wait_time Polling interval for unavailable inventory.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        @brief Consumer lifecycle execution.
        Invariant: All requested items are eventually added to the cart or retried 
        indefinitely if stock is missing.
        """
        id_ = self.marketplace.new_cart()
        for cart in self.carts:
            for entry in cart:
                action = entry["type"]
                product = entry["product"]
                quantity = entry["quantity"]
                counter = 0
                
                if action == "add":
                    /**
                     * Block Logic: Fulfillment loop for item acquisition.
                     * Logic: Repeatedly attempts to reserve the product from the marketplace.
                     */
                    while counter < quantity:
                        added = self.marketplace.add_to_cart(id_, product)
                        if added:
                            counter += 1
                        else:
                            # Protocol: Waits for inventory replenishment.
                            sleep(self.retry_wait_time)
                else:
                    # Logic: Returns items from the local cart back to global inventory.
                    while counter < quantity:
                        self.marketplace.remove_from_cart(id_, product)
                        counter += 1
        
        # Finalization: Commits the transaction and prints results.
        new_cart = self.marketplace.place_order(id_)
        for product in new_cart:
            print(self.name, "bought", product, flush=True)


from threading import Lock
import logging
from logging.handlers import RotatingFileHandler

class Marketplace:
    """
    @brief Centralized transaction manager and inventory repository.
    Architecture: Employs a multi-lock strategy to minimize contention across 
    registration, publishing, and purchasing operations.
    """

    # Shared Logger: Configured with rotation to manage disk footprint.
    logger = logging.getLogger('marketplace.log')
    logger.setLevel(logging.INFO)
    logger.addHandler(RotatingFileHandler('marketplace.log', maxBytes=2000, backupCount=10))

    def __init__(self, queue_size_per_producer):
        """
        @param queue_size_per_producer Hard limit on items per producer to prevent overflow.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = -1
        self.consumer_id = -1
        # State: Registry of producers and their available inventory.
        self.producers = {}
        # State: Registry of active consumer shopping carts.
        self.carts = {}
        
        # Synchronization Primitives: Specialized locks for different operational domains.
        self.publish_lock = Lock()
        self.register_lock = Lock()
        self.producer_lock = Lock()
        self.cart_lock = Lock()

    def register_producer(self):
        """
        @brief Onboards a new supplier with a unique monotonic identifier.
        """
        self.register_lock.acquire()
        self.logger.info('Start - register_producer')
        self.producer_id += 1
        self.producers[self.producer_id] = []
        self.logger.info('End - register_producer')
        self.register_lock.release()
        return self.producer_id

    def publish(self, producer_id, product):
        """
        @brief Injects a product into a producer's buffer.
        @return True if within capacity limits, False otherwise.
        """
        self.publish_lock.acquire()
        self.logger.info('Start - publish')
        success = False
        # Logic: Enforces buffer bounds. Items stored as [value, availability_flag].
        if len(self.producers.get(producer_id)) < self.queue_size_per_producer:
            self.producers.get(producer_id).append([product, 1])
            success = True
        self.logger.info('End - publish')
        self.publish_lock.release()
        return success

    def new_cart(self):
        """
        @brief Allocates a new transaction context for a consumer.
        """
        self.cart_lock.acquire()
        self.logger.info('Start - new_cart')
        self.consumer_id += 1
        self.carts[self.consumer_id] = []
        self.logger.info('End - new_cart')
        self.cart_lock.release()
        return self.consumer_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Atomically transfers an item from global inventory to a specific cart.
        Search Strategy: Linear sweep across all producers.
        """
        found_flag = False
        self.producer_lock.acquire()
        self.logger.info('Start - add_to_cart, params = %d, %s', cart_id, product)
        
        for producer in list(self.producers):
            for i in range(len(self.producers[producer])):
                # Logic: Checks if item matches name and is currently available (flag == 1).
                if (self.producers[producer][i][0] == product and
                        self.producers[producer][i][1] == 1):
                    # Atomic State Change: Mark as unavailable and link to cart.
                    self.producers[producer][i][1] = 0
                    self.carts[cart_id].append([producer, product, i])
                    found_flag = True
                    break
            if found_flag:
                break

        self.logger.info('End - add_to_cart')
        self.producer_lock.release()
        return found_flag

    def remove_from_cart(self, cart_id, product):
        """
        @brief Reverts an item reservation, restoring global availability.
        """
        self.logger.info('Start - remove_from_cart, params = %d, %s', cart_id, product)
        for entry in self.carts[cart_id]:
            if entry[1] == product:
                # Restoration: Remove from cart and reset availability flag in source buffer.
                self.carts[cart_id].remove(entry)
                self.producers[entry[0]][entry[2]][1] = 1
                break
        self.logger.info('End - remove_from_cart')

    def place_order(self, cart_id):
        """
        @brief Finalizes the transaction and returns the list of acquired items.
        """
        self.logger.info('Start - place_order, params = %d', cart_id)
        return [entry[1] for entry in self.carts[cart_id]]


class Producer(Thread):
    """
    @brief Autonomous manufacturing agent.
    Functional Utility: Manages the supply side lifecycle, including production 
    latency and buffer-overflow retries.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        @brief Main production loop.
        Invariant: Continuously attempts to fulfill quotas for assigned goods.
        """
        while True:
            id_ = self.marketplace.register_producer()
            for product in self.products:
                counter = 0
                while counter < product[1]:
                    # Logic: Publication attempt with backoff.
                    published = self.marketplace.publish(id_, product[0])
                    if published:
                        counter += 1
                        # Production Latency: Simulates time taken to create the item.
                        sleep(product[2])
                    else:
                        sleep(self.republish_wait_time)
