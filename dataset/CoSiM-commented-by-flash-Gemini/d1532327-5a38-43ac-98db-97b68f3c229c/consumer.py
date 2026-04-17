"""
@d1532327-5a38-43ac-98db-97b68f3c229c/consumer.py
@brief multi-threaded electronic marketplace with centralized availability pooling.
This module implements a coordinated trading environment where Producers supply goods 
to a global availability pool. Consumers interact with this pool to perform 
transactional operations (add/remove items) asynchronously. The system utilizes 
dual-locking to protect inventory state and session data, while maintaining a 
detailed execution log for auditing purposes.

Domain: Concurrent Systems, Inventory Pooling, Execution Logging.
"""

from threading import Thread
import time


class Consumer(Thread):
    """
    Consumer entity simulating a shopper.
    Functional Utility: Manages a sequence of shopping carts and performs 
    automated transactions using a polling-based retry mechanism.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the consumer thread.
        @param carts: Nested list of shopping operation batches.
        @param marketplace: Central trading coordinator.
        @param retry_wait_time: delay between failed acquisition attempts.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.cart_id = -1

    def run(self):
        """
        Main execution loop for shopper actions.
        Logic: Iteratively processes each cart, performing adds and removes 
        sequentially until the entire batch is fulfilled.
        """
        for c in self.carts:
            # Atomic creation of a new transaction context.
            self.cart_id = self.marketplace.new_cart()
            for op in c:
                op_type = op['type']
                if op_type == "add":
                    i = 0
                    # Block Logic: Fulfillment loop.
                    while i < op['quantity']:
                        ret = self.marketplace.add_to_cart(self.cart_id, op['product'])
                        if ret is True:
                            i = i + 1
                        else:
                            # Backoff: wait for marketplace supply to refresh.
                            time.sleep(self.retry_wait_time)
                elif op_type == "remove":
                    i = 0
                    while i < op['quantity']:
                        ret = self.marketplace.remove_from_cart(self.cart_id, op['product'])
                        if ret is True:
                            i = i + 1
                        else:
                            time.sleep(self.retry_wait_time)
            
            # Commit: finalize the session and print the manifest.
            my_cart = self.marketplace.place_order(self.cart_id)
            for p in my_cart:
                print(self.name + ' bought ' + str(p))


from threading import Lock
from logging import getLogger
import logging.handlers


class Marketplace:
    """
    Centralized coordinator for producers and consumers.
    Functional Utility: Manages a global availability pool (available_products_pairs) 
    and employs dual-locking to isolate inventory updates from session management.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.
        @param queue_size_per_producer: Capacity limit per supply line.
        """
        self.queue_size_per_producer = queue_size_per_producer

        self.producersList = []
        self.cartsList = []
        # Efficiency Logic: Central pool of all items currently for sale.
        self.available_products_pairs = []

        # Granular Locking: Protects consumer sessions and the global pool.
        self.add_remove_lock = Lock()
        # Granular Locking: Protects producer-specific buffer state.
        self.producer_lock = Lock()

        # Audit System: configured for rotating file-based logging.
        self.info_logger = getLogger(__name__)
        self.info_logger.setLevel(logging.INFO)
        self.info_logger.addHandler(logging.handlers.RotatingFileHandler("marketplace.log"))

    def register_producer(self):
        """Allocates a new supply line and returns a unique producer_id."""
        self.info_logger.info("Function register_producer with parameters: %s", str(self))
        new_producer = []
        self.producersList.append(new_producer)
        self.info_logger.info("Function register_producer returns value: "
                              + str(len(self.producersList) - 1))
        return len(self.producersList) - 1


    def publish(self, producer_id, product):
        """
        Accepts a product into the supply line and the global pool.
        Logic: Enforces per-producer capacity limits before atomic insertion.
        @return: True if accepted, False if supply line is full.
        """
        self.info_logger.info("Function publish with parameters: "
                              + str(self)
                              + str(producer_id)
                              + str(product))

        self.producer_lock.acquire()
        if len(self.producersList[producer_id]) == self.queue_size_per_producer:
            self.producer_lock.release()
            self.info_logger.info("Function publish returns value: False")
            return False

        # Double Insertion: update producer buffer and global availability pool.
        self.producersList[producer_id].append(product)
        self.available_products_pairs.append((product, producer_id))
        self.producer_lock.release()

        self.info_logger.info("Function publish returns value: True")
        return True


    def new_cart(self):
        """Creates a new unique consumer session."""
        self.info_logger.info("Function new_cart with parameters:" + str(self))
        self.add_remove_lock.acquire()
        new_c = []
        self.cartsList.append(new_c)
        self.info_logger.info("Function new_cart returns value: "
                              + str(len(self.cartsList) - 1))
        self.add_remove_lock.release()
        return len(self.cartsList) - 1


    def add_to_cart(self, cart_id, product):
        """
        Transfers an item from the global pool to a specific session cart.
        Logic: performs a linear search in the availability pool.
        """
        self.info_logger.info("Function add_to_cart with parameters:"
                              + str(cart_id)
                              + str(product))
        self.add_remove_lock.acquire()
        for pair in self.available_products_pairs:
            if pair[0] == product:
                # Transition: move from global pool to session storage.
                self.cartsList[cart_id].append(pair)
                self.available_products_pairs.remove(pair)
                self.add_remove_lock.release()
                self.info_logger.info("Function add_to_cart returns value: True")
                return True
        self.add_remove_lock.release()
        self.info_logger.info("Function add_to_cart returns value: False")
        return False


    def remove_from_cart(self, cart_id, product):
        """Restores an item from a cart back to the global availability pool."""
        self.info_logger.info("Function remove_from_cart with parameters: "
                              + str(cart_id)
                              + str(product))
        self.add_remove_lock.acquire()
        for pair in self.cartsList[cart_id]:
            if pair[0] == product:
                # Reversal: restore to global pool.
                self.available_products_pairs.append(pair)
                self.cartsList[cart_id].remove(pair)
                self.info_logger.info("Function remove_from_cart returns value: True")
                self.add_remove_lock.release()
                return True
        self.info_logger.info("Function remove_from_cart returns value: False")
        self.add_remove_lock.release()
        return False


    def place_order(self, cart_id):
        """
        Finalizes the transaction session.
        Logic: Cleans up producer buffers by removing purchased items.
        """
        self.info_logger.info("Function place_order with parameters: " + str(cart_id))
        prod_list = []
        self.producer_lock.acquire()
        # Block Logic: Inventory cleanup.
        for pair in self.cartsList[cart_id]:
            prod_list.append(pair[0])
            # Functional Utility: synchronized removal from producer's primary buffer.
            self.producersList[pair[1]].remove(pair[0])
        self.producer_lock.release()
        self.info_logger.info("Function place_order returns value: " + str(prod_list))
        return prod_list


from threading import Thread
import time


class Producer(Thread):
    """
    Simulation thread representing a manufacturing entity.
    Functional Utility: Manages the production cycle and handles backpressure 
    via periodic retries.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = marketplace.register_producer()

    def run(self):
        """
        Main manufacturing loop.
        Algorithm: Iterative production and publication with duration simulation.
        """
        while True:
            for p in self.products:
                i = 0
                while i < p[1]:
                    ret = self.marketplace.publish(self.producer_id, p[0])
                    if ret is True:
                        i = i + 1
                        # Simulate production overhead.
                        time.sleep(float(p[2]))
                    else:
                        # Functional Utility: wait for marketplace demand.
                        time.sleep(float(self.republish_wait_time))
