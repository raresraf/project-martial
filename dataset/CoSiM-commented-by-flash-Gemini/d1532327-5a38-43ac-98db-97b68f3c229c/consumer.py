"""
@d1532327-5a38-43ac-98db-97b68f3c229c/consumer.py
@brief Distributed marketplace simulation with persistent audit logging and producer-affinity tracking.
* Algorithm: Resource state management using paired product-producer tuples with fine-grained mutual exclusion for inventory transitions.
* Functional Utility: Orchestrates a multi-threaded virtual market where inventory is tracked via a global registry, ensuring consistent ownership transfer between producers and consumers.
"""

from threading import Thread
import time


class Consumer(Thread):
    """
    @brief Consumer entity that performs synchronized shopping transactions across multiple lists.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes the consumer and prepares its transaction context.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.cart_id = -1

    def run(self):
        """
        @brief Main execution loop for shopping activities.
        Algorithm: Iterative operation processing with busy-wait retry and sleep-based backoff for stock contention.
        """
        for c in self.carts:
            # Logic: Initializes a new transaction identifier.
            self.cart_id = self.marketplace.new_cart()
            for op in c:
                op_type = op['type']
                if op_type == "add":
                    i = 0
                    while i < op['quantity']:
                        ret = self.marketplace.add_to_cart(self.cart_id, op['product'])
                        if ret is True:
                            i = i + 1
                        else:
                            # Functional Utility: Throttles retry attempts during stock-outs.
                            time.sleep(self.retry_wait_time)
                elif op_type == "remove":
                    i = 0
                    while i < op['quantity']:
                        ret = self.marketplace.remove_from_cart(self.cart_id, op['product'])
                        if ret is True:
                            i = i + 1
                        else:
                            time.sleep(self.retry_wait_time)
            
            # Post-condition: Commits the order and displays successful acquisitions.
            my_cart = self.marketplace.place_order(self.cart_id)
            for p in my_cart:
                print(self.name + ' bought ' + str(p))


from threading import Lock
from logging import getLogger
import logging.handlers


class Marketplace:
    """
    @brief Centralized hub for inventory tracking and transaction orchestration with automated logging.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the marketplace and its dedicated audit logger.
        """
        self.queue_size_per_producer = queue_size_per_producer

        self.producersList = [] # Intent: Stores historical product lists for each producer.
        self.cartsList = []     # Intent: Stores active product-producer pairs for each consumer.
        self.available_products_pairs = [] # Intent: Global pool of products available for acquisition.

        self.add_remove_lock = Lock() # Intent: Serializes consumer-level inventory transitions.
        self.producer_lock = Lock()   # Intent: Serializes producer-level stock modifications.

        # Block Logic: Audit Logging Configuration.
        self.info_logger = getLogger(__name__)
        self.info_logger.setLevel(logging.INFO)
        self.info_logger.addHandler(logging.handlers.RotatingFileHandler("marketplace.log"))

    def register_producer(self):
        """
        @brief Onboards a new producer and returns its unique sequence identifier.
        """
        self.info_logger.info("Function register_producer with parameters: %s", str(self))

        new_producer = []
        self.producersList.append(new_producer)

        self.info_logger.info("Function register_producer returns value: "
                              + str(len(self.producersList) - 1))
        return len(self.producersList) - 1


    def publish(self, producer_id, product):
        """
        @brief Adds a product to the global available pool if producer quota permits.
        Algorithm: Concurrent append to both producer-private and global registry.
        """
        self.info_logger.info("Function publish with parameters: "
                              + str(self)
                              + str(producer_id)
                              + str(product))

        self.producer_lock.acquire()

        # Logic: Enforces production capacity constraints.
        if len(self.producersList[producer_id]) == self.queue_size_per_producer:
            self.producer_lock.release()
            self.info_logger.info("Function publish returns value: False")
            return False

        self.producersList[producer_id].append(product)
        # Logic: Maintains producer-affinity to support consistent returns/commits.
        self.available_products_pairs.append((product, producer_id))

        self.producer_lock.release()
        self.info_logger.info("Function publish returns value: True")
        return True


    def new_cart(self):
        """
        @brief Allocates a new transaction identifier for a consumer.
        """
        self.info_logger.info("Function new_cart with parameters:"
                              + str(self))

        self.add_remove_lock.acquire()

        new_c = []
        self.cartsList.append(new_c)

        self.info_logger.info("Function new_cart returns value: "
                              + str(len(self.cartsList) - 1))

        self.add_remove_lock.release()
        return len(self.cartsList) - 1


    def add_to_cart(self, cart_id, product):
        """
        @brief Transfers a product from the global pool to a specific consumer cart.
        Invariant: Uses add_remove_lock to ensure atomic ownership transition.
        """
        self.info_logger.info("Function add_to_cart with parameters:"
                              + str(cart_id)
                              + str(product))

        self.add_remove_lock.acquire()

        # Logic: Linear scan for product availability in the global pool.
        for pair in self.available_products_pairs:
            if pair[0] == product:
                self.cartsList[cart_id].append(pair)
                self.available_products_pairs.remove(pair)
                self.add_remove_lock.release()
                self.info_logger.info("Function add_to_cart returns value: True")
                return True

        self.add_remove_lock.release()
        self.info_logger.info("Function add_to_cart returns value: False")
        return False


    def remove_from_cart(self, cart_id, product):
        """
        @brief Returns a product from a cart back to the global available pool.
        """
        self.info_logger.info("Function remove_from_cart with parameters: "
                              + str(cart_id)
                              + str(product))

        self.add_remove_lock.acquire()

        for pair in self.cartsList[cart_id]:
            if pair[0] == product:
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
        @brief Finalizes the transaction and commits removals from producer catalogs.
        Pre-condition: All products in the cart must have their original producer affinity preserved.
        """
        self.info_logger.info("Function place_order with parameters: "
                              + str(cart_id))

        prod_list = []
        self.producer_lock.acquire()

        # Logic: Permanently removes purchased items from producer tracking.
        for pair in self.cartsList[cart_id]:
            prod_list.append(pair[0])
            self.producersList[pair[1]].remove(pair[0])

        self.producer_lock.release()
        self.info_logger.info("Function place_order returns value: "
                              + str(prod_list))
        return prod_list


from threading import Thread
import time


class Producer(Thread):
    """
    @brief Producer agent that generates goods for the marketplace based on fixed quotas.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes the producer and registers it with the marketplace.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        @brief Main production lifecycle.
        Algorithm: Iterative batch generation with individual item publication and delay.
        """
        while True:
            for p in self.products:
                i = 0
                while i < p[1]:
                    # Logic: Blocks until the item is accepted by the marketplace.
                    ret = self.marketplace.publish(self.producer_id, p[0])
                    if ret is True:
                        i = i + 1
                        # Domain: Simulates production time.
                        time.sleep(float(p[2]))
                    else:
                        # Functional Utility: Throttles attempts when producer queue is full.
                        time.sleep(float(self.republish_wait_time))
