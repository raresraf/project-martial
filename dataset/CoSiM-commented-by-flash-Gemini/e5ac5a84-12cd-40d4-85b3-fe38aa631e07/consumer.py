"""
@e5ac5a84-12cd-40d4-85b3-fe38aa631e07/consumer.py
@brief Distributed marketplace simulation with specialized locking and persistent audit logging.
* Algorithm: Concurrent producer-consumer model with shared state managed via producer-affinity maps and a set of granular mutual exclusion locks.
* Functional Utility: Facilitates a virtual market where producers generate goods and consumers manage items via cart sessions, ensuring thread-safe inventory transitions and detailed execution tracking.
"""

from threading import Thread
import time


class Consumer(Thread):
    """
    @brief Consumer entity that performs synchronized shopping operations across multiple carts.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes the consumer with its assigned shopping lists and market connection.
        """
        Thread.__init__(self, **kwargs)
        self.retry_wait_time = retry_wait_time
        self.carts = carts
        self.marketplace = marketplace

    def run(self):
        """
        @brief Main execution loop for shopping activities.
        Algorithm: Iterative processing of carts with a busy-wait retry strategy for inventory acquisition.
        """
        for cart in self.carts:
            # Logic: Initializes a new transaction identifier.
            cart_id = self.marketplace.new_cart()
            operations_number = 0

            for operation in cart:
                # Block Logic: Processes requested quantity for each product entry.
                while operations_number < operation["quantity"]:
                    if operation["type"] == "add":
                        # Logic: Continuous attempt to secure product from inventory.
                        add_to_cart = self.marketplace.add_to_cart(cart_id, operation["product"])
                        if not add_to_cart:
                            # Functional Utility: Throttles retry attempts during stock-outs.
                            time.sleep(self.retry_wait_time)
                        else:
                            operations_number = operations_number + 1
                    else:
                        # Logic: Returns items from cart to inventory.
                        self.marketplace.remove_from_cart(cart_id, operation["product"])
                        operations_number = operations_number + 1
                operations_number = 0

            # Post-condition: Completes the transaction.
            self.marketplace.place_order(cart_id)


from threading import Lock, currentThread
import logging
from logging.handlers import RotatingFileHandler


class Marketplace:
    """
    @brief Centralized inventory controller with multi-lock synchronization and integrated audit logging.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the marketplace and its granular synchronization primitives.
        """
        self.producers_ids = []
        self.producers_sizes = [] # Intent: Tracks current stock count per producer.
        self.carts_number = 0
        self.carts = []
        
        # Block Logic: Specialized Locks.
        self.print_lock = Lock()      # Intent: Serializes console output for purchase confirmation.
        self.num_carts_lock = Lock()  # Intent: Serializes cart ID generation.
        self.register_lock = Lock()   # Intent: Serializes producer onboarding.
        self.sizes_lock = Lock()      # Intent: Serializes inventory state transitions (add/remove).
        
        self.max_elements_for_producer = queue_size_per_producer
        self.product_to_producer = {} # Intent: Reverse-lookup map from product to producer ID.
        self.products = []            # Intent: Global list of available items.
        
        # Block Logic: Audit Logging Configuration.
        self.logger = logging.getLogger('marketplace')
        self.logger.setLevel(logging.INFO)
        log_form = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        rotating_file_handler = RotatingFileHandler('marketplace.log', 'a', 16384)
        rotating_file_handler.setFormatter(log_form)
        self.logger.addHandler(rotating_file_handler)

    def register_producer(self):
        """
        @brief Onboards a new producer and initializes its inventory tracking.
        """
        with self.register_lock:
            prod_id = len(self.producers_ids)
            self.producers_ids.append(prod_id)
            self.producers_sizes.append(0)

        self.logger.info("prod_id = %s", str(prod_id))
        return prod_id

    def publish(self, producer_id, product):
        """
        @brief Adds a product to the global pool if the producer's quota allows.
        Algorithm: Search for producer index and atomic increment of its current stock count.
        """
        self.logger.info("producer_id = %s product = %s", str(producer_id), str(product))
        prod_id = int(producer_id)

        # Logic: Validates capacity constraints across all registered producers.
        for i in range(0, len(self.producers_ids)):
            if self.producers_ids[i] == prod_id:
                if self.producers_sizes[i] >= self.max_elements_for_producer:
                    return False
                self.producers_sizes[i] = self.producers_sizes[i] + 1

        self.products.append(product)
        self.product_to_producer[product] = prod_id
        self.logger.info("return_value = %s", "True")
        return True

    def new_cart(self):
        """
        @brief Allocates a new transaction identifier and session buffer.
        """
        with self.num_carts_lock:
            self.carts_number = self.carts_number + 1
            cart_id = self.carts_number

        self.carts.append({"id": cart_id, "list": []})
        self.logger.info("cart_id = %s", str(cart_id))
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Transfers a product unit from the general pool to a specific consumer cart.
        Invariant: Uses sizes_lock to ensure atomic inventory ownership transfer.
        """
        self.logger.info("cart_id = %s product = %s", str(cart_id), str(product))
        with self.sizes_lock:
            if product in self.products:
                # Logic: Identifies source producer to restore its available quota.
                prod_id = self.product_to_producer[product]
                for i in range(0, len(self.producers_ids)):
                    if self.producers_ids[i] == prod_id:
                        self.producers_sizes[i] = self.producers_sizes[i] - 1
                
                self.products.remove(product)
                # Logic: Locates target cart buffer and appends item.
                cart = [x for x in self.carts if x["id"] == cart_id][0]
                cart["list"].append(product)
                return True
        
        self.logger.info("return_value = %s", "False")
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Returns an item from a cart back to its original producer's inventory.
        """
        self.logger.info("cart_id = %s product = %s", str(cart_id), str(product))
        cart = [x for x in self.carts if x["id"] == cart_id][0]
        cart["list"].remove(product)
        self.products.append(product)

        with self.sizes_lock:
            prod_id = self.product_to_producer[product]
            for i in range(0, len(self.producers_ids)):
                if self.producers_ids[i] == prod_id:
                    # Logic: Re-occupies producer quota upon return.
                    self.producers_sizes[i] = self.producers_sizes[i] + 1

    def place_order(self, cart_id):
        """
        @brief Finalizes the transaction and commits the purchase log.
        """
        self.logger.info("cart_id = %s", str(cart_id))
        cart = [x for x in self.carts if x["id"] == cart_id][0]
        self.carts.remove(cart)
        
        for product in cart["list"]:
            # Invariant: Uses print_lock to synchronize access to standard output streams.
            with self.print_lock:
                print("{} bought {}".format(currentThread().getName(), product))
        
        self.logger.info("cart_items = %s", str(cart["list"]))
        return cart["list"]


from threading import Thread
import time


class Producer(Thread):
    """
    @brief Producer agent that generates goods for the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes the producer with its manufacturing schedule.
        """
        Thread.__init__(self, **kwargs)
        self.republish_wait_time = republish_wait_time
        self.products = products
        self.marketplace = marketplace
        self.prod_id = self.marketplace.register_producer()

    def run(self):
        """
        @brief Main production lifecycle.
        Algorithm: Iterative batch generation with fixed production latency and quota-constrained publication.
        """
        while True:
            for (product, number_products, time_sleep) in self.products:
                for i in range(number_products):
                    # Logic: Attempts to publish; retries on buffer saturation.
                    if self.marketplace.publish(str(self.prod_id), product):
                        # Domain: Manufacturing simulation delay.
                        time.sleep(time_sleep)
                    else:
                        # Functional Utility: Throttles attempts when market capacity is reached.
                        time.sleep(self.republish_wait_time)
                        # Note: Ineffective loop decrement attempt present in original logic.


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Immutable base schema for marketplace products.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Specialized product type for tea items.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Specialized product type for coffee items.
    """
    acidity: str
    roast_level: str
