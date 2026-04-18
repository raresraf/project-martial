"""
@77cbff7b-95bd-4d87-a74e-8f75e88b6008/consumer.py
@brief Concurrent e-commerce simulation demonstrating Producer-Consumer synchronization patterns.
Architecture: Multi-threaded model where Producers generate inventory and Consumers process purchase requests via a central Marketplace.
Functional Utility: Handles inventory management, transactional cart operations, and session-based order fulfillment.
Synchronization: Employs threading.Lock for mutual exclusion over shared data structures (carts, product lists).
"""

import sys
import logging
import time
from threading import Thread, Lock, currentThread
from logging.handlers import RotatingFileHandler


class Consumer(Thread):
    """
    @brief Consumer agent responsible for executing shopping strategies in a concurrent environment.
    Logic: Orchestrates cart fulfillment by iteratively attempting acquisition of requested commodities.
    Error Handling: Implements a polling retry mechanism with configurable sleep intervals for out-of-stock scenarios.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @param carts List of shopping lists to be processed sequentially.
        @param marketplace Shared resource mediator.
        @param retry_wait_time Temporal duration to yield when the marketplace is depleted.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts 
        self.marketplace = marketplace 
        self.retry_wait_time = retry_wait_time 
        # Functional Utility: Maps semantic operation types to executable marketplace methods.
        self.operations = {"add": marketplace.add_to_cart,
                           "remove": marketplace.remove_from_cart}

    def run(self):
        """
        @brief Main execution loop for the consumer thread.
        Logic: Iterates through assigned carts, ensuring each requested item is acquired before finalizing the transaction.
        """
        for cart in self.carts:
            # Logic: Initializes a new isolated session within the marketplace.
            cart_id = self.marketplace.new_cart() 

            for operation in cart:
                quantity = operation["quantity"] 

                # Block Logic: Fulfillment loop.
                # Invariant: Must continue retrying until the requested quantity is successfully reserved in the cart.
                while quantity > 0:
                    operation_type = operation["type"]
                    product = operation["product"]

                    # Logic: Executes the mapped marketplace command (add/remove).
                    if self.operations[operation_type](cart_id, product) is not False:
                        quantity -= 1 
                    else:
                        # Synchronization: Polling yield to allow producers time to replenish inventory.
                        time.sleep(self.retry_wait_time) 

            # Finalization: Transitions the cart from active to completed.
            self.marketplace.place_order(cart_id) 


class Marketplace:
    """
    @brief Centralized resource manager for inventory control and transactional synchronization.
    State Management: Tracks producer capacity, global product availability, and active consumer carts.
    Synchronization: Uses fine-grained locks (carts_lock, producers_lock) to minimize thread contention.
    """

    def __init__(self, queue_size_per_producer):
        """
        @param queue_size_per_producer Maximum backlog allowed per producer for flow control.
        """
        self.carts_lock = Lock() # Protects global cart registry.
        self.carts = [] 

        self.producers_lock = Lock() # Protects global inventory and producer metrics.
        self.producers_capacity = queue_size_per_producer 
        self.producers_sizes = [] 
        self.products = [] # Global pool of (Product, ProducerID) available for purchase.

        # Block Logic: Configures transactional auditing.
        # Functional Utility: Persistent log with rotation to prevent storage exhaustion.
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s : %(message)s')
        formatter.converter = time.gmtime 

        file_handler = RotatingFileHandler(
            "marketplace.log", maxBytes=4096, backupCount=0) 
        file_handler.setFormatter(formatter) 

        logger = logging.getLogger("marketplace") 
        logger.setLevel(logging.INFO) 
        logger.addHandler(file_handler) 
        self.logger = logger

    def register_producer(self):
        """
        @brief Onboards a new producer and initializes its performance metrics.
        @return Unique producer identifier.
        """
        self.logger.info("enter register_producer()")

        with self.producers_lock: 
            self.producers_sizes.append(0) 
            self.logger.info("leave register_producer")
            return len(self.producers_sizes) - 1 

    def publish(self, producer_id, product):
        """
        @brief Allows a producer to add items to the marketplace pool.
        Constraint: Operation is rejected if the producer has reached their individual queue capacity.
        """
        self.logger.info(
            "enter publish(%d, %s)", producer_id, str(product))

        with self.producers_lock: 
            # Block Logic: Backpressure check.
            if self.producers_sizes[producer_id] == self.producers_capacity:
                self.logger.info("leave publish")
                return False 

            self.producers_sizes[producer_id] += 1 
            self.products.append((product, producer_id)) 
            self.logger.info("leave publish")
            return True

    def new_cart(self):
        """
        @brief Allocates a new shopping cart for an authenticated consumer.
        @return Unique cart identifier.
        """
        self.logger.info("enter new_cart()")
        with self.carts_lock: 
            self.carts.append([]) 
            self.logger.info("leave new_cart")
            return len(self.carts) - 1 

    def add_to_cart(self, cart_id, product):
        """
        @brief Atomically moves a product from global inventory to a specific consumer cart.
        Logic: Scans the marketplace global list for the requested item. First-available strategy.
        """
        self.logger.info(
            "enter add_to_cart(%d, %s)", cart_id, str(product))

        self.producers_lock.acquire()
        # Block Logic: Search and reserve item.
        for (prod, prod_id) in self.products:
            if prod == product:
                # Invariant: Item must be removed from global inventory to prevent double-selling.
                self.producers_sizes[prod_id] -= 1
                self.products.remove((prod, prod_id))
                self.producers_lock.release()
                self.carts[cart_id].append((prod, prod_id))
                self.logger.info("leave add_to_cart")
                return True

        self.producers_lock.release()
        self.logger.info("leave add_to_cart")
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Returns a reserved product to the marketplace inventory.
        Logic: Restores the product to its original producer's pool.
        """
        self.logger.info("enter remove_from_cart(%d, %s)", cart_id, str(product))

        for (prod, prod_id) in self.carts[cart_id]:
            if prod == product:
                self.carts[cart_id].remove((prod, prod_id))
                self.producers_lock.acquire()
                self.products.append((prod, prod_id))
                self.producers_sizes[prod_id] += 1
                self.producers_lock.release()
                self.logger.info("leave remove_from_cart")
                return

    def place_order(self, cart_id):
        """
        @brief Finalizes the transaction and serializes the results.
        Side Effect: Flushes the cart and outputs the shopping list to stdout.
        """
        self.logger.info("enter place_order(%d)", cart_id)

        order = ""
        for (product, _) in self.carts[cart_id]:
            order += "{} bought {}
".format(
                currentThread().getName(), product)
        sys.stdout.write(order)
        self.logger.info("leave place_order")
        return self.carts[cart_id]


class Producer(Thread):
    """
    @brief Producer agent responsible for supply-side inventory management.
    Logic: Continuously generates products based on assigned quotas and production times.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @param products List of (ProductID, Quantity, WaitTime) production targets.
        @param marketplace Shared resource mediator.
        @param republish_wait_time Duration to wait when the marketplace is full.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

        # Initialization: Registers with the marketplace to obtain a persistent ID.
        self.producer_id = marketplace.register_producer()

    def run(self):
        """
        @brief Main production loop.
        Logic: Iterates through production targets, publishing to the marketplace with simulated processing delays.
        """
        while True:
            for (product, quantity, wait_time) in self.products:
                while quantity > 0:
                    # Block Logic: Resource publication with backpressure handling.
                    if self.marketplace.publish(self.producer_id, product):
                        quantity -= 1
                        time.sleep(wait_time / 20)
                    else:
                        # Synchronization: Exponentially scaled wait during marketplace saturation.
                        time.sleep(self.republish_wait_time / 20)
