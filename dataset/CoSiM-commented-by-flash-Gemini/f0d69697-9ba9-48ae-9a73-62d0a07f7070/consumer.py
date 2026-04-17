"""
@f0d69697-9ba9-48ae-9a73-62d0a07f7070/consumer.py
@brief multi-threaded electronic marketplace with session pre-allocation and global pooling.
This module implements a coordinated trading environment where Producers supply goods 
to a central availability pool. Consumers operate using a pre-allocation strategy, 
securing all required session identifiers (carts) before beginning the transaction 
phase. The system uses granular locking to protect supply-line capacity and session 
state, ensuring atomic item transfers between the global inventory and consumer contexts.

Domain: Concurrent Systems, Session Pre-allocation, Inventory Pooling.
"""

from threading import Thread
import time


class Consumer(Thread):
    """
    Consumer entity simulating an automated shopper.
    Functional Utility: Manages multiple shopping sessions using a two-phase 
    lifecycle: session pre-allocation followed by iterative task fulfillment.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the consumer thread.
        @param carts: Nested list of shopping operation batches.
        @param marketplace: Central trading hub.
        @param retry_wait_time: delay between failed acquisition attempts.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        Main shopper execution loop.
        Logic: Phase 1: Pre-allocates all session contexts. Phase 2: Iteratively 
        processes each cart, performing adds and removes sequentially.
        """
        num_of_carts = len(self.carts)
        my_carts = []

        # Block Logic: Session Pre-allocation phase.
        for _ in range(0, num_of_carts):
            new_cart_id = self.marketplace.new_cart()
            my_carts.append(new_cart_id)

        # Block Logic: Workload execution phase.
        for current_cart in self.carts:
            # Claim the next pre-allocated session identifier.
            current_cart_id = my_carts.pop(0)

            for current_operation in current_cart:
                desired_quantity = current_operation["quantity"]
                current_quantity = 0

                # Block Logic: Fulfillment loop.
                # Logic: Continues until the requested quantity is successfully acquired/removed.
                while current_quantity < desired_quantity:
                    current_operation_type = current_operation["type"]
                    current_operation_product = current_operation["product"]

                    if current_operation_type == "add":
                        current_operation_status = self.marketplace\
                            .add_to_cart(current_cart_id, current_operation_product)
                    else:
                        current_operation_status = self.marketplace \
                            .remove_from_cart(current_cart_id, current_operation_product)

                    if current_operation_status is True or current_operation_status is None:
                        current_quantity = current_quantity + 1
                    else:
                        # Functional Utility: polling delay during resource contention.
                        time.sleep(self.retry_wait_time)

            # Commit: finalize the session and print purchased inventory.
            bought_products = self.marketplace.place_order(current_cart_id)
            for bought_product in bought_products:
                print(self.kwargs["name"] + " bought " + str(bought_product))

from threading import Lock

class Marketplace:
    """
    Central hub for coordinating transactions and inventory state.
    Functional Utility: Manages producer load and consumer sessions using specialized 
    mutexes to ensure thread-safe transitions in the global product pool.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace hub.
        @param queue_size_per_producer: Capacity limit per supply line.
        """
        # Persistence Mapping: tracks product origins for transaction reversal.
        self.producer_of_product = {}

        # Granular Synchronization: protects supply-line occupancy counts.
        self.queue_size_of_producer = {}
        self.queue_size_of_producer_lock = Lock()

        self.queue_size_per_producer = queue_size_per_producer

        # Granular Synchronization: protects session metadata.
        self.carts = {}
        self.carts_lock = Lock()

        # Global Inventory Pool.
        self.all_products = []

    def register_producer(self):
        """
        Allocates a new unique supply line index.
        Logic: Uses a dedicated mutex to ensure atomic registration.
        """
        self.queue_size_of_producer_lock.acquire()
        current_producers_number = len(self.queue_size_of_producer)
        self.queue_size_of_producer[current_producers_number] = 0
        self.queue_size_of_producer_lock.release()
        return current_producers_number

    def publish(self, producer_id, product):
        """
        Accepts a product from a producer into the global pool.
        Logic: verifies current occupancy against maximum capacity.
        @return: True if accepted, False if full.
        """
        if self.queue_size_of_producer[producer_id] >= self.queue_size_per_producer:
            return False

        self.queue_size_of_producer_lock.acquire()
        # Atomic update of supply state.
        self.queue_size_of_producer[producer_id] = self.queue_size_of_producer[producer_id] + 1
        self.producer_of_product[product] = producer_id
        self.all_products.append(product)
        self.queue_size_of_producer_lock.release()
        return True

    def new_cart(self):
        """Allocates a new shopper session context."""
        self.carts_lock.acquire()
        current_carts_number = len(self.carts)
        self.carts[current_carts_number] = []
        self.carts_lock.release()
        return current_carts_number

    def add_to_cart(self, cart_id, product):
        """
        Transfers an item from the global pool to a specific shopper session.
        Logic: performs an atomic claim on the item and restores producer capacity.
        """
        self.queue_size_of_producer_lock.acquire()

        # Availability Check.
        if product not in self.all_products:
            self.queue_size_of_producer_lock.release()
            return False

        # Transactional Transfer.
        producer_id = self.producer_of_product[product]
        self.queue_size_of_producer[producer_id] = self.queue_size_of_producer[producer_id] - 1
        self.all_products.remove(product)
        self.queue_size_of_producer_lock.release()

        # Update session storage.
        self.carts[cart_id].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        """Restores a product from a cart back to the global availability pool."""
        self.carts[cart_id].remove(product)
        self.all_products.append(product)
        
        # Transaction Reversal: update producer occupancy.
        producer_id = self.producer_of_product[product]
        self.queue_size_of_producer_lock.acquire()
        self.queue_size_of_producer[producer_id] = self.queue_size_of_producer[producer_id] + 1
        self.queue_size_of_producer_lock.release()

    def place_order(self, cart_id):
        """Finalizes the purchase and flushes the session context."""
        bought_products = self.carts[cart_id]
        self.carts[cart_id] = []
        return bought_products


from threading import Thread
import time


class Producer(Thread):
    """
    Simulation thread representing a manufacturing entity.
    Functional Utility: Manages continuous product cycles and handles backpressure 
    via periodic retries.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes the producer and secures a supply line ID.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        Main production cycle.
        Algorithm: Iterative manufacturing with synchronous backoff retries.
        """
        while True:
            for current_product in self.products:
                current_product_type = current_product[0]
                current_product_quantity_desired = current_product[1]
                current_product_quantity = 0
                current_product_time_to_create = current_product[2]

                # Block Logic: Manufacturing loop.
                while current_quantity < current_product_quantity_desired:
                    current_transaction_status = self.marketplace\
                        .publish(self.producer_id, current_product_type)

                    if current_transaction_status is True:
                        # Simulate production overhead.
                        time.sleep(current_product_time_to_create)
                        current_product_quantity = current_product_quantity + 1
                    else:
                        # Functional Utility: Poll backoff when hub is full.
                        time.sleep(self.republish_wait_time)
