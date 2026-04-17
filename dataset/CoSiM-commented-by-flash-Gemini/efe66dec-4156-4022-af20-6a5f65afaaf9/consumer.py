"""
@efe66dec-4156-4022-af20-6a5f65afaaf9/consumer.py
@brief multi-threaded electronic marketplace with granular synchronization.
This module implements a coordinated trading hub where Producers supply goods 
and Consumers execute transactions. The system utilizes multiple specialized 
mutex locks to isolate different state categories (registration, session management, 
and item transfers), ensuring atomic updates while minimizing global contention.

Domain: Concurrent Systems, Granular Locking, Producer-Consumer Simulation.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    Consumer entity representing an automated shopper.
    Functional Utility: Manages multiple shopping sessions (carts) and performs 
    automated transactions using a polling-based retry strategy.
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
        self.consumer_name = kwargs["name"]

    def run(self):
        """
        Main execution loop for shopper actions.
        Logic: Orchestrates session creation and sequential fulfillment of 
        'add' and 'remove' tasks for each assigned cart.
        """
        for cart in self.carts:
            # Atomic creation of a new transaction context.
            cart_id = self.marketplace.new_cart()

            for event in cart:
                count = 0
                # Block Logic: workload fulfillment loop.
                while count < event["quantity"]:
                    if event["type"] == "add":
                        if self.marketplace.add_to_cart(cart_id, event["product"]):
                            count += 1
                        else:
                            # Functional Utility: Fixed-interval backoff when out of stock.
                            time.sleep(self.retry_wait_time)


                    if event["type"] == "remove":
                        # Transaction reversal: restore item to global supply.
                        self.marketplace.remove_from_cart(cart_id, event["product"])
                        count += 1
            
            # Commit: finalize the session and print purchased inventory.
            products_list = self.marketplace.place_order(cart_id)
            for product in products_list:
                print(self.consumer_name + " bought " + str(product))

from threading import Lock

class Marketplace:
    """
    Central coordinator managing inventory buffers and shopper sessions.
    Functional Utility: Uses granular mutexes to protect different state 
    mutations, ensuring safe cross-producer item transfers.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace hub.
        @param queue_size_per_producer: Capacity limit per supply line.
        """
        # Granular Synchronization primitives.
        self.register_lock = Lock()
        self.add_remove_lock = Lock()
        self.cart_lock = Lock()

        self.queue_capacity = queue_size_per_producer
        self.nr_producers = 0
        self.nr_carts = 0
        
        # Persistence mapping: tracks supply lines and active sessions.
        self.producer_queues = {}
        self.carts = []

    def register_producer(self):
        """
        Allocates a new unique supply line index.
        Logic: Uses a dedicated mutex to ensure atomic registration.
        """
        with self.register_lock:
            self.nr_producers += 1
            producer_id = "prod" + str(self.nr_producers)
            self.producer_queues[producer_id] = []

        return producer_id

    def publish(self, producer_id, product):
        """
        Accepts a product from a producer into the local supply buffer.
        Logic: verifies producer capacity before publication.
        @return: True if accepted, False otherwise.
        """
        if len(self.producer_queues[producer_id]) < self.queue_capacity:
            self.producer_queues[producer_id].append(product)
            return True
        return False

    def new_cart(self):
        """Creates a new unique consumer session."""
        with self.cart_lock:
            cart_id = self.nr_carts
            self.carts.append([])
            self.nr_carts += 1
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Transfers an item from a producer's buffer to a consumer cart.
        Logic: Performs a global search across all buffers and atomically moves 
        the item if found, updating the origin map.
        """
        for (producer_id, products_queue) in self.producer_queues.items():
            if product in products_queue:
                # Transactional Transfer.
                products_queue.remove(product)
                # Association: stores both the product and its origin producer.
                self.carts[cart_id].append((product, producer_id))
                return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Restores a product from a cart back to its originating producer.
        Logic: uses the add_remove_lock to protect the session list update.
        """
        with self.add_remove_lock:
            index = 0
            for (cart_product, producer_id) in self.carts[cart_id]:
                if cart_product == product:
                    # Transaction Reversal.
                    self.producer_queues[producer_id].append(product)
                    break
                index += 1
        # Atomic removal from the session.
        self.carts[cart_id].pop(index)

    def place_order(self, cart_id):
        """Finalizes the purchase and returns the manifest of session goods."""
        return [elem[0] for elem in self.carts[cart_id]]


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
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        Main production cycle.
        Algorithm: Iterative manufacturing with synchronous backoff retries.
        """
        while True:
            for product in self.products:
                count = 0
                while count < product[1]:
                    if self.marketplace.publish(self.producer_id, product[0]):
                        # Simulate manufacturing time.
                        time.sleep(product[2])
                        count += 1
                    else:
                        # Functional Utility: Poll backoff when hub is full.
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """Base data model for goods."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """Product specialization."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """Product specialization."""
    acidity: str
    roast_level: str
