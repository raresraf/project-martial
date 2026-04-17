"""
@dbeaadd3-20fd-44a1-9e4a-ff4790170198/consumer.py
@brief multi-threaded electronic marketplace with semaphore-based availability tracking.
This module implements a coordinated trading environment where Producers supply goods 
and Consumers execute transactions. The system utilizes a global semaphore to track 
the aggregate number of available items across all producers, ensuring that consumers 
only attempt to claim stock when the hub is non-empty. Individual item states are 
managed via availability flags and per-category re-entrant locks.

Domain: Concurrent Systems, Semaphore Coordination, Producer-Consumer Simulation.
"""

import time
from threading import Thread


class Consumer(Thread):
    """
    Simulation entity representing a shopper.
    Functional Utility: Executes automated shopping batches across multiple carts 
    using a polling strategy with backoff for resource acquisition.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the consumer thread.
        @param carts: Nested list of shopping operation batches.
        @param marketplace: Central trading coordinator.
        @param retry_wait_time: delay between failed acquisition attempts.
        """
        super().__init__()
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def add_command(self, id_cart, product, quantity):
        """
        Attempts to acquire a specific quantity of a product.
        Algorithm: Iterative poll-wait loop ensuring fulfillment before proceeding.
        """
        for _ in range(quantity):
            status = False
            while not status:
                status = self.marketplace.add_to_cart(id_cart, product)
                if not status:
                    # Backoff: wait for stock replenishment.
                    time.sleep(self.retry_wait_time)

    def remove_command(self, id_cart, product, quantity):
        """Restores a specific quantity from the cart back to the global supply."""
        for _ in range(quantity):
            self.marketplace.remove_from_cart(id_cart, product)

    def run(self):
        """
        Main execution loop for shopper actions.
        Logic: Orchestrates session creation, task execution, and order finalization.
        """
        for carts in self.carts:
            # Atomic creation of a new transaction context.
            id_cart = self.marketplace.new_cart()
            for i in carts:
                command = i.get('type')
                if command == 'add':
                    self.add_command(id_cart, i.get('product'), i.get('quantity'))
                else:
                    self.remove_command(id_cart, i.get('product'), i.get('quantity'))

            # Commit: finalize the session and print results.
            return_list = self.marketplace.place_order(id_cart)
            for i in enumerate(return_list):
                res = self.kwargs.get('name') + " bought " + format(i[1])
                print(res)

import time
from threading import Semaphore, RLock


class Marketplace:
    """
    Central hub for coordinating transactions and inventory availability.
    Functional Utility: Employs a global semaphore (consumers_semaphore) to 
    decouple supply production from consumption requests, maintaining 
    total order for item availability.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.
        @param queue_size_per_producer: Capacity limit per supply line.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.id_producer = -1
        self.id_carts = -1
        # Registry of available slots per producer.
        self.producers_list = []  
        # Structured storage for published items: [[(product, status), ...], ...]
        self.market_contains = []  
        # Storage for active consumer sessions.
        self.carts_contains = []  
        
        # specialized mutexes for registration and session management.
        self.lock_producers = RLock()
        self.lock_consumers = RLock()
        
        # Performance Heuristic: tracks completed transactions to manage lifecycle.
        self.number_of_orders_placed = -1  
        # Core Coordination: tracks total number of 'True' status items in market_contains.
        self.consumers_semaphore = Semaphore(0)

    def register_producer(self):
        """Allocates a new supply line and associated capacity tracking."""
        self.market_contains.append([])
        self.producers_list.append(self.queue_size_per_producer)
        with self.lock_producers:
            self.id_producer += 1
            return self.id_producer

    def publish(self, producer_id, product, wait_time_for_making_product):
        """
        Accepts a product into the market and increments the availability counter.
        Logic: verifies producer capacity and signals consumers via the semaphore.
        @return: True if accepted, False otherwise.
        """
        if self.producers_list[producer_id] != 0:
            # Atomic update of supply state.
            self.market_contains[producer_id].append([product, True])
            self.producers_list[producer_id] -= 1
            # Signal: an additional item is now available for consumption.
            self.consumers_semaphore.release()
            # Simulate manufacturing overhead.
            time.sleep(wait_time_for_making_product)
            return True
        return False

    def new_cart(self):
        """Initializes a new shopper session context."""
        with self.lock_consumers:
            self.id_carts += 1
            self.carts_contains.append([])
            return self.id_carts

    def add_to_cart(self, cart_id, product):
        """
        Transfers an item from the global pool to a consumer cart.
        Logic: Synchronizes via the semaphore before performing a linear search 
        for an available product unit.
        """
        # Block until at least one item is available in the hub.
        self.consumers_semaphore.acquire()
        for lists in self.market_contains:
            for item in lists:
                if item[0] is product and item[1] is True:
                    # atomic claim: move to session storage and update status.
                    self.carts_contains[cart_id].append(product)
                    with self.lock_consumers:
                        self.producers_list[self.market_contains.index(lists)] += 1
                        item[1] = False
                    return True
        return False

    def remove_from_cart(self, cart_id, product):
        """Restores a product from a cart back to its originating producer."""
        self.carts_contains[cart_id].remove(product)
        for lists in self.market_contains:
            for item in lists:
                if item[0] is product and item[1] is False:
                    # Transaction reversal: restore capacity and status.
                    with self.lock_consumers:
                        self.producers_list[self.market_contains.index(lists)] -= 1
                        item[1] = True
        # Signal: the restored item is now available again.
        self.consumers_semaphore.release()

    def place_order(self, cart_id):
        """Finalizes the session and updates the completion tracking heuristic."""
        with self.lock_consumers:
            self.number_of_orders_placed += 1
            return_list = self.carts_contains[cart_id]
            return return_list

    def number_of_orders(self):
        """Heuristic check to determine if all registered customers have completed checkout."""
        with self.lock_producers:
            if self.number_of_orders_placed == self.id_carts:
                return False
            return True

import time
from threading import Thread


class Producer(Thread):
    """
    Simulation thread representing a manufacturing entity.
    Functional Utility: Manages the production cycle and handles backpressure 
    via periodic retries, observing marketplace activity to determine termination.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        super().__init__()
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def helper_run(self, producer_id, command_info):
        """
        Executes the publication quota for a specific product category.
        Algorithm: Iterative try-wait loop with hub-based termination check.
        """
        for _ in range(command_info[1]):
            status = False
            while not status:
                status = self.marketplace.publish(producer_id, command_info[0], command_info[2])
                if not status:
                    # Backpressure backoff.
                    time.sleep(self.republish_wait_time)
                # Termination Check: stop production if all orders are filled.
                if not self.marketplace.number_of_orders():
                    status = True

    def run(self):
        """
        Main production cycle.
        Logic: Continuously supplies goods as long as active customer demand 
        is projected by the marketplace heuristic.
        """
        id_prod = self.marketplace.register_producer()
        time_to_run = True
        while time_to_run:
            for i in self.products:
                self.helper_run(id_prod, i)
            time_to_run = self.marketplace.number_of_orders()
