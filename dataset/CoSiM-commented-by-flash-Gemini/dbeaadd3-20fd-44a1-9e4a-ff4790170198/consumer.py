"""
@dbeaadd3-20fd-44a1-9e4a-ff4790170198/consumer.py
@brief Distributed marketplace simulation using semaphore-based signaling and re-entrant locking.
* Algorithm: Event-driven producer-consumer model where `Marketplace` acts as a monitor with `Semaphore` for item availability and `RLock` for shared state protection.
* Functional Utility: Orchestrates the lifecycle of products from creation (Producer) to acquisition (Consumer), ensuring thread-safe inventory management and synchronized order fulfillment.
"""

import time
from threading import Thread


class Consumer(Thread):
    """
    @brief Consumer entity that performs multi-step shopping transactions across multiple carts.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes the consumer with its assigned shopping lists and market connection.
        """
        super().__init__()
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def add_command(self, id_cart, product, quantity):
        """
        @brief Attempts to add a specific quantity of a product to the cart.
        Logic: Uses a busy-wait retry loop with sleep-based backoff for failed acquisitions.
        """
        for _ in range(quantity):
            status = False
            while not status:
                status = self.marketplace.add_to_cart(id_cart, product)
                if not status:
                    # Functional Utility: Throttles retry attempts during stock contention.
                    time.sleep(self.retry_wait_time)

    def remove_command(self, id_cart, product, quantity):
        """
        @brief Returns a specific quantity of a product from the cart to the marketplace.
        """
        for _ in range(quantity):
            self.marketplace.remove_from_cart(id_cart, product)

    def run(self):
        """
        @brief Main execution loop for shopping activities.
        Algorithm: Iterative processing of carts followed by final order display.
        """
        for carts in self.carts:
            # Logic: Initializes a new transaction context.
            id_cart = self.marketplace.new_cart()
            for i in carts:
                command = i.get('type')
                if command == 'add':
                    self.add_command(id_cart, i.get('product'), i.get('quantity'))
                else:
                    self.remove_command(id_cart, i.get('product'), i.get('quantity'))

            # Post-condition: Commits the transaction and displays successful purchases.
            return_list = self.marketplace.place_order(id_cart)

            for i in enumerate(return_list):
                res = self.kwargs.get('name') + " bought " + format(i[1])
                print(res)

import time
from threading import Semaphore, RLock


class Marketplace:
    """
    @brief Centralized hub for inventory tracking and transaction signaling.
    """

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the marketplace with capacity limits and synchronization primitives.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.id_producer = -1
        self.id_carts = -1
        self.producers_list = []   # Intent: Tracks available capacity (quota) for each producer.
        self.market_contains = []  # Intent: Global pool of products available for acquisition.
        self.carts_contains = []   # Intent: Registry of products held in active consumer carts.
        
        self.lock_producers = RLock() # Intent: Serializes producer onboarding and global order count.
        self.lock_consumers = RLock() # Intent: Serializes cart creation and ownership transitions.
        
        self.number_of_orders_placed = -1 # Intent: Counter for completed customer transactions.
        self.consumers_semaphore = Semaphore(0) # Intent: Signals item availability to waiting consumers.

    def register_producer(self):
        """
        @brief Onboards a new producer and initializes its inventory buffer.
        """
        self.market_contains.append([])
        self.producers_list.append(self.queue_size_per_producer)
        with self.lock_producers:
            self.id_producer += 1
            return self.id_producer

    def publish(self, producer_id, product, wait_time_for_making_product):
        """
        @brief Adds a product to the market if the producer's quota allows.
        Logic: Signals availability to consumers via semaphore and simulates production latency.
        """
        if self.producers_list[producer_id] != 0:
            self.market_contains[producer_id].append([product, True])
            self.producers_list[producer_id] -= 1
            # Logic: Increments availability signal.
            self.consumers_semaphore.release()
            # Domain: Manufacturing simulation delay.
            time.sleep(wait_time_for_making_product)
            return True
        return False

    def new_cart(self):
        """
        @brief Allocates a new transaction identifier.
        """
        with self.lock_consumers:
            self.id_carts += 1
            self.carts_contains.append([])
            return self.id_carts

    def add_to_cart(self, cart_id, product):
        """
        @brief Transfers a product unit from the market pool to a specific cart.
        Invariant: Uses consumers_semaphore to block until items are available.
        Algorithm: Search-and-reserve across all producer buffers.
        """
        self.consumers_semaphore.acquire()
        for lists in self.market_contains:
            for item in lists:
                # Logic: Atomic state transition (Available -> Taken).
                if item[0] is product and item[1] is True:
                    self.carts_contains[cart_id].append(product)
                    with self.lock_consumers:
                        # Logic: Restores quota for the producer of the consumed item.
                        self.producers_list[self.market_contains.index(lists)] += 1
                        item[1] = False
                    return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Returns a previously acquired item back to the market pool.
        """
        self.carts_contains[cart_id].remove(product)
        for lists in self.market_contains:
            for item in lists:
                # Logic: Atomic state transition (Taken -> Available).
                if item[0] is product and item[1] is False:
                    with self.lock_consumers:
                        self.producers_list[self.market_contains.index(lists)] -= 1
                        item[1] = True
        # Post-condition: Restores availability signal for other consumers.
        self.consumers_semaphore.release()

    def place_order(self, cart_id):
        """
        @brief Finalizes a transaction and returns the list of purchased products.
        """
        with self.lock_consumers:
            self.number_of_orders_placed += 1
            return_list = self.carts_contains[cart_id]
            return return_list

    def number_of_orders(self):
        """
        @brief Coordination utility to check if all initiated carts have been finalized.
        """
        with self.lock_producers:
            if self.number_of_orders_placed == self.id_carts:
                return False
            return True

import time
from threading import Thread


class Producer(Thread):
    """
    @brief Producer agent that generates goods as long as active consumer sessions exist.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes the producer with its production plan.
        """
        super().__init__()
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def helper_run(self, producer_id, command_info):
        """
        @brief Handles batch production for a single product type.
        Algorithm: Iterative publication with quota-based retry and completion check.
        """
        for _ in range(command_info[1]):
            status = False
            while not status:
                status = self.marketplace.publish(producer_id, command_info[0], command_info[2])
                if not status:
                    # Logic: Quota full; wait for market consumption.
                    time.sleep(self.republish_wait_time)
                # Logic: Shutdown check - stops production if all orders are finalized.
                if not self.marketplace.number_of_orders():
                    status = True

    def run(self):
        """
        @brief Main production lifecycle loop.
        """
        id_prod = self.marketplace.register_producer()
        time_to_run = True
        while time_to_run:
            for i in self.products:
                self.helper_run(id_prod, i)
            # Logic: Refresh lifecycle status based on global order completion.
            time_to_run = self.marketplace.number_of_orders()
