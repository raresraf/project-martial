"""
A multithreaded producer-consumer simulation of an online marketplace.

This script models a marketplace with producers who publish products and
consumers who purchase them. The simulation uses Python's threading module,
including RLock for protecting shared data and a Semaphore for signaling
product availability between producers and consumers.
"""

import time
from threading import Thread, Semaphore, RLock


class Consumer(Thread):
    """
    Represents a consumer thread that simulates purchasing items.

    Each consumer processes a list of commands (add/remove) for a series of carts,
    interacts with the central marketplace to acquire or return products, and
    finally places an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer thread.

        Args:
            carts (list): A list of shopping sessions, each with a list of commands.
            marketplace (Marketplace): The shared marketplace object.
            retry_wait_time (int): Time in seconds to wait before retrying a failed 'add' command.
            **kwargs: Used to pass in the consumer's name for printing.
        """
        super().__init__()
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def add_command(self, id_cart, product, quantity):
        """
        Executes an 'add' command by repeatedly trying to add a product to the cart.

        Args:
            id_cart (int): The ID of the current shopping cart.
            product: The product to add.
            quantity (int): The number of times to add the product.
        """
        for _ in range(quantity):
            status = False
            # Block Logic: This busy-wait loop retries adding a product.
            # Note: The primary blocking mechanism is the semaphore in the Marketplace,
            # making this redundant if the semaphore logic is sufficient.
            while not status:
                status = self.marketplace.add_to_cart(id_cart, product)
                if not status:
                    time.sleep(self.retry_wait_time)

    def remove_command(self, id_cart, product, quantity):
        """
        Executes a 'remove' command.

        Args:
            id_cart (int): The ID of the current shopping cart.
            product: The product to remove.
            quantity (int): The number of times to remove the product.
        """
        for _ in range(quantity):
            self.marketplace.remove_from_cart(id_cart, product)

    def run(self):
        """The main entry point for the consumer thread."""
        for carts in self.carts:
            id_cart = self.marketplace.new_cart()
            for i in carts:
                command = i.get('type')
                if command == 'add':
                    self.add_command(id_cart, i.get('product'), i.get('quantity'))
                else:
                    self.remove_command(id_cart, i.get('product'), i.get('quantity'))

            return_list = self.marketplace.place_order(id_cart)

            # Block Logic: Prints the items bought in the final order.
            for i in enumerate(return_list):
                res = self.kwargs.get('name') + " bought " + format(i[1])
                print(res)


class Marketplace:
    """
    Manages the state of the marketplace, including producers, products, and carts.

    This class is the synchronization hub. It uses an RLock to protect shared data
    and a Semaphore to signal product availability, blocking consumers when the
    marketplace is empty and waking them when a producer adds a new item.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.

        Args:
            queue_size_per_producer (int): Max products a single producer can have listed.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.id_producer = -1
        self.id_carts = -1
        self.producers_list = []      # Tracks available capacity for each producer.
        self.market_contains = []     # The actual product listings.
        self.carts_contains = []      # The contents of each shopping cart.
        self.lock_producers = RLock()
        self.lock_consumers = RLock()
        self.number_of_orders_placed = -1
        # Semaphore acts as a counter of available products for consumers.
        self.consumers_semaphore = Semaphore(0)

    def register_producer(self):
        """
        Registers a new producer, allocating space for their products.
        Returns:
            int: The ID for the new producer.
        """
        self.market_contains.append([])
        self.producers_list.append(self.queue_size_per_producer)
        with self.lock_producers:
            self.id_producer += 1
            return self.id_producer

    def publish(self, producer_id, product, wait_time_for_making_product):
        """
        Allows a producer to publish a product to the marketplace.

        Returns:
            bool: True if publishing was successful, False otherwise.
        """
        if self.producers_list[producer_id] != 0:
            self.market_contains[producer_id].append([product, True])
            self.producers_list[producer_id] -= 1
            # Functional Utility: Signals to one waiting consumer that a product is available.
            self.consumers_semaphore.release()
            time.sleep(wait_time_for_making_product)
            return True
        return False

    def new_cart(self):
        """Creates a new empty cart for a consumer."""
        with self.lock_consumers:
            self.id_carts += 1
            self.carts_contains.append([])
            return self.id_carts

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a consumer's cart. This method will block if no products are available.

        Returns:
            bool: True if the product was added, False otherwise.
        """
        # Functional Utility: Blocks if the semaphore count is zero (no products available).
        # Waits here until a producer calls `release()`.
        self.consumers_semaphore.acquire()
        for lists in self.market_contains:
            for item in lists:
                # Pre-condition: Find the requested product and ensure it's available.
                if item[0] is product and item[1] is True:
                    self.carts_contains[cart_id].append(product)
                    with self.lock_consumers:
                        self.producers_list[self.market_contains.index(lists)] += 1
                        # Invariant: Mark the item as unavailable.
                        item[1] = False
                    return True
        return False

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a cart, making it available again."""
        self.carts_contains[cart_id].remove(product)
        for lists in self.market_contains:
            for item in lists:
                if item[0] is product and item[1] is False:
                    with self.lock_consumers:
                        self.producers_list[self.market_contains.index(lists)] -= 1
                        item[1] = True
        # A removed item is now available, so signal consumers.
        self.consumers_semaphore.release()

    def place_order(self, cart_id):
        """Finalizes an order and returns the items in the cart."""
        with self.lock_consumers:
            self.number_of_orders_placed += 1
            return_list = self.carts_contains[cart_id]
            return return_list

    def number_of_orders(self):
        """
        A mechanism to signal producers to shut down once all consumers are done.
        
        Returns:
            bool: False if all carts have been processed into orders, True otherwise.
        """
        with self.lock_producers:
            if self.number_of_orders_placed == self.id_carts:
                return False
            return True


class Producer(Thread):
    """
    Represents a producer thread that publishes products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes the Producer thread.
        
        Args:
            products (list): A list of product recipes to produce.
            marketplace (Marketplace): The shared marketplace object.
            republish_wait_time (int): Time to wait before retrying a failed publish.
            **kwargs: Not used.
        """
        super().__init__()
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def helper_run(self, producer_id, command_info):
        """
        Helper method to handle the publishing of a single product type.
        
        Args:
            producer_id (int): The ID of this producer.
            command_info (tuple): Contains product, quantity, and creation time.
        """
        for _ in range(command_info[1]):
            status = False
            # Block Logic: Retry publishing until successful or simulation ends.
            while not status:
                status = self.marketplace.publish(producer_id, command_info[0], command_info[2])
                if not status:
                    time.sleep(self.republish_wait_time)
                # Check for simulation end condition.
                if not self.marketplace.number_of_orders():
                    status = True

    def run(self):
        """The main entry point for the producer thread."""
        id_prod = self.marketplace.register_producer()
        time_to_run = True
        # Invariant: Producer continues to run as long as there are active consumers.
        while time_to_run:
            for i in self.products:
                self.helper_run(id_prod, i)
            # Check if all consumers have finished their work.
            time_to_run = self.marketplace.number_of_orders()
