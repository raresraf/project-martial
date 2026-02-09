"""
A multi-threaded producer-consumer marketplace simulation.

This script implements a marketplace system where multiple Producer threads can publish
products and multiple Consumer threads can purchase them. The Marketplace class acts
as a central hub, using several locks to manage concurrent access to shared data
structures, though some operations have complex and potentially unsafe interactions.
"""

from threading import Thread
import time


class Consumer(Thread):
    """
    Represents a consumer that processes a list of shopping carts.

    Each consumer thread executes a series of 'add' or 'remove' operations for
    each cart, interacting with the shared marketplace to acquire products.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer instance.

        Args:
            carts (list): A list of shopping lists (carts) to be processed.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait before retrying an operation
                                     that failed (e.g., product not available).
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.mk_p = marketplace
        # Maps operation types to the corresponding marketplace methods.
        self.ops = {"add": self.mk_p.add_to_cart,
                    "remove": self.mk_p.remove_from_cart}
        self.wait = retry_wait_time


    def run(self):
        """
        The main execution logic for the consumer thread.

        Iterates through each assigned cart, creates a new cart session in the
        marketplace, and processes each add/remove operation. It handles the
        varied return signals from the marketplace to determine if an
        operation was successful or if it needs to wait and retry.
        """
        for cart in self.carts:
            # Pre-condition: A new, unique cart ID is obtained from the marketplace.
            _id = self.mk_p.new_cart()

            # Block Logic: Process all operations within a single shopping cart.
            for op_in_cart in cart:
                no_of_op = 0
                while no_of_op < op_in_cart["quantity"]:
                    # Executes the appropriate marketplace method ('add' or 'remove').
                    result = self.ops[op_in_cart["type"]](_id, op_in_cart["product"])

                    # Block Logic: Handles the result of the operation. The logic
                    # assumes `None` or `True` indicates success for an operation.
                    # `False` indicates a temporary failure, triggering a wait.
                    if result is None:
                        no_of_op += 1
                    elif result is True:
                        no_of_op += 1
                    else:
                        time.sleep(self.wait)

            # Finalizes the transaction for the current cart.
            self.mk_p.place_order(_id)

from threading import Lock, currentThread

class Marketplace:
    """
    Manages product inventory and transactions between producers and consumers.

    This class uses a set of locks to coordinate access to its shared data
    structures, including a global product list and producer-specific queues.
    """
    # Lock for controlling access to producer registration.
    lock_reg = Lock()
    # Lock for creating new carts.
    lock_carts = Lock()
    # Lock for altering shared product lists.
    lock_alter = Lock()
    # Lock to ensure print statements are atomic and not interleaved.
    print = Lock()
    no_carts = None 
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products any
                                           single producer can have listed at once.
        """
        self.no_carts = 0
        self.max_prod_q_size = queue_size_per_producer
        self.prods = []  # A global list of all available products.
        self.carts = {}  # A dictionary mapping cart IDs to lists of products.
        self.producers = {}  # A mapping of a product to its producer's ID.
        self.prod_q_sizes = []  # A list of queue sizes, indexed by producer ID.

    def register_producer(self):
        """
        Safely registers a new producer, providing a unique ID.

        Returns:
            int: The unique ID assigned to the producer.
        """
        with self.lock_reg:
            _id = len(self.prod_q_sizes)
            self.prod_q_sizes.append(0)

        return _id

    def publish(self, producer_id, product):
        """
        Publishes a product to the global marketplace inventory.

        Checks if the producer's queue is full. If not, it adds the product
        to the global product list and updates the producer's queue size.

        Args:
            producer_id (str): The string representation of the producer's ID.
            product: The product to be published.

        Returns:
            bool: True if publishing was successful, False otherwise.
        """
        _id = int(producer_id)

        # Pre-condition: The producer's individual queue must not be at capacity.
        if self.prod_q_sizes[_id] >= self.max_prod_q_size:
            return False

        self.prod_q_sizes[_id] += 1
        self.prods.append(product)
        self.producers[product] = _id

        return True

    def new_cart(self):
        """
        Safely creates a new, empty cart and returns its unique ID.

        Returns:
            int: The unique ID for the new cart.
        """
        with self.lock_carts:
            self.no_carts += 1
            cart_id = self.no_carts

        self.carts[cart_id] = []

        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Attempts to add a product from the global pool to a consumer's cart.

        This operation is protected by a lock to ensure that checking for the
        product and removing it from the global list is an atomic operation.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product: The product to add.

        Returns:
            bool: True if the product was successfully added, False if not found.
        """
        with self.lock_alter:
            # Pre-condition: The product must be present in the global inventory.
            if product not in self.prods:
                return False

            # Decrement producer's queue size and remove from global list.
            self.prod_q_sizes[self.producers[product]] -= 1
            self.prods.remove(product)

        self.carts[cart_id].append(product)

        return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to the global product pool.

        Args:
            cart_id (int): The ID of the cart to remove from.
            product: The product to be removed.
        """
        self.carts[cart_id].remove(product)
        self.prods.append(product)

        with self.lock_alter:
            self.prod_q_sizes[self.producers[product]] += 1


    def place_order(self, cart_id):
        """
        Finalizes an order, printing the purchased items.

        The printing is synchronized with a lock to prevent garbled output from
        multiple consumer threads printing at the same time.

        Args:
            cart_id (int): The ID of the cart to be processed.

        Returns:
            list: The list of products that were in the cart.
        """
        prod_list = self.carts.pop(cart_id, None)

        for product in prod_list:
            with self.print:
                print(currentThread().getName(), "bought", product)

        return prod_list


from threading import Thread
import time

class Producer(Thread):
    """
    Represents a producer that generates and publishes products.

    The producer runs in an infinite loop, periodically attempting to publish
    a predefined list of products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer instance.

        Args:
            products (list): A list of products to generate, including quantity
                             and production time.
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying to publish
                                         if the marketplace is full.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)

        self.products = products
        self.mk_p = marketplace
        self.wait = republish_wait_time

        self._id = self.mk_p.register_producer()

    def run(self):
        """
        The main execution logic for the producer thread.

        Continuously cycles through its product list, attempting to publish
        each one. If publishing fails (e.g., queue is full), it waits before
        retrying. If successful, it waits for a specified time before
        producing the next item.
        """
        while True:
            # Block Logic: Iterates through the producer's catalog of items to create.
            for (prod, quant, wait) in self.products:
                i = 0
                while i < quant:
                    # Attempt to publish one item.
                    ret = self.mk_p.publish(str(self._id), prod)

                    if ret:
                        # Invariant: If publish is successful, wait for the item's
                        # specific production time before continuing.
                        time.sleep(wait)
                        i += 1
                    else:
                        # If the marketplace queue is full, wait and retry.
                        time.sleep(self.wait)