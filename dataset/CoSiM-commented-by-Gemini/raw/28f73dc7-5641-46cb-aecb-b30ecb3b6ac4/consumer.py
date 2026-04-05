"""A multi-threaded producer-consumer marketplace simulation.

This script implements a marketplace system where multiple Producer threads can
publish products and multiple Consumer threads can purchase them. The Marketplace
class acts as a central hub, using several locks to manage concurrent access to
shared data structures.
"""

from threading import Thread, Lock, currentThread
import time


class Consumer(Thread):
    """Represents a consumer that processes a list of shopping carts.

    Each consumer thread executes a series of 'add' or 'remove' operations for
    each cart, interacting with the shared marketplace to acquire products.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes a Consumer instance.

        Args:
            carts (list): A list of shopping lists (carts) to be processed. Each
                cart is a list of dictionaries, where each dictionary defines an
                operation ('add' or 'remove'), a product, and a quantity.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying an
                operation that failed (e.g., product not available).
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.mk_p = marketplace
        # Maps operation strings to their corresponding marketplace methods.
        self.ops = {"add": self.mk_p.add_to_cart,
                    "remove": self.mk_p.remove_from_cart}
        self.wait = retry_wait_time


    def run(self):
        """The main execution logic for the consumer thread.

        Iterates through each assigned cart, creates a new cart session in the
        marketplace, and processes each add/remove operation. It handles the
        varied return signals from the marketplace to determine if an
        operation was successful or if it needs to wait and retry.
        """
        for cart in self.carts:
            # Pre-condition: A new, unique cart ID is obtained from the marketplace
            # to begin a shopping session.
            _id = self.mk_p.new_cart()

            # Block Logic: Process all operations within a single shopping cart.
            for op_in_cart in cart:
                no_of_op = 0
                while no_of_op < op_in_cart["quantity"]:
                    # Executes the appropriate marketplace method ('add' or 'remove').
                    result = self.ops[op_in_cart["type"]](_id, op_in_cart["product"])

                    # Block Logic: Handles the result of the operation. The logic
                    # assumes `True` indicates success. `False` indicates a
                    # temporary failure (e.g., product out of stock), triggering
                    # a wait and retry.
                    if result is True:
                        no_of_op += 1
                    else:
                        time.sleep(self.wait)

            # Finalizes the transaction for the current cart.
            self.mk_p.place_order(_id)


class Marketplace:
    """Manages product inventory and transactions between producers and consumers.

    This class acts as the central, thread-safe repository for all products and
    shopping carts. It uses a set of locks to coordinate access to its shared
    data structures, preventing race conditions.

    Attributes:
        lock_reg (Lock): A lock for controlling access to producer registration,
            ensuring that each producer receives a unique ID.
        lock_carts (Lock): A lock for creating new carts, ensuring cart IDs are
            unique.
        lock_alter (Lock): A lock for modifying shared product lists during
            add-to-cart operations, making the check-and-remove atomic.
        print (Lock): A lock to ensure that print statements from different
            threads are not interleaved, producing clean output.
    """
    lock_reg = Lock()
    lock_carts = Lock()
    lock_alter = Lock()
    print = Lock()
    no_carts = None
    def __init__(self, queue_size_per_producer):
        """Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products any
                single producer can have listed in the marketplace at once.
        """
        self.no_carts = 0
        self.max_prod_q_size = queue_size_per_producer
        self.prods = []  # A global list of all available products.
        self.carts = {}  # A dictionary mapping cart IDs to lists of products.
        self.producers = {}  # A mapping of a product to its producer's ID.
        self.prod_q_sizes = []  # A list of current queue sizes, indexed by producer ID.

    def register_producer(self):
        """Registers a new producer and assigns a unique ID.

        Uses a lock to ensure that the producer ID generation is atomic.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        with self.lock_reg:
            _id = len(self.prod_q_sizes)
            self.prod_q_sizes.append(0)
        return _id

    def publish(self, producer_id, product):
        """Publishes a product from a producer to the marketplace.

        Checks if the producer's personal queue is full. If not, it adds the
        product to the global product list and updates the producer's queue size.

        Args:
            producer_id (str): The string representation of the producer's ID.
            product: The product to be published.

        Returns:
            bool: True if publishing was successful, False if the producer's
                queue was full.
        """
        _id = int(producer_id)

        # Pre-condition: The producer's individual queue must not be at capacity.
        if self.prod_q_sizes[_id] >= self.max_prod_q_size:
            return False

        # This block is not protected by a lock, which could lead to a race
        # condition if multiple threads from the same producer call publish.
        # However, the current Producer implementation is single-threaded per
        # instance, mitigating this risk.
        self.prod_q_sizes[_id] += 1
        self.prods.append(product)
        self.producers[product] = _id

        return True

    def new_cart(self):
        """Creates a new, empty shopping cart and returns its unique ID.

        Uses a lock to ensure that cart ID generation is atomic.

        Returns:
            int: The unique ID for the new cart.
        """
        with self.lock_carts:
            self.no_carts += 1
            cart_id = self.no_carts

        self.carts[cart_id] = []
        return cart_id

    def add_to_cart(self, cart_id, product):
        """Attempts to add a product from the marketplace to a consumer's cart.

        This operation is protected by a lock to ensure that checking for the
        product's availability and removing it from the global list is an
        atomic operation, preventing multiple consumers from grabbing the same item.

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

            # Atomically update producer queue size and remove from global list.
            self.prod_q_sizes[self.producers[product]] -= 1
            self.prods.remove(product)

        self.carts[cart_id].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a cart and returns it to the marketplace.

        This method adds the product back to the global pool and increments the
        original producer's queue size.

        Args:
            cart_id (int): The ID of the cart to remove from.
            product: The product to be removed and returned to the inventory.
        """
        # Note: This operation could be unsafe if the product is not in the cart.
        self.carts[cart_id].remove(product)
        self.prods.append(product)

        with self.lock_alter:
            self.prod_q_sizes[self.producers[product]] += 1

        return True


    def place_order(self, cart_id):
        """Finalizes an order, printing the purchased items for a given cart.

        The printing is synchronized with a lock to prevent garbled output from
        multiple consumer threads printing at the same time.

        Args:
            cart_id (int): The ID of the cart to be processed.

        Returns:
            list: The list of products that were in the finalized cart.
        """
        prod_list = self.carts.pop(cart_id, None)

        for product in prod_list:
            # Use a lock to make the print operation atomic.
            with self.print:
                print(currentThread().getName(), "bought", product)

        return prod_list


class Producer(Thread):
    """Represents a producer that generates and publishes products.

    The producer runs in an infinite loop, periodically attempting to publish
    a predefined list of products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes a Producer instance.

        Args:
            products (list): A list of products for the producer to generate.
                Each item is a tuple of (product_name, quantity, production_time).
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying to publish
                if the producer's marketplace queue is full.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)

        self.products = products
        self.mk_p = marketplace
        self.wait = republish_wait_time

        self._id = self.mk_p.register_producer()

    def run(self):
        """The main execution logic for the producer thread.

        Continuously cycles through its product list, attempting to publish
        each one. If publishing fails (e.g., its queue is full), it waits
        before retrying. If successful, it "waits" for a specified production
        time before producing the next item.
        """
        while True:
            # Block Logic: Iterates through the producer's catalog of items to create.
            for (prod, quant, wait) in self.products:
                i = 0
                while i < quant:
                    # Attempt to publish one item to the marketplace.
                    ret = self.mk_p.publish(str(self._id), prod)

                    if ret:
                        # Invariant: If publish is successful, simulate production
                        # time by sleeping before continuing.
                        time.sleep(wait)
                        i += 1
                    else:
                        # If the marketplace queue for this producer is full,
                        # wait for a short period and then retry.
                        time.sleep(self.wait)