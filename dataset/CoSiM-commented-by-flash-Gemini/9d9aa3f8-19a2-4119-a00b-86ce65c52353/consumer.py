

"""
This module implements a multi-threaded simulation of a producer-consumer marketplace.
It defines `Consumer` threads that simulate shopping activities by adding/removing
products from carts and placing orders. `Producers` continually publish products
to the `Marketplace`, which acts as a central hub for managing available products,
carts, and facilitating transactions with thread-safe operations.

The module also defines a `Product` base class and its subclasses `Tea` and `Coffee`,
used to represent the items being traded in the marketplace.

Architectural Intent:
The system is designed to simulate concurrent buying and selling operations
where multiple producers and consumers interact with a shared marketplace.
Thread synchronization mechanisms (like `Lock` and `collections.defaultdict`)
are used to ensure data consistency and prevent race conditions during
product publishing, cart modifications, and order placement.
"""

from threading import Lock, Thread
import time
import collections
from dataclasses import dataclass


class Consumer(Thread):
    """
    Represents a consumer in the marketplace simulation.

    Each consumer runs as a separate thread, executing a predefined list of
    shopping cart operations (adding and removing products) and ultimately
    placing orders. It interacts with the `Marketplace` instance to perform
    these actions.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        :param carts: A list of cart operations. Each item in `carts` represents a
                      single shopping session, containing a list of operations
                      (e.g., {"type": "add", "product": Product, "quantity": int}).
        :param marketplace: The `Marketplace` instance the consumer will interact with.
        :param retry_wait_time: The time in seconds to wait before retrying a failed
                                marketplace operation (e.g., adding a product when stock is low).
        :param kwargs: Arbitrary keyword arguments, typically used for thread naming
                       (e.g., `name="Consumer-1"`).
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs  # Store kwargs for potential use, e.g., accessing 'name' for printing.
        self.lock = Lock()  # A lock for internal consumer state, if needed (currently not heavily used for shared data).

    def run(self):
        """
        The main execution method for the Consumer thread.

        It iterates through each defined shopping cart, performs the specified
        add/remove operations, and then places the final order, printing
        the items successfully bought.
        """
        # Block Logic: Iterate through each distinct shopping cart defined for this consumer.
        for i in self.carts:
            # Functional Utility: Request a new, unique cart ID from the marketplace.
            id_cart = self.marketplace.new_cart()

            # Block Logic: Process each individual operation (add or remove) within the current cart.
            for operation in i:
                operation_counter = 0  # Tracks the number of times this specific operation has been successfully performed.

                # Block Logic: Continue attempting the operation until the desired quantity is met.
                # This loop handles retries for 'add' operations if the product is unavailable.
                while operation_counter < operation["quantity"]:
                    # Functional Utility: Use a lock to protect critical sections,
                    # although for this `while` loop, it primarily ensures `operation_counter`
                    # is updated safely if there were more complex internal states shared.
                    with self.lock:
                        if operation["type"] == "add":
                            # Functional Utility: Attempt to add the product to the cart via the marketplace.
                            ret = self.marketplace.add_to_cart(id_cart, operation["product"])
                            if not ret:
                                # Inline: If adding failed (e.g., product out of stock), wait and retry.
                                time.sleep(self.retry_wait_time)
                            else:
                                # Inline: If successful, increment the counter for this operation.
                                operation_counter += 1
                        else:  # Operation type is "remove"
                            # Functional Utility: Remove the product from the cart via the marketplace.
                            self.marketplace.remove_from_cart(id_cart, operation["product"])
                            operation_counter += 1  # Increment counter as removal is typically successful if item was there.

            # Block Logic: Once all cart operations are complete for `id_cart`, place the order.
            # Functional Utility: Iterate through the products successfully placed in the order and print a confirmation.
            for product in self.marketplace.place_order(id_cart):
                print("%s bought %s" % (self.kwargs['name'], product))


class Marketplace:
    """
    A central hub for managing product listings from producers and shopping
    carts for consumers.

    It ensures thread-safe operations for registering producers, publishing
    products, creating new carts, adding/removing products from carts, and
    placing orders.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        :param queue_size_per_producer: The maximum number of products a single
                                        producer can have listed in the marketplace at one time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.id_producer = 0  # Counter for assigning unique producer IDs.
        self.id_cart = 0  # Counter for assigning unique cart IDs.

        # A dictionary where keys are producer IDs and values are lists of products
        # currently published by that producer. `defaultdict(list)` simplifies adding products.
        self.products_lists = collections.defaultdict(list)

        # A dictionary where keys are cart IDs and values are lists of products
        # currently in that cart.
        self.carts_lists = collections.defaultdict(list)

        # A dictionary to track which producer supplied which product to a cart.
        # Key: product, Value: list of producer_ids who supplied this product.
        # This is used for correctly returning products to a producer's stock upon removal from a cart.
        self.bought_items = collections.defaultdict(list)

        # Locks to ensure thread-safe access to marketplace data structures.
        self.lock_carts = Lock()
        self.lock_producers = Lock()

    def register_producer(self):
        """
        Registers a new producer with the marketplace and assigns a unique ID.

        :return: The newly assigned unique integer ID for the producer.
        """
        # Block Logic: Acquire lock to ensure thread-safe incrementing of producer ID.
        with self.lock_producers:
            self.id_producer += 1
            return self.id_producer

    def publish(self, producer_id, product):
        """
        Publishes a product from a specific producer to the marketplace.

        The operation is successful only if the producer's current queue size
        does not exceed `queue_size_per_producer`.

        :param producer_id: The ID of the producer publishing the product.
        :param product: The `Product` object to be published.
        :return: `True` if the product was successfully published, `False` otherwise.
        """
        # Pre-condition: Check if the producer has reached its maximum allowed number of published products.
        if producer_id in self.products_lists:
            if len(self.products_lists[producer_id]) >= self.queue_size_per_producer:
                return False  # Cannot publish, queue is full.

        # Functional Utility: Add the product to the producer's list of published products.
        self.products_lists[producer_id].append(product)
        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart and assigns it a unique ID.

        :return: The unique integer ID of the newly created cart.
        """
        # Block Logic: Acquire lock to ensure thread-safe incrementing of cart ID.
        with self.lock_carts:
            self.id_cart += 1
            return self.id_cart

    def add_to_cart(self, cart_id, product):
        """
        Adds a specified product to a consumer's cart.

        This involves finding the product among available listings, moving it
        from a producer's inventory to the cart, and tracking the transaction.

        :param cart_id: The ID of the cart to which the product should be added.
        :param product: The `Product` object to add.
        :return: `True` if the product was successfully added, `False` if not found or unavailable.
        """
        # Block Logic: Iterate through all producers' product lists to find the desired product.
        for key, values in self.products_lists.items(): # `key` here refers to `producer_id`
            # Block Logic: Iterate through products listed by the current producer.
            for j in values:  # `j` here refers to a `Product` object
                if j == product:
                    # Functional Utility: Track which producer provided this product (for potential returns).
                    self.bought_items[j].append(key)
                    # Functional Utility: Add the product to the consumer's cart.
                    self.carts_lists[cart_id].append(j)
                    # Functional Utility: Remove the product from the producer's available stock.
                    self.products_lists[key].remove(j)
                    return True  # Product successfully added.
        return False  # Product not found or unavailable from any producer.

    def remove_from_cart(self, cart_id, product):
        """
        Removes a specified product from a consumer's cart and returns it to
        the original producer's stock.

        :param cart_id: The ID of the cart from which the product should be removed.
        :param product: The `Product` object to remove.
        """
        # Functional Utility: Remove the product from the consumer's cart.
        self.carts_lists[cart_id].remove(product)
        # Functional Utility: Return the product to the stock of the producer it was originally bought from.
        # `self.bought_items[product].pop()` retrieves the last producer ID that sold this product.
        self.products_lists[self.bought_items[product].pop()].append(product)

    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart.

        In this simulation, placing an order simply means returning the list
        of products that were successfully added to the cart.

        :param cart_id: The ID of the cart to place the order for.
        :return: A list of `Product` objects that are in the specified cart.
        """
        return self.carts_lists[cart_id]


class Producer(Thread):
    """
    Represents a producer in the marketplace simulation.

    Each producer runs as a separate thread, continuously publishing a
    predefined list of products to the `Marketplace`. It handles
    retries if the marketplace's capacity for its products is full.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        :param products: A list of products this producer will offer. Each item is a tuple
                         (Product object, quantity to publish, time to wait after publishing one).
        :param marketplace: The `Marketplace` instance the producer will interact with.
        :param republish_wait_time: The time in seconds to wait before retrying to publish
                                    a product if the marketplace is full.
        :param kwargs: Arbitrary keyword arguments, typically used for thread naming
                       (e.g., `name="Producer-1"`).
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # Functional Utility: Register with the marketplace to obtain a unique producer ID.
        self.id_producer = self.marketplace.register_producer()
        self.lock = Lock()  # A lock for internal producer state, if needed.

    def run(self):
        """
        The main execution method for the Producer thread.

        It continuously attempts to publish its products to the marketplace,
        respecting quantity limits and retry delays.
        """
        # Block Logic: The producer continuously tries to publish products.
        while True:
            # Block Logic: Iterate through each type of product this producer offers.
            for i in self.products:  # `i` is a tuple (Product object, quantity, wait_time)
                operation_counter = 0  # Tracks how many of the current product have been published.

                # Block Logic: Keep publishing the current product until the desired quantity is met.
                while operation_counter < i[1]:
                    # Functional Utility: Use a lock to protect critical sections,
                    # especially when interacting with the marketplace.
                    with self.lock:
                        # Functional Utility: Attempt to publish the product to the marketplace.
                        if not self.marketplace.publish(self.id_producer, i[0]):
                            # Inline: If publishing failed (e.g., marketplace queue full), wait and retry.
                            time.sleep(self.republish_wait_time)
                        else:
                            # Inline: If successful, increment the count of published items for this product.
                            operation_counter += 1
                            # Inline: Wait for a product-specific time before attempting to publish another.
                            time.sleep(i[2])


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Base class for all products in the marketplace.

    Uses `dataclass` for concise definition of data-holding classes.
    `frozen=True` makes instances immutable.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Represents a specific type of product: Tea.

    Inherits from `Product` and adds a `type` attribute.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Represents a specific type of product: Coffee.

    Inherits from `Product` and adds `acidity` and `roast_level` attributes.
    """
    acidity: str
    roast_level: str
