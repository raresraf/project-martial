"""
@file consumer.py
@brief Implements a multi-threaded producer-consumer simulation with a marketplace, defining product types and concurrency control.

This module sets up a system where `Producer` threads generate `Product` instances,
which are managed by a `Marketplace`. `Consumer` threads then interact with this
`Marketplace` to add and remove products from their shopping carts, and finally
place orders. It demonstrates multi-threading concepts and basic synchronization
using a single `Lock` within the `Marketplace` for shared resource protection.
"""

from threading import Thread, Lock
import time
from dataclasses import dataclass # Used for Product data structures


class Consumer(Thread):
    """
    @class Consumer
    @brief Represents a customer thread that interacts with the Marketplace to buy products.

    Each consumer thread simulates a shopping experience by obtaining a new cart,
    executing a series of 'add' and 'remove' actions for various products, and
    ultimately placing an order. It incorporates a retry mechanism for 'add'
    operations if the desired product is not immediately available.
    """

    # Invariant: Stores the name of the consumer thread, primarily for output.
    name = None

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a Consumer thread.
        @param carts A list of shopping cart definitions, where each cart is a list of product actions.
                     Each action is a dictionary with 'type' (e.g., "add", "remove"), 'quantity', and 'product'.
        @param marketplace The shared Marketplace instance this consumer will interact with.
        @param retry_wait_time The time (in seconds) to wait before retrying an 'add' action if the product is unavailable.
        @param kwargs Additional keyword arguments, including 'name', passed to the Thread constructor.
        """
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        # Invariant: Assigns the thread's name from kwargs for identification in output.
        self.name = kwargs["name"]

    def run(self):
        """
        @brief The main execution logic for the Consumer thread.

        Block Logic:
        1. For each predefined cart configuration, a new cart is requested from the marketplace.
        2. Iterates through the product actions (add/remove) within each cart.
        3. For 'add' actions, it repeatedly tries to add the product, waiting if unavailable.
        4. For 'remove' actions, it removes products from the cart.
        5. Finally, it places the order for the current cart and prints the bought items.
        """
        # Block Logic: Iterates through each distinct shopping cart scenario defined for this consumer.
        for cart in self.carts:
            cart_id = self.marketplace.new_cart() # Inline: Requests a new unique shopping cart from the marketplace.
            # Block Logic: Processes each product action within the current cart.
            for prod in cart:
                # Pre-condition: Checks if the action is to add a product.
                if prod["type"] == "add":
                    # Block Logic: Continuously attempts to add the specified quantity of the product.
                    # Invariant: `prod["quantity"]` decreases until all items are added.
                    while prod["quantity"] > 0:
                        # Pre-condition: Attempts to add the product to the cart.
                        check = self.marketplace.add_to_cart(cart_id, prod["product"])
                        if check:
                            prod["quantity"] -= 1 # Inline: Decrements quantity on successful addition.
                        else:
                            # Block Logic: If adding fails, waits before retrying.
                            time.sleep(self.retry_wait_time)
                else: # Action type is "remove".
                    # Block Logic: Continuously attempts to remove the specified quantity of the product.
                    # Invariant: `prod["quantity"]` decreases until all items are removed.
                    while prod["quantity"] > 0:
                        self.marketplace.remove_from_cart(cart_id, prod["product"])
                        prod["quantity"] -= 1 # Inline: Decrements quantity on successful removal.
            # Block Logic: Places the order for the completed cart.
            cart_list = self.marketplace.place_order(cart_id)
            cart_list.reverse() # Inline: Reverses the list for printing, possibly for chronological order of purchase.
            # Block Logic: Prints each product that was successfully bought.
            for elem in cart_list:
                print(self.name + " bought " + str(elem))


class Marketplace:
    """
    @class Marketplace
    @brief Manages product inventories, producer registrations, customer carts, and order processing.

    This class serves as the central coordination point for all producer and consumer
    interactions. It maintains global counters for producer and cart IDs and uses
    a single `Lock` (`self.lock`) to attempt to protect critical sections of data.
    Note: The locking strategy for `add_to_cart` and `place_order` might not
    provide full thread safety due to fine-grained lock acquire/release inside loops.
    """

    # Invariant: A class-level counter for assigning unique producer IDs.
    prod_id = 0
    # Invariant: A class-level counter for assigning unique cart IDs.
    cart_id = 0

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace with a specified queue size for producers.
        @param queue_size_per_producer The maximum number of products a single producer can have in its inventory queue.
        """
        self.queue_size_per_producer = queue_size_per_producer
        # Invariant: Dictionary mapping producer IDs to their product inventory lists.
        self.prod_dict = {}
        # Invariant: Dictionary mapping cart IDs to lists of products currently in the cart.
        self.cart_dict = {}
        # Invariant: A single re-entrant lock used to protect access to shared marketplace data structures.
        self.lock = Lock()

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace and assigns it a unique ID.
        @return The newly assigned unique producer ID.
        """
        # Block Logic: Increments the class-level producer ID counter to get a new unique ID.
        Marketplace.prod_id += 1
        self.prod_dict[Marketplace.prod_id] = [] # Inline: Initializes an empty product list for the new producer.
        return Marketplace.prod_id

    def publish(self, producer_id, product):
        """
        @brief Allows a producer to publish a product to its inventory.
        The product is added only if the producer's inventory has capacity.
        @param producer_id The ID of the producer attempting to publish.
        @param product The product object to be published.
        @return `True` if the product was successfully published, `False` otherwise.
        """
        # Pre-condition: Checks if the producer's current inventory size is less than the allowed maximum.
        if len(self.prod_dict[producer_id]) < self.queue_size_per_producer:
            self.prod_dict[producer_id].append(product) # Inline: Adds the product to the producer's inventory.
            return True
        return False

    def new_cart(self):
        """
        @brief Creates a new shopping cart for a customer and assigns a unique cart ID.
        @return The newly assigned unique cart ID.
        """
        # Block Logic: Increments the class-level cart ID counter to get a new unique ID.
        Marketplace.cart_id += 1
        self.cart_dict[Marketplace.cart_id] = [] # Inline: Initializes an empty cart for the new ID.
        return Marketplace.cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Attempts to add a specific product to a customer's cart.

        Block Logic:
        1. Iterates through all known producers' inventories.
        2. If the product is found in a producer's inventory, it's moved to the customer's cart.
        Note: The locking mechanism `self.lock` is acquired and released inside nested loops,
        which might not ensure full thread safety for the entire search and transfer operation.
        @param cart_id The ID of the customer's cart.
        @param product The product to add.
        @return `True` if the product was successfully added, `False` otherwise (e.g., product not found).
        """
        # Block Logic: Iterates through each producer's inventory to find the product.
        for key in self.prod_dict:
            self.lock.acquire() # Inline: Acquires the global lock (potential issue: released inside loop).
            # Block Logic: Iterates through products in the current producer's inventory.
            for prod in self.prod_dict[key]:
                # Pre-condition: Checks if the current product matches the one to be added.
                if product == prod:
                    self.cart_dict[cart_id].append(product) # Inline: Adds product to cart.
                    self.lock.release() # Inline: Releases the global lock.
                    return True
            self.lock.release() # Inline: Releases the global lock.
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a customer's cart.

        Note: This implementation simply removes the product from the cart
        and does not return it to any producer's inventory, which is an
        oversight if a full producer-consumer cycle is intended.
        @param cart_id The ID of the customer's cart.
        @param product The product to remove.
        """
        # Block Logic: Removes the specified product from the given cart.
        self.cart_dict[cart_id].remove(product)

    def place_order(self, cart_id):
        """
        @brief Finalizes a customer's cart into an order and removes the ordered products
        from their respective producers' inventories.

        Block Logic:
        1. Iterates through products in the customer's cart.
        2. For each product, it finds the producer from which it originated (by matching product instances)
           and removes it from that producer's inventory.
        Note: The mechanism to find the original producer for each product in `self.cart_dict[cart_id]`
        involves iterating through all producer inventories, which is inefficient.
        @param cart_id The ID of the customer's cart.
        @return A list of products that were part of the placed order.
        """
        # Block Logic: Iterates through products in the cart to remove them from producer inventories.
        for prod in self.cart_dict[cart_id]:
            for key in self.prod_dict: # Block Logic: Iterates through producers.
                for product in self.prod_dict[key]: # Block Logic: Iterates through products of a producer.
                    # Pre-condition: Checks if the product in the cart matches a product in a producer's inventory.
                    if product == prod:
                        self.prod_dict[key].remove(product) # Inline: Removes the product from the producer's inventory.

        return self.cart_dict[cart_id]


class Producer(Thread):
    """
    @class Producer
    @brief Represents a thread that continuously registers with the Marketplace and publishes products.

    Each producer thread generates a stream of predefined products and attempts to add them
    to its allocated inventory space within the Marketplace. It handles scenarios where
    its product queue might be full by waiting and retrying publication.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a Producer thread.
        @param products A list of product definitions. Each definition is a list: `[product_object, quantity, sleep_time_after_publish]`.
        @param marketplace The shared Marketplace instance to interact with.
        @param republish_wait_time The time (in seconds) to wait if a product cannot be published due to a full queue.
        @param kwargs Additional keyword arguments, including 'daemon', passed to the Thread constructor.
        """
        Thread.__init__(self, daemon=kwargs["daemon"])
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        @brief The main execution logic for the Producer thread.

        Block Logic:
        1. Registers itself with the marketplace to obtain a unique producer ID.
        2. Enters an infinite loop to continuously publish its products.
        3. For each product in its list, it attempts to publish the specified quantity.
        4. If a product is successfully published, it waits for `sleep_time_after_publish`.
        5. If publishing fails (due to a full queue), it waits for `republish_wait_time`
           and retries the same product.
        """
        prod_id = self.marketplace.register_producer() # Inline: Registers with the marketplace and gets an ID.
        # Block Logic: The producer continuously attempts to publish its products.
        while True:
            # Block Logic: Iterates through each product definition to publish the specified quantity.
            for prod in self.products:
                quantity = prod[1] # Inline: Initial quantity to publish for the current product.
                # Block Logic: Attempts to publish the product until the specified quantity is met.
                # Invariant: `quantity` decreases on successful publish.
                while quantity > 0:
                    # Pre-condition: Attempts to publish the product to the marketplace.
                    check = self.marketplace.publish(prod_id, prod[0])
                    if check:
                        quantity -= 1 # Inline: Decrements quantity on successful publish.
                        time.sleep(prod[2]) # Inline: Waits for a specified time after successful publish.
                    else:
                        # Block Logic: If publish fails (queue full), waits and retries the same product.
                        time.sleep(self.republish_wait_time)


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @dataclass
    @brief Base data class for defining a product.

    Attributes:
        name (str): The name of the product.
        price (int): The price of the product.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @dataclass
    @brief Data class for representing a Tea product, inheriting from Product.

    Attributes:
        type (str): The type of tea (e.g., "Green", "Black").
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @dataclass
    @brief Data class for representing a Coffee product, inheriting from Product.

    Attributes:
        acidity (str): Describes the coffee's acidity level.
        roast_level (str): Describes the coffee's roast level.
    """
    acidity: str
    roast_level: str
