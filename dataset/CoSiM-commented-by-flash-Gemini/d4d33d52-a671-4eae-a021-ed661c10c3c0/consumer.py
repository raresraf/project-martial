"""
@d4d33d52-a671-4eae-a021-ed661c10c3c0/consumer.py
@brief Implements a multi-threaded producer-consumer system with a marketplace.

This module defines the core components for a concurrent marketplace simulation,
including consumer threads that place orders, producer threads that supply products,
and a central marketplace managing inventory and transactions. It utilizes threading
primitives for synchronization and data structures to represent products and carts.
"""

from threading import Thread, Lock
import time
from dataclasses import dataclass


class Consumer(Thread):
    """
    @brief Represents a consumer thread that interacts with the marketplace to buy and remove products.

    Inherits from `threading.Thread` and simulates a customer placing orders by
    adding and removing products from a shopping cart within a shared marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping carts, each containing a list of orders.
            marketplace (Marketplace): The shared marketplace instance to interact with.
            retry_wait_time (float): The time in seconds to wait before retrying an operation
                                     if a product is not immediately available.
            **kwargs: Arbitrary keyword arguments, including 'name' for the consumer.
        """
        Thread.__init__(self)

        self.name = kwargs["name"]
        self.no_product_wait_time = retry_wait_time
        self.shop = marketplace
        self.carts = carts

    def run(self):
        """
        @brief Executes the consumer's purchasing logic.

        Iterates through each cart assigned to this consumer, simulates adding and
        removing products, and finally places the order.
        """
        for cart in self.carts:
            # Pre-condition: 'cart' is a list of orders.
            # Invariant: Each order in 'cart' will be processed.
            cart_id = self.shop.new_cart()
            for order in cart:
                # Pre-condition: 'order' is a dictionary with 'type', 'product', and 'quantity'.
                # Invariant: The order operation (add or remove) will be attempted.
                if order["type"] == "add":
                    self.add_to_cart(cart_id, order)
                elif order["type"] == "remove":
                    self.remove_from_cart(cart_id, order)
            bought = self.shop.place_order(cart_id)
            self.print_what_was_bought(bought)

    def add_to_cart(self, cart_id, order):
        """
        @brief Adds a specified quantity of a product to the consumer's cart.

        If the product is not immediately available in the marketplace, the consumer
        waits for a defined period and retries.

        Args:
            cart_id (int): The ID of the consumer's shopping cart.
            order (dict): A dictionary containing 'product' and 'quantity' to add.
        """
        i = order["quantity"]
        # Pre-condition: 'i' is the quantity of the product to add.
        # Invariant: The loop continues until the desired quantity is added or an unresolvable issue occurs.
        while i > 0:
            if not self.shop.add_to_cart(cart_id, order["product"]):
                # Inline: Product not available, wait and retry.
                time.sleep(self.no_product_wait_time)
                continue
            i -= 1

    def remove_from_cart(self, cart_id, order):
        """
        @brief Removes a specified quantity of a product from the consumer's cart.

        Args:
            cart_id (int): The ID of the consumer's shopping cart.
            order (dict): A dictionary containing 'product' and 'quantity' to remove.
        """
        # Pre-condition: 'order["quantity"]' specifies how many items to remove.
        # Invariant: The loop attempts to remove each item from the cart.
        for _ in range(order["quantity"]):
            self.shop.remove_from_cart(cart_id, order["product"])

    def print_what_was_bought(self, bought):
        """
        @brief Prints the list of products successfully bought by the consumer.

        Args:
            bought (list): A list of products that were successfully purchased.
        """
        # Pre-condition: 'bought' is a list of products.
        # Invariant: Each product in 'bought' will be printed.
        for product in bought:
            print(self.name, "bought", product)


class Marketplace:
    """
    @brief Central hub for managing products, producers, and consumer carts.

    Handles registration of producers, publishing of products, creation of carts,
    adding/removing items from carts, and processing orders, ensuring thread safety
    through the use of locks.
    """

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have in the marketplace at any time.
        """
        # Functional Utility: Protects access to the producer registration list.
        self.reg_prod_lock = Lock()
        # Functional Utility: Stores the current count of published items for each producer.
        self.prod_list = []
        # Functional Utility: Defines the maximum queue size for each producer.
        self.prod_max_queue = queue_size_per_producer

        # Functional Utility: Protects access to the dictionary of available shop items.
        self.shop_items_lock = Lock()
        # Functional Utility: Stores available products, mapping product to (lock, list of producer_ids).
        self.shop_items = dict()

        # Functional Utility: Stores consumer carts, mapping cart_id to a list of products.
        self.carts = dict()
        # Functional Utility: Protects access to the cart ID counter.
        self.cart_id_lock = Lock()
        # Functional Utility: Counter for assigning unique cart IDs.
        self.cart_id = 0

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace.

        Assigns a unique ID to the producer and initializes their product count to zero.

        Returns:
            int: The unique ID assigned to the registered producer.
        """
        # Block Logic: Ensures exclusive access to the producer list during registration.
        self.reg_prod_lock.acquire()
        self.prod_list.append(0)
        ret_id = len(self.prod_list) - 1
        self.reg_prod_lock.release()
        return ret_id

    def publish(self, producer_id, product) -> bool:
        """
        @brief Publishes a product from a producer to the marketplace.

        The product is added to the marketplace if the producer has not exceeded
        its maximum allowed queue size.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (Product): The product to publish.

        Returns:
            bool: True if the product was successfully published, False otherwise.
        """
        # Block Logic: Checks if the producer's queue has space.
        # Pre-condition: 'producer_id' is a valid, registered producer ID.
        if self.prod_list[producer_id] < self.prod_max_queue:
            self.prod_list[producer_id] += 1
            # Block Logic: Ensures atomic update of shop_items data structure.
            self.shop_items_lock.acquire()
            if product in self.shop_items.keys():
                self.shop_items_lock.release()
                # Block Logic: Acquires product-specific lock before modifying its producer list.
                self.shop_items[product][0].acquire()
                self.shop_items[product][1].append(producer_id)
                self.shop_items[product][0].release()
            else:
                # Inline: Initializes a new lock and producer list for a new product.
                self.shop_items[product] = (Lock(), [producer_id])
                self.shop_items_lock.release()
            return True
        return False

    def new_cart(self):
        """
        @brief Creates a new shopping cart and assigns it a unique ID.

        Returns:
            int: The unique ID of the newly created cart.
        """
        # Block Logic: Ensures exclusive access to the cart ID counter and cart dictionary.
        self.cart_id_lock.acquire()
        cart_id_var = self.cart_id
        self.carts[cart_id_var] = []
        self.cart_id += 1
        self.cart_id_lock.release()
        return cart_id_var

    def add_to_cart(self, cart_id, product) -> bool:
        """
        @brief Adds a product to a specific shopping cart.

        If the product is available, it is moved from the marketplace inventory
        to the cart. The producer's published count is decremented.

        Args:
            cart_id (int): The ID of the shopping cart.
            product (Product): The product to add.

        Returns:
            bool: True if the product was successfully added, False otherwise.
        """
        # Block Logic: Ensures atomic check for product existence in the marketplace.
        self.shop_items_lock.acquire()
        if product not in self.shop_items.keys():
            self.shop_items_lock.release()
            return False
        self.shop_items_lock.release()

        # Block Logic: Acquires product-specific lock to modify its available producers.
        self.shop_items[product][0].acquire()
        if len(self.shop_items[product][1]) > 0:
            prod_id = self.shop_items[product][1][0]
            self.shop_items[product][1].pop(0)
            self.shop_items[product][0].release()
            # Block Logic: Decrements the producer's count if a valid producer ID is found.
            if prod_id != -1:
                self.prod_list[prod_id] -= 1
            self.carts[cart_id].append(product)
            return True
        self.shop_items[product][0].release()
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a specific shopping cart.

        The removed product is not returned to the marketplace inventory but rather
        marked as available for re-publishing by any producer.

        Args:
            cart_id (int): The ID of the shopping cart.
            product (Product): The product to remove.
        """
        try:
            # Block Logic: Attempts to find and remove the product from the cart.
            product_index = self.carts[cart_id].index(product)
            self.carts[cart_id].pop(product_index)
            # Block Logic: Acquires product-specific lock to update its producer list.
            # Inline: Appends -1 to signify an item is removed from a cart, making it available
            # for re-publishing by a producer without directly re-adding to a specific producer's queue.
            self.shop_items[product][0].acquire()
            self.shop_items[product][1].append(-1)
            self.shop_items[product][0].release()
        except ValueError:
            # Inline: Handles cases where the product is not found in the cart.
            return

    def place_order(self, cart_id):
        """
        @brief Places an order for the items in a given cart.

        Returns the list of items in the cart, effectively finalizing the purchase.

        Args:
            cart_id (int): The ID of the shopping cart to place an order for.

        Returns:
            list: A list of products that were in the cart (the bought items).
        """
        cart = self.carts[cart_id]
        return cart


class Producer(Thread):
    """
    @brief Represents a producer thread that continuously supplies products to the marketplace.

    Inherits from `threading.Thread` and simulates a producer generating products
    and attempting to publish them to a shared marketplace, respecting its queue limits.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a Producer thread.

        Args:
            products (list): A list of tuples, each containing a product, quantity,
                             and production time for that product.
            marketplace (Marketplace): The shared marketplace instance to interact with.
            republish_wait_time (float): The time in seconds to wait before retrying
                                         to publish a product if the marketplace is full.
            **kwargs: Arbitrary keyword arguments (e.g., thread daemon status).
        """
        # Inline: Sets the thread as a daemon so it doesn't prevent program exit.
        Thread.__init__(self, daemon=True)
        self.production_list = products
        self.shop = marketplace
        # Functional Utility: Registers with the marketplace to get a unique producer ID.
        self.prod_id = marketplace.register_producer()
        self.shop_full_wait_time = republish_wait_time

    def run(self):
        """
        @brief Executes the producer's product generation and publishing logic.

        Continuously attempts to publish products from its production list to the marketplace,
        waiting if the marketplace's capacity for this producer is reached.
        """
        # Invariant: The producer continuously attempts to publish products.
        while True:
            # Pre-condition: 'self.production_list' contains products to be produced.
            # Invariant: Each product in the list will be attempted for production.
            for order in self.production_list:
                product = order[0]
                quantity = order[1]
                production_time = order[2]
                # Pre-condition: 'quantity' indicates remaining items to produce for the current order.
                # Invariant: The loop continues until the required quantity for the current order is produced.
                while quantity > 0:
                    # Block Logic: Attempts to publish the product to the marketplace.
                    if self.shop.publish(self.prod_id, product):
                        quantity -= 1
                        # Inline: Simulates the time taken to produce one unit of the product.
                        time.sleep(production_time)
                    else:
                        # Inline: Marketplace is full for this producer, wait before retrying.
                        time.sleep(self.shop_full_wait_time)


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Represents a generic product with a name and price.

    This is a frozen dataclass, meaning instances are immutable after creation.
    It defines custom hash and equality methods based solely on the product name.
    """
    name: str
    price: int

    def __hash__(self) -> int:
        """
        @brief Computes a hash for the Product based on its name.

        This allows Product objects to be used in hash-based collections (e.g., dict keys, set members).
        """
        return hash((self.name))

    def __eq__(self, other) -> bool:
        """
        @brief Compares two Product objects for equality based on their names.
        """
        # Pre-condition: 'other' is another object to compare against.
        # Invariant: Equality is determined by the 'name' attribute.
        return self.name == other.name


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Represents a specific type of Product: Tea.

    Extends the base Product with an additional attribute for 'type'.
    """
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Represents a specific type of Product: Coffee.

    Extends the base Product with additional attributes for 'acidity' and 'roast_level'.
    """
    acidity: str
    roast_level: str
