

"""
This module simulates a multi-threaded marketplace environment. It defines the core
components for a product ecosystem, including Producers that publish items,
Consumers that interact with the marketplace by adding and removing items from carts,
and a central Marketplace coordinating these actions. It utilizes threading for
concurrent operations and synchronization primitives to manage shared resources
like product inventories and shopping carts.

Classes:
- `Consumer`: Represents a buyer that interacts with the marketplace.
- `Marketplace`: The central hub managing products, producers, and consumer carts.
- `Producer`: Represents a seller that publishes products to the marketplace.
- `Product`: Base dataclass for defining products.
- `Tea`: Dataclass for a specific type of product (inherits from Product).
- `Coffee`: Dataclass for another specific type of product (inherits from Product).
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    Represents a consumer (buyer) in the marketplace simulation. Each consumer
    runs as a separate thread, interacting with the `Marketplace` to create carts,
    add/remove products, and place orders. It simulates retry logic for adding
    products if they are not immediately available.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of cart configurations, where each configuration
                          is a list of dictionaries defining product operations.
                          Example: [[{'product': ProductA, 'quantity': 1, 'type': 'add'}]]
            marketplace (Marketplace): The marketplace instance to interact with.
            retry_wait_time (float): The time (in seconds) to wait before retrying
                                     an 'add' operation if the product is unavailable.
            **kwargs: Arbitrary keyword arguments passed to the Thread constructor,
                      e.g., 'name' for the consumer thread's name.
        """
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.consumer_name = kwargs["name"]

    def run(self):
        """
        The main execution method for the consumer thread.
        It iterates through its predefined carts, performs add/remove operations,
        and places orders.
        """
        # Block Logic: Process each cart defined for this consumer.
        for cart in self.carts:

            # Pre-condition: Create a new cart in the marketplace.
            cart_id = self.marketplace.new_cart()

            # Block Logic: Iterate through the product operations (add/remove) for the current cart.
            for data in cart:
                i = 0
                item = data["product"]
                operation = data["type"]

                # Block Logic: Perform the specified operation for the given quantity.
                # Invariant: 'i' counts the number of successfully processed items for the current operation.
                while i < data["quantity"]:

                    # Block Logic: Handle 'add' operation.
                    if operation == "add":
                        available = self.marketplace.add_to_cart(cart_id, item)
                        if available:
                            i += 1  # Successfully added, increment count.
                        else:
                            # Product not available, wait and retry.
                            time.sleep(self.retry_wait_time)

                    # Block Logic: Handle 'remove' operation.
                    # Pre-condition: Assume item is in cart to be removed.
                    if operation == "remove":
                        self.marketplace.remove_from_cart(cart_id, item)
                        i += 1  # Item removed, increment count.

            # Once all items for the current cart are processed, place the order.
            order = self.marketplace.place_order(cart_id)

            # Block Logic: Print the items successfully bought by this consumer.
            for item in order:
                print(self.consumer_name + " bought "+ str(item[0]))


from threading import Lock

class Marketplace:
    """
    The central marketplace component that manages producers, consumers,
    product inventory, and shopping carts. It provides thread-safe operations
    for registering producers, publishing products, creating carts,
    and managing items within those carts.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of items each producer
                                           can have in the marketplace's inventory queue.
        """
        self.queue_size_per_producer = queue_size_per_producer

        self.num_prod = 0  # Counter for assigning unique producer IDs.
        self.num_carts = 0 # Counter for assigning unique cart IDs.
        
        # List to keep track of the number of items each producer has currently published.
        self.prod_num_items = []
        # Dictionary to store published items, keyed by producer_id.
        # Each value is a list of products published by that producer.
        self.items = {}
        # Dictionary to store shopping carts, keyed by cart_id.
        # Each value is a list of (product, producer_id) tuples in that cart.
        self.carts = {}

        # Locks for ensuring thread safety during critical operations.
        self.register_lock = Lock()  # Protects producer registration.
        self.new_cart_lock = Lock()  # Protects new cart creation.
        self.cart_lock = Lock()      # Protects cart modification and item removal from inventory.


    def register_producer(self):
        """
        Registers a new producer with the marketplace, assigning it a unique ID.
        It also initializes tracking for the producer's published items.

        Returns:
            int: The unique ID assigned to the registered producer.
        """
        # Block Logic: Acquire lock to safely generate a new producer ID.
        with self.register_lock:
            prod_id = self.num_prod
            self.num_prod += 1

        self.prod_num_items.append(0)  # Initialize item count for the new producer.
        self.items[prod_id] = []       # Initialize product list for the new producer.
        return prod_id

    def publish(self, producer_id, product):
        """
        Publishes a product from a producer to the marketplace's inventory.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (Product): The product object to be published.

        Returns:
            bool: True if the product was successfully published (queue not full),
                  False otherwise.
        """
        # Pre-condition: Check if the producer's queue has reached its maximum size.
        if self.prod_num_items[producer_id] >= self.queue_size_per_producer:
            return False
        
        # Add the product to the producer's inventory and increment its item count.
        self.items[producer_id].append(product)
        self.prod_num_items[producer_id] += 1

        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart and assigns it a unique ID.

        Returns:
            int: The unique ID of the newly created cart.
        """
        # Block Logic: Acquire lock to safely generate a new cart ID.
        with self.new_cart_lock:
            cart_id = self.num_carts
            self.num_carts += 1

        self.carts[cart_id] = [] # Initialize an empty list for the new cart.

        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Attempts to add a product to a specified shopping cart.
        This involves finding the product in any producer's inventory,
        removing it from there, and adding it to the consumer's cart.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (Product): The product object to add.

        Returns:
            bool: True if the product was successfully added to the cart,
                  False if the product was not found in any producer's inventory.
        """
        found = False
        # Block Logic: Acquire lock to ensure atomic operations on inventory and carts.
        with self.cart_lock:
            # Block Logic: Iterate through all producers' inventories to find the product.
            for i in self.items:
                if product in self.items[i]:
                    # Product found: remove from producer's inventory and update counts.
                    self.items[i].remove(product)
                    self.prod_num_items[i] -= 1
                    prod_id = i
                    found = True
                    break

        if found:
            # If product was found and removed from inventory, add it to the cart.
            self.carts[cart_id].append((product, prod_id))

        return found

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a specified shopping cart and returns it to
        its original producer's inventory.

        Args:
            cart_id (int): The ID of the cart to remove the product from.
            product (Product): The product object to remove.
        """
        # Block Logic: Iterate through items in the specified cart to find the product.
        # This loop also retrieves the original producer ID associated with the item.
        for item, producer in self.carts[cart_id]:
            if item is product: # Functional Utility: Use 'is' for object identity check.
                prod_id = producer
                self.carts[cart_id].remove((item, producer))
                break

        # Return the product to its producer's inventory.
        self.items[prod_id].append(product)

        # Block Logic: Acquire lock to safely increment the producer's item count.
        with self.cart_lock:
            self.prod_num_items[prod_id] += 1

    def place_order(self, cart_id):
        """
        Finalizes an order by retrieving the contents of a specified cart
        and then removing the cart from the marketplace.

        Args:
            cart_id (int): The ID of the cart to place an order for.

        Returns:
            list: A list of (product, producer_id) tuples representing the items in the order.
        """
        # Pop the cart from the carts dictionary, effectively finalizing the order.
        res = self.carts.pop(cart_id)
        return res


from threading import Thread
import time

class Producer(Thread):
    """
    Represents a producer (seller) in the marketplace simulation. Each producer
    runs as a separate thread, registers itself with the `Marketplace`, and then
    continuously attempts to publish its predefined set of products. It includes
    retry logic if the marketplace's queue for its products is full.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of tuples, where each tuple contains
                             (product_item, quantity_to_publish, wait_time_between_pubs).
                             Example: [(ProductA, 5, 0.1), (ProductB, 3, 0.5)]
            marketplace (Marketplace): The marketplace instance to interact with.
            republish_wait_time (float): The time (in seconds) to wait before retrying
                                         to publish a product if the queue is full.
            **kwargs: Arbitrary keyword arguments passed to the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products

        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        The main execution method for the producer thread.
        It registers itself with the marketplace and then enters a continuous
        loop to publish its products according to their specified quantities and delays.
        """
        # Pre-condition: Register the producer with the marketplace to get a unique ID.
        prod_id = self.marketplace.register_producer()
        # Block Logic: Main loop for continuous product publishing.
        # Invariant: The producer attempts to publish all its defined products repeatedly.
        while True:
            # Block Logic: Iterate through each type of product this producer offers.
            for (item, quantity, wait_time) in self.products:
                i = 0
                # Block Logic: Attempt to publish the specified quantity of the current product.
                # Invariant: 'i' counts the number of successfully published items of the current product.
                while i < quantity:
                    available = self.marketplace.publish(prod_id, item)

                    if available:
                        # If publishing was successful, wait for the specified time before publishing the next item.
                        time.sleep(wait_time)
                        i += 1  # Increment count of successfully published items.
                    else:
                        # If the marketplace queue for this producer is full, wait and retry.
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Base dataclass representing a generic product in the marketplace.
    Attributes are frozen, making Product instances immutable.
    """
    name: str
    """The name of the product."""
    price: int
    """The price of the product."""


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Dataclass representing a Tea product, inheriting from `Product`.
    Adds specific attributes relevant to tea.
    """
    type: str
    """The type of tea (e.g., 'Green', 'Black', 'Herbal')."""


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Dataclass representing a Coffee product, inheriting from `Product`.
    Adds specific attributes relevant to coffee.
    """
    acidity: str
    """The acidity level of the coffee (e.g., 'Low', 'Medium', 'High')."""
    roast_level: str
    """The roast level of the coffee (e.g., 'Light', 'Medium', 'Dark')."""
