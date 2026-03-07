"""
@file consumer.py
@brief Implements a simulated e-commerce marketplace system.
This module defines classes for Producers (threads that supply products),
Consumers (threads that buy products), and a central Marketplace (managing inventory,
carts, and orders). It utilizes multi-threading and locking mechanisms to simulate
concurrent interactions in a marketplace environment.
"""

from threading import Thread
import time


class Consumer(Thread):
    """
    @brief Represents a consumer thread in the marketplace simulation.
    Consumers create carts, add/remove products, and place orders.
    They retry adding products if the marketplace is temporarily unable to fulfill the request.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a new Consumer thread.

        @param carts (list): A list of shopping carts, each containing a sequence of operations (add/remove product).
        @param marketplace (Marketplace): A reference to the shared Marketplace instance.
        @param retry_wait_time (float): The time (in seconds) to wait before retrying an add operation.
        @param kwargs: Additional keyword arguments passed to the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        @brief The main execution loop for the Consumer thread.
        Iterates through its assigned carts, performs add/remove operations, and places orders.
        """
        # Block Logic: Iterates through each shopping cart assigned to this consumer.
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()  # Creates a new shopping cart in the marketplace.

            # Block Logic: Processes each operation within the current cart.
            for operation in cart:
                quantity = operation["quantity"]
                # Conditional Logic: Handles 'add' operations.
                if operation["type"] == "add":
                    # Block Logic: Attempts to add the product 'quantity' times.
                    for _ in range(quantity):
                        # Loop until the product is successfully added to the cart.
                        # `time.sleep` simulates a delay before retrying if the add operation fails.
                        while self.marketplace.add_to_cart(cart_id, operation["product"]) is False:
                            time.sleep(self.retry_wait_time)

                # Conditional Logic: Handles 'remove' operations.
                if operation["type"] == "remove":
                    # Block Logic: Attempts to remove the product 'quantity' times.
                    for _ in range(quantity):
                        self.marketplace.remove_from_cart(cart_id, operation["product"])

            self.marketplace.place_order(cart_id)  # Places the order for the completed cart.


from threading import Lock, currentThread

class Marketplace:
    """
    @brief Manages the central logic of the e-commerce marketplace.
    It handles product inventory, producer registration, cart creation,
    adding/removing items from carts, and placing orders.
    Uses a threading.Lock for critical section protection.
    """

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace.

        @param queue_size_per_producer (int): The maximum number of items a producer can have
                                             waiting in the marketplace at any given time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.lock = Lock()  # Global lock to protect shared marketplace data.
        self.cid = 0  # Counter for generating unique cart IDs.
        self.producer_items = []  # List tracking item counts for each producer.
        self.products = []  # List of available products in the marketplace.
        self.carts = {}  # Dictionary storing carts, keyed by cart_id.
        self.producers = {}  # Dictionary mapping products to their producer IDs.

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace.

        @return int: A unique producer ID.
        """
        self.lock.acquire()  # Acquires lock to protect shared data.
        prod_id = len(self.producer_items)  # Assigns a new unique producer ID.
        self.producer_items.append(0)  # Initializes item count for the new producer.
        self.lock.release()  # Releases the lock.
        return prod_id

    def publish(self, producer_id, product):
        """
        @brief Publishes a product to the marketplace by a producer.

        @param producer_id (int/str): The ID of the producer publishing the product.
        @param product (Product): The product to publish.
        @return bool: True if the product was successfully published, False otherwise (e.g., queue full).
        """
        producer_id = int(producer_id)  # Ensures producer_id is an integer.

        # Conditional Logic: Checks if the producer has exceeded its queue size limit.
        if self.producer_items[producer_id] >= self.queue_size_per_producer:
            return False  # Publication failed due to queue size limit.

        self.producer_items[producer_id] += 1  # Increments producer's item count.
        self.products.append(product)  # Adds the product to the marketplace's available products.
        self.producers[product] = producer_id  # Associates the product with its producer.

        return True  # Publication successful.

    def new_cart(self):
        """
        @brief Creates a new empty shopping cart in the marketplace.

        @return int: A unique cart ID.
        """
        self.lock.acquire()  # Acquires lock to protect shared data.
        self.cid += 1  # Increments cart ID counter.
        cart_id = self.cid  # Assigns new unique cart ID.
        self.lock.release()  # Releases the lock.

        self.carts[cart_id] = []  # Initializes an empty list for the new cart.
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a product to a specific shopping cart.

        @param cart_id (int): The ID of the cart to add to.
        @param product (Product): The product to add.
        @return bool: True if the product was successfully added, False if the product is not available.
        """
        self.lock.acquire()  # Acquires lock to protect shared data.
        # Conditional Logic: Checks if the product is currently available in the marketplace.
        if product not in self.products:
            self.lock.release()  # Releases the lock before returning.
            return False  # Product not available.

        # Decrements the item count for the producer of this product.
        self.producer_items[self.producers[product]] -= 1
        # Removes the product from the marketplace's available products list.
        self.products.remove(product)

        self.carts[cart_id].append(product)  # Adds the product to the specified cart.
        self.lock.release()  # Releases the lock.

        return True  # Product successfully added.

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a specific shopping cart and returns it to the marketplace.

        @param cart_id (int): The ID of the cart to remove from.
        @param product (Product): The product to remove.
        """
        self.carts[cart_id].remove(product)  # Removes product from the cart.
        self.products.append(product)  # Returns product to the marketplace's available products.

        self.lock.acquire()  # Acquires lock to protect shared data.
        # Increments the item count for the producer of this product.
        self.producer_items[self.producers[product]] += 1
        self.lock.release()  # Releases the lock.

    def place_order(self, cart_id):
        """
        @brief Places an order for a given cart, printing the transaction details.

        @param cart_id (int): The ID of the cart to place an order for.
        @return list: The list of products in the placed order.
        """
        products_list = self.carts.get(cart_id)  # Retrieves the list of products in the cart.
        # Block Logic: Iterates through each product in the order and prints a purchase message.
        for product in products_list:
            self.lock.acquire()  # Acquires lock to ensure atomic print operation.
            print("{} bought {}".format(currentThread().getName(), product))
            self.lock.release()  # Releases the lock.

        return products_list


from threading import Thread
import time


class Producer(Thread):
    """
    @brief Represents a producer thread in the marketplace simulation.
    Producers continuously publish products to the marketplace.
    They retry publishing if the marketplace is temporarily unable to accept the product.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a new Producer thread.

        @param products (list): A list of (product, quantity, wait_time) tuples
                                representing the products this producer offers.
        @param marketplace (Marketplace): A reference to the shared Marketplace instance.
        @param republish_wait_time (float): The time (in seconds) to wait before retrying a publish operation.
        @param kwargs: Additional keyword arguments passed to the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.prod_id = self.marketplace.register_producer()  # Registers itself with the marketplace.

    def run(self):
        """
        @brief The main execution loop for the Producer thread.
        Continuously attempts to publish its products to the marketplace.
        """
        while True:
            # Block Logic: Iterates through each product type this producer offers.
            for product, quantity, wait_time in self.products:
                # Block Logic: Attempts to publish each product 'quantity' times.
                for _ in range(quantity):
                    # Loop until the product is successfully published.
                    # `time.sleep` simulates a delay before retrying if the publish operation fails.
                    while self.marketplace.publish(str(self.prod_id), product) is False:
                        time.sleep(self.republish_wait_time)

                    time.sleep(wait_time)  # Simulates a delay between publishing individual items.


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Base class for products in the marketplace.
    Uses `dataclasses.dataclass` for concise definition.
    `frozen=True` makes instances immutable.
    """
    name: str  # Name of the product.
    price: int # Price of the product.

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Represents a Tea product, inheriting from Product.
    Adds a specific attribute for tea.
    """
    type: str  # Type of tea (e.g., "Green", "Black").

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Represents a Coffee product, inheriting from Product.
    Adds specific attributes for coffee.
    """
    acidity: str      # Acidity level of the coffee.
    roast_level: str  # Roast level of the coffee.
