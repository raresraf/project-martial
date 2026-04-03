"""
@893c6660-fab2-4972-ab08-bc6fb44c8d6a/consumer.py
@brief This script simulates a marketplace with producers and consumers.
It defines classes for managing products, carts, and the marketplace logic,
including thread-safe operations.
Domain: Concurrency, Object-Oriented Programming, Simulation.
"""

from threading import Thread, Lock
from time import sleep
from queue import Full
from dataclasses import dataclass


class Consumer(Thread):
    """
    @brief Represents a consumer agent in the marketplace simulation.
    Consumers create carts, add/remove products, and place orders.
    Algorithm: Iterative cart operation and order placement.
    Time Complexity: Depends on the number of carts and operations per cart.
    Space Complexity: O(1) for consumer state, O(N) for cart contents within the marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a Consumer thread.
        @param carts: A list of cart operations (add/remove product, quantity).
        @param marketplace: The marketplace instance to interact with.
        @param retry_wait_time: Time to wait before retrying an operation.
        @param kwargs: Additional keyword arguments, including 'name' for the consumer.
        """
        Thread.__init__(self)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs["name"]

    def run(self):
        """
        @brief The main execution method for the consumer thread.
        It iterates through predefined cart operations and places an order.
        """
        # Block Logic: Process each predefined cart for the consumer.
        # Invariant: Each cart is processed sequentially.
        for cart in self.carts:
            # Pre-condition: A new cart is created in the marketplace.
            cart_id = self.marketplace.new_cart()

            # Block Logic: Execute each operation within the current cart.
            # Invariant: Operations are processed as defined in the 'cart' list.
            for operation in cart:
                # Block Logic: Perform the specified quantity of add or remove operations.
                # Invariant: The loop runs 'quantity' times for each operation.
                for _ in range(operation["quantity"]):
                    # Block Logic: Handle 'add' operations.
                    # Pre-condition: Product must be available in the marketplace.
                    # Invariant: Retries adding the product until successful.
                    if operation["type"] == "add":
                        while not self.marketplace.add_to_cart(cart_id, operation["product"]):
                            sleep(self.retry_wait_time)
                    # Block Logic: Handle 'remove' operations.
                    # Pre-condition: Product must be in the cart to be removed.
                    elif operation["type"] == "remove":
                        self.marketplace.remove_from_cart(cart_id, operation["product"])

            # Post-condition: All products in the cart are ordered.
            products = self.marketplace.place_order(cart_id)

            # Block Logic: Print the details of the purchased products.
            # Invariant: Each product successfully placed in the order is reported.
            for product in products:
                print("{0} bought {1}".format(self.name, product))


class SafeList:
    """
    @brief A thread-safe list implementation, optionally with a maximum size.
    It uses a mutex to protect access to the underlying list.
    Domain: Concurrency, Data Structures.
    """

    def __init__(self, maxsize=0):
        """
        @brief Initializes a SafeList instance.
        @param maxsize: The maximum allowed size for the list (0 for no limit).
        """
        self.mutex = Lock()
        self.list = []
        self.maxsize = maxsize

    def put(self, item):
        """
        @brief Adds an item to the list if it's not full.
        @param item: The item to add.
        @raises Full: If the list has a maxsize and is already full.
        """
        # Block Logic: Acquire mutex to ensure exclusive access to the list.
        with self.mutex:
            # Pre-condition: Check if the list has a size limit and is currently full.
            if self.maxsize != 0 and self.maxsize == len(self.list):
                raise Full
            self.list.append(item)

    def put_anyway(self, item):
        """
        @brief Adds an item to the list without checking for maxsize.
        @param item: The item to add.
        """
        # Block Logic: Acquire mutex to ensure exclusive access to the list.
        with self.mutex:
            self.list.append(item)

    def remove(self, item):
        """
        @brief Removes the first occurrence of an item from the list.
        @param item: The item to remove.
        @return: True if the item was removed, False otherwise.
        """
        # Block Logic: Acquire mutex to ensure exclusive access to the list.
        with self.mutex:
            # Pre-condition: Check if the item exists in the list.
            if item not in self.list:
                return False
            self.list.remove(item)
            return True


class Cart:
    """
    @brief Represents a shopping cart, holding products and their associated producer IDs.
    """

    def __init__(self):
        """
        @brief Initializes an empty Cart.
        """
        self.products = []

    def add_product(self, product, producer_id):
        """
        @brief Adds a product to the cart with its originating producer ID.
        @param product: The product to add.
        @param producer_id: The ID of the producer who supplied the product.
        """
        self.products.append({"product": product, "producer_id": producer_id})

    def remove_product(self, product):
        """
        @brief Removes a specified product from the cart and returns its producer ID.
        @param product: The product to remove.
        @return: The producer ID of the removed product, or None if not found.
        """
        # Block Logic: Iterate through products to find and remove the target product.
        # Invariant: Only the first match for the product is removed.
        for item in self.products:
            if item["product"] == product:
                self.products.remove(item)
                return item["producer_id"]
        return None

    def get_products(self):
        """
        @brief Returns an iterator for all products in the cart (without producer IDs).
        @return: An iterator over product names.
        """
        return map(lambda item: item["product"], self.products)


class Marketplace:
    """
    @brief Manages producers, product queues, and consumer carts in a thread-safe manner.
    It acts as the central hub for all product and order transactions.
    Domain: Concurrency, Resource Management, Producer-Consumer Pattern.
    """

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace.
        @param queue_size_per_producer: The maximum number of products a producer can have in its queue.
        """
        self.queue_size_per_producer = queue_size_per_producer

        self.producer_queues = {}
        self.producer_id_generator = 0
        self.producer_id_generator_lock = Lock()

        self.carts = {}
        self.cart_id_generator = 0
        self.cart_id_generator_lock = Lock()

    def register_producer(self):
        """
        @brief Registers a new producer and assigns it a unique ID and a product queue.
        @return: The unique ID assigned to the new producer.
        """
        # Block Logic: Acquire lock to safely generate a new producer ID and create a queue.
        with self.producer_id_generator_lock:
            current_prod_id = self.producer_id_generator
            self.producer_queues[current_prod_id] = SafeList(maxsize=self.queue_size_per_producer)

            self.producer_id_generator += 1
            return current_prod_id

    def publish(self, producer_id, product):
        """
        @brief Attempts to publish a product from a producer to its queue.
        @param producer_id: The ID of the producer publishing the product.
        @param product: The product to publish.
        @return: True if the product was published, False if the queue is full.
        """
        try:
            # Pre-condition: The producer's queue must not be full.
            self.producer_queues[producer_id].put(product)
            return True
        except Full:
            # Post-condition: If queue is full, product is not published.
            return False

    def new_cart(self):
        """
        @brief Creates a new shopping cart and assigns it a unique ID.
        @return: The unique ID assigned to the new cart.
        """
        # Block Logic: Acquire lock to safely generate a new cart ID and create a cart.
        with self.cart_id_generator_lock:
            current_cart_id = self.cart_id_generator
            self.carts[current_cart_id] = Cart()

            self.cart_id_generator += 1
            return current_cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a product to a specified cart by taking it from a producer's queue.
        @param cart_id: The ID of the cart to add the product to.
        @param product: The product to add.
        @return: True if the product was successfully added, False otherwise.
        """
        producers_num = 0
        # Inline: Acquire lock to get the current number of registered producers safely.
        with self.producer_id_generator_lock:
            producers_num = self.producer_id_generator

        # Block Logic: Iterate through all producer queues to find the product.
        # Invariant: The loop continues until the product is found and added or all queues are checked.
        for producer_id in range(producers_num):
            # Pre-condition: The product must be available in a producer's queue.
            if self.producer_queues[producer_id].remove(product):
                self.carts[cart_id].add_product(product, producer_id)
                return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a specified cart and returns it to its originating producer's queue.
        @param cart_id: The ID of the cart to remove the product from.
        @param product: The product to remove.
        """
        # Pre-condition: The product must exist in the cart.
        producer_id = self.carts[cart_id].remove_product(product)

        # Post-condition: The product is returned to the producer's queue.
        # Inline: Products removed from a cart are returned to the producer's queue without size constraints.
        self.producer_queues[producer_id].put_anyway(product)

    def place_order(self, cart_id):
        """
        @brief Retrieves all products from a specified cart, effectively placing an order.
        @param cart_id: The ID of the cart to place the order from.
        @return: A list of products from the cart.
        """
        return self.carts[cart_id].get_products()


class Producer(Thread):
    """
    @brief Represents a producer agent in the marketplace simulation.
    Producers continuously produce and publish products to the marketplace.
    Algorithm: Continuous production and publishing with retry mechanism.
    Time Complexity: Runs indefinitely, production time depends on 'production_time' and 'republish_wait_time'.
    Space Complexity: O(1) for producer state, O(N) for product list.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a Producer thread.
        @param products: A list of tuples, each containing (product, quantity, production_time).
        @param marketplace: The marketplace instance to interact with.
        @param republish_wait_time: Time to wait before retrying publishing a product.
        @param kwargs: Additional keyword arguments, including 'daemon' and 'name'.
        """
        # Inline: Initialize the Thread parent class, setting daemon status.
        Thread.__init__(self, daemon=kwargs["daemon"])
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.name = kwargs["name"]

    def run(self):
        """
        @brief The main execution method for the producer thread.
        It registers with the marketplace and continuously publishes products.
        """
        # Pre-condition: Producer registers itself with the marketplace to get a unique ID.
        producer_id = self.marketplace.register_producer()

        # Block Logic: Main production loop, runs indefinitely.
        # Invariant: Producer continuously attempts to produce and publish products.
        while True:
            # Block Logic: Iterate through the predefined list of products to produce.
            # Invariant: Each product is produced for its specified quantity after a production delay.
            for (product, quantity, production_time) in self.products:
                # Pre-condition: Wait for the specified production time for the current product.
                sleep(production_time)

                # Block Logic: Attempt to publish the product 'quantity' times.
                # Invariant: Each product instance is published individually.
                for _ in range(quantity):
                    # Block Logic: Publish the product, retrying if the marketplace queue is full.
                    # Pre-condition: The producer's queue in the marketplace must not be full.
                    # Invariant: Retries publishing until successful.
                    while not self.marketplace.publish(producer_id, product):
                        sleep(self.republish_wait_time)


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Base class for products in the marketplace.
    Uses dataclass for automatic __init__, __repr__, etc.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Represents a Tea product, inheriting from Product.
    Adds a 'type' attribute specific to tea.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Represents a Coffee product, inheriting from Product.
    Adds 'acidity' and 'roast_level' attributes specific to coffee.
    """
    acidity: str
    roast_level: str
