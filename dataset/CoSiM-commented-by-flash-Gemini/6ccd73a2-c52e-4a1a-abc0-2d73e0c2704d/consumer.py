
"""
@file consumer.py
@brief This module defines the components for a multi-threaded producer-consumer marketplace simulation.

It includes classes for:
- Consumer: Represents a buyer that interacts with the marketplace to add/remove products to/from a cart and place orders.
- Marketplace: Acts as the central hub where producers publish products and consumers manage their carts. It handles product availability and order placement.
- Producer: Represents a seller that continuously publishes products to the marketplace.
- Product (and its subclasses Tea, Coffee): Data structures to define the characteristics of products in the marketplace.

The simulation uses threading for concurrent producer and consumer operations, and locks within the Marketplace to ensure thread-safe access to shared resources like product queues and shopping carts.
"""

import time
from threading import Thread, Lock, currentThread
from dataclasses import dataclass


class Consumer(Thread):
    """
    @brief Represents a consumer thread in the marketplace simulation.

    Each consumer manages multiple shopping carts, adding and removing products,
    and ultimately placing orders. It retries operations if products are
    unavailable or operations fail.

    Attributes:
        carts (list): A list of shopping cart configurations, where each cart
                      is a list of product dictionaries with quantity and type.
        marketplace (Marketplace): A reference to the shared marketplace instance.
        retry_wait_time (float): The time to wait before retrying a failed operation.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a new Consumer thread.
        @param carts (list): List of carts, each containing products to add/remove.
        @param marketplace (Marketplace): The marketplace instance to interact with.
        @param retry_wait_time (float): Time in seconds to wait before retrying.
        @param kwargs: Additional keyword arguments to pass to the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        @brief The main execution loop for the Consumer thread.
        It iterates through its assigned carts, attempts to add/remove products,
        and places orders. Operations are retried if they fail due to product unavailability.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            for product in cart:
                i = 0
                # Block Logic: Repeatedly attempts to add/remove products to/from the cart.
                # Continues until the desired quantity of the product is processed.
                while i < product["quantity"]:
                    # Block Logic: Handles product removal from the cart.
                    if product["type"] == "remove":
                        res = self.marketplace.remove_from_cart(cart_id, product["product"])
                        # If removal is successful, increment counter. Otherwise, wait and retry.
                        if res == 1:
                            i += 1
                        else:
                            time.sleep(self.retry_wait_time)
                    # Block Logic: Handles product addition to the cart.
                    else:
                        res = self.marketplace.add_to_cart(cart_id, product["product"])
                        # If addition is successful, increment counter. Otherwise, wait and retry.
                        if res:
                            i += 1
                        else:
                            time.sleep(self.retry_wait_time)
            self.marketplace.place_order(cart_id)


class Marketplace:
    """
    @brief Manages products from producers and carts for consumers in a thread-safe manner.

    It acts as the central point for registering producers, publishing products,
    creating new carts, adding/removing products from carts, and placing orders.

    Attributes:
        max_queue_size (int): The maximum number of products a producer can have in its queue.
        producer_dictionary (dict): Stores products published by each producer, keyed by producer ID.
        current_producer_id (int): A counter for assigning unique producer IDs.
        all_carts (dict): Stores all active shopping carts, keyed by cart ID.
        add_lock (Lock): A lock to synchronize access during adding products to carts.
        remove_lock (Lock): A lock to synchronize access during removing products from carts.
        carts_lock (Lock): A lock to synchronize access during cart creation.
        register_lock (Lock): A lock to synchronize producer registration.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace.
        @param queue_size_per_producer (int): The maximum size of the product queue for each producer.
        """
        self.max_queue_size = queue_size_per_producer
        self.producer_dictionary = {}
        self.current_producer_id = -1
        self.all_carts = {}
        self.add_lock = Lock()
        self.remove_lock = Lock()
        self.carts_lock = Lock()
        self.register_lock = Lock()

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace, assigning a unique ID.
        @return (int): The unique ID assigned to the new producer.
        """
        self.register_lock.acquire()
        self.current_producer_id += 1
        self.register_lock.release()
        self.producer_dictionary[self.current_producer_id] = []
        return self.current_producer_id

    def publish(self, producer_id, product):
        """
        @brief Publishes a product from a producer to the marketplace.
        The product is added to the producer's queue if there is space.
        @param producer_id (int): The ID of the producer publishing the product.
        @param product (Product): The product to publish.
        @return (bool): True if the product was successfully published, False otherwise.
        """
        p_id = int(producer_id)

        # Block Logic: Checks if the producer's queue has reached its maximum size.
        if len(self.producer_dictionary[p_id]) >= self.max_queue_size:
            return False

        self.producer_dictionary[p_id].append(product)

        return True

    def new_cart(self):
        """
        @brief Creates a new shopping cart and assigns a unique cart ID.
        @return (int): The unique ID of the newly created cart.
        """
        self.carts_lock.acquire()
        cart_id = len(self.all_carts) + 1
        self.carts_lock.release()
        self.all_carts[cart_id] = []

        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a product to a specified shopping cart.
        It first checks the current producer's stock, then other producers' stock.
        Uses a lock to ensure thread-safe operations.
        @param cart_id (int): The ID of the cart to add the product to.
        @param product (Product): The product to add.
        @return (bool): True if the product was successfully added, False otherwise.
        """
        self.add_lock.acquire()
        ok_add = 0
        # Block Logic: Attempts to find the product in the current producer's queue first.
        if self.producer_dictionary[self.current_producer_id].count(product) > 0:
            self.producer_dictionary[self.current_producer_id].remove(product)
            self.all_carts[cart_id].append(product)
            ok_add = 1
        # Block Logic: If not found with current producer, searches other producers.
        else:
            for (_, queue) in self.producer_dictionary.items():
                if queue.count(product) > 0:
                    queue.remove(product)
                    self.all_carts[cart_id].append(product)
                    ok_add = 1
                    break
        self.add_lock.release()

        if ok_add == 0:
            return False
        return True

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a specified shopping cart and potentially returns it to a producer's stock.
        It attempts to return the product to the current producer first, then to other producers if space permits.
        Uses a lock to ensure thread-safe operations.
        @param cart_id (int): The ID of the cart to remove the product from.
        @param product (Product): The product to remove.
        @return (int): 1 if the product was successfully removed and returned to stock, 0 otherwise.
        """
        ok_remove = 0
        # Block Logic: Attempts to return the product to the current producer's queue.
        if len(self.producer_dictionary[self.current_producer_id]) < self.max_queue_size:
            self.producer_dictionary[self.current_producer_id].append(product)
            ok_remove = 1
        # Block Logic: If current producer's queue is full, attempts to return to other producers.
        else:
            for (_, queue) in self.producer_dictionary.items():
                if len(queue) < self.max_queue_size:
                    queue.append(product)
                    ok_remove = 1
                    break
        if ok_remove == 1:
            self.remove_lock.acquire()
            self.all_carts[cart_id].remove(product)
            self.remove_lock.release()
        return ok_remove

    def place_order(self, cart_id):
        """
        @brief Places an order for the items in a specified cart.
        This operation simulates the finalization of a purchase, printing details to stdout.
        @param cart_id (int): The ID of the cart for which to place the order.
        """
        for prod in self.all_carts[cart_id]:
            print(str(currentThread().getName()) + " bought " + str(prod))


class Producer(Thread):
    """
    @brief Represents a producer thread in the marketplace simulation.

    Each producer continuously generates and publishes products to the marketplace.
    It retries publishing if the marketplace queue is full.

    Attributes:
        products (list): A list of products the producer will generate, including
                         quantity and time to wait after publishing.
        marketplace (Marketplace): A reference to the shared marketplace instance.
        republish_wait_time (float): The time to wait before retrying to publish a product.
        product_id (int): The unique ID assigned to this producer by the marketplace.
    """
    
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a new Producer thread.
        @param products (list): List of products to be produced (name, quantity, wait time).
        @param marketplace (Marketplace): The marketplace instance to publish to.
        @param republish_wait_time (float): Time in seconds to wait before retrying to publish.
        @param kwargs: Additional keyword arguments to pass to the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.product_id = self.marketplace.register_producer()

    def run(self):
        """
        @brief The main execution loop for the Producer thread.
        It continuously attempts to publish its defined products to the marketplace.
        If publishing fails (e.g., marketplace queue is full), it waits and retries.
        """
        while True:
            # Block Logic: Iterates through each product type the producer is responsible for.
            for elem in self.products:
                curr_prod = 0
                # Block Logic: Attempts to publish the specified quantity of the current product.
                while curr_prod < elem[1]:
                    publish_ok = self.marketplace.publish(str(self.product_id), elem[0])
                    # If publishing is successful, wait for the defined time and increment.
                    # Otherwise, wait for republish_wait_time and retry.
                    if publish_ok:
                        time.sleep(elem[2])
                        curr_prod += 1
                    else:
                        time.sleep(self.republish_wait_time)


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Base data class for products in the marketplace.

    Uses `dataclass` for automatic generation of `__init__`, `__repr__`, etc.
    It is frozen to make product instances immutable and hashable, suitable for sets and dictionary keys.

    Attributes:
        name (str): The name of the product.
        price (int): The price of the product.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Represents a Tea product, inheriting from Product.

    Adds a specific attribute for the type of tea.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Represents a Coffee product, inheriting from Product.

    Adds specific attributes for the acidity and roast level of the coffee.
    """
    acidity: str
    roast_level: str
