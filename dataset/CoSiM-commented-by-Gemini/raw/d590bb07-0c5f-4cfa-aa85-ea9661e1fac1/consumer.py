"""
This module implements a producer-consumer simulation for a marketplace.

This version demonstrates a different, more complex, and potentially less safe
approach to handling concurrency and inventory compared to other examples.
"""

from threading import Thread, Lock
import time
from dataclasses import dataclass


class Consumer(Thread):
    """
    Represents a consumer thread that interacts with the marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer.

        Args:
            carts (list): A list of shopping operations.
            marketplace (Marketplace): The marketplace instance.
            retry_wait_time (float): Time to wait before retrying an action.
            **kwargs: Keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def consumer_add_to_cart(self, quantity, cart_id, product_id):
        """
        Tries to add a specified quantity of a product to a cart, retrying if it fails.

        Args:
            quantity (int): The number of products to add.
            cart_id (int): The ID of the cart.
            product_id (Product): The product to add.
        """
        counter = 0
        while counter < quantity:
            if not self.marketplace.add_to_cart(cart_id, product_id):
                time.sleep(self.retry_wait_time)
            else:
                counter += 1

    def run(self):
        """
        The main execution method for the consumer thread. It processes a series
        of add/remove operations and then places an order.
        """
        cart_id = self.marketplace.new_cart()

        for cart in self.carts:
            for entry in cart:
                if entry.get("type") == "remove":
                    for _ in range(entry.get("quantity")):
                        self.marketplace.remove_from_cart(cart_id, entry.get("product"))
                else:
                    self.consumer_add_to_cart(entry.get("quantity"), cart_id, entry.get("product"))

        # After all operations, print the final contents of the cart.
        for product in self.marketplace.place_order(cart_id):
            print(self.name, "bought", product)


class Marketplace:
    """
    Manages producers, consumers, and inventory.
    WARNING: This implementation has several potential race conditions and inefficiencies.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): Max items a producer can queue.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_id = -1
        self.producer_queue = {}
        self.producer_lock = Lock()
        self.consumer_id = -1
        self.consumer_queue = {}
        self.consumer_lock = Lock()

    def register_producer(self):
        """
        Registers a new producer.
        NOTE: This method is not thread-safe. Two producers calling this
        concurrently could receive the same ID.
        """
        self.producer_id += 1
        self.producer_queue[self.producer_id] = []
        return self.producer_id

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product.
        """
        if len(self.producer_queue.get(producer_id)) < self.queue_size_per_producer:
            with self.producer_lock:
                self.producer_queue.get(producer_id).append(product)
            return True
        return False

    def new_cart(self):
        """
        Creates a new cart for a consumer.
        NOTE: This method is not thread-safe and can lead to race conditions.
        """
        self.consumer_id += 1
        self.consumer_queue[self.consumer_id] = []
        return self.consumer_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a consumer's cart by searching through all producer queues.
        NOTE: This is inefficient and has potential concurrency issues.
        """
        for producer in self.producer_queue:
            for item in self.producer_queue.get(producer):
                if item == product:
                    with self.consumer_lock:
                        self.consumer_queue.get(cart_id).append([product, producer])
                        # This modification is not protected by the producer_lock.
                        self.producer_queue.get(producer).remove(product)
                    return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to the producer's queue.
        """
        for item, producer in self.consumer_queue.get(cart_id):
            if item == product:
                # This read/write is not protected by the consumer_lock.
                self.consumer_queue.get(cart_id).remove([product, producer])
                with self.producer_lock:
                    self.producer_queue.get(producer).append(product)
                break

    def place_order(self, cart_id):
        """
        Returns the final list of products in the cart.
        """
        return [product for product, _ in self.consumer_queue.get(cart_id)]


class Producer(Thread):
    """
    Represents a producer thread that creates and publishes products.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs

    def produce(self, product, quantity, produce_time, producer_id):
        """
        Produces a given quantity of a product.

        Args:
            product (Product): The product to produce.
            quantity (int): The number of items to produce.
            produce_time (float): Time taken to produce one item.
            producer_id (int): The ID of the producer.
        """
        counter = 0
        while counter < quantity:
            if not self.marketplace.publish(producer_id, product):
                time.sleep(self.republish_wait_time)
            else:
                time.sleep(produce_time)
                counter += 1

    def run(self):
        """
        The main loop for the producer.
        NOTE: This loop continuously re-registers the same producer, which is likely a bug.
        """
        while True:
            producer_id = self.marketplace.register_producer()
            for product_info in self.products:
                self.produce(product_info[0], product_info[1], product_info[2], producer_id)


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a product."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass for Tea."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass for Coffee."""
    acidity: str
    roast_level: str
