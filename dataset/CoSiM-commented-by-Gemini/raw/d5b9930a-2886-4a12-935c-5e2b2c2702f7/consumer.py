"""
This module implements a producer-consumer simulation for a marketplace,
using thread-safe queues for inventory but containing several complex and
potentially flawed logic patterns.
"""

from threading import Thread, Lock
from time import sleep
from queue import Queue, Full, Empty
from typing import Dict
from dataclasses import dataclass


class Consumer(Thread):
    """
    Represents a consumer that processes a list of carts, each containing
    a series of add/remove orders.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes the Consumer thread."""
        super().__init__(name=kwargs.get("name"))
        self.carts: list = carts
        self.marketplace = marketplace
        self.retry_time = retry_wait_time
        self.output_str = "%s bought %s"

    def run(self):
        """
        The main loop for the consumer. It processes each cart sequentially.
        """
        while len(self.carts) > 0:
            order = self.carts.pop(0)
            cart_id = self.marketplace.new_cart()

            # Process all requests (add/remove) for the current order.
            while len(order) > 0:
                request = order.pop(0)

                if request["type"] == "add":
                    added_products = 0
                    while added_products < request["quantity"]:
                        if self.marketplace.add_to_cart(cart_id, request["product"]):
                            added_products += 1
                        else:
                            # Wait and retry if the product cannot be added.
                            sleep(self.retry_time)
                
                elif request["type"] == "remove":
                    for _ in range(request["quantity"]):
                        self.marketplace.remove_from_cart(cart_id, request["product"])

            # Finalize the order and print the results.
            cart_items = self.marketplace.place_order(cart_id)
            for product in cart_items:
                print(self.output_str % (self.name, product))


class Marketplace:
    """
    Manages the inventory and carts.
    This implementation uses thread-safe Queues but has complex and potentially
    problematic logic for adding items to carts.
    """

    def __init__(self, queue_size_per_producer: int):
        self.register_lock = Lock()
        self.producers_no: int = 0
        self.queue_size = queue_size_per_producer
        self.producer_queues: Dict[int, Queue] = {}

        self.cart_lock = Lock()
        self.consumers_no: int = 0
        self.consumer_carts: Dict[int, list] = {}

        # Register a "default" producer 0, perhaps for returned items.
        self.register_producer(ignore_limit=True)

    def register_producer(self, ignore_limit: bool = False) -> int:
        """
        Registers a new producer and creates an inventory queue for them.
        NOTE: This method is not thread-safe. `self.producers_no` is incremented
        after the lock is released, which can lead to a race condition.
        """
        with self.register_lock:
            producer_id = self.producers_no
            if ignore_limit:
                self.producer_queues[producer_id] = Queue()
            else:
                self.producer_queues[producer_id] = Queue(self.queue_size)
        self.producers_no += 1
        return producer_id

    def publish(self, producer_id: int, product) -> bool:
        """
        Publishes a product to a specific producer's queue.
        This is a non-blocking operation.
        """
        try:
            self.producer_queues[producer_id].put_nowait(product)
        except Full:
            return False
        return True

    def new_cart(self) -> int:
        """
        Creates a new cart for a consumer.
        NOTE: This method is not thread-safe. `self.consumers_no` is incremented
        after the lock is released, creating a race condition.
        """
        with self.cart_lock:
            cart_id = self.consumers_no
            self.consumer_carts[cart_id] = []
        self.consumers_no += 1
        return cart_id

    def add_to_cart(self, cart_id: int, product) -> bool:
        """
        Adds a product to a cart by searching all producer queues.
        WARNING: This method uses a destructive read-and-replace pattern that
        is very inefficient and can lead to livelock if a producer queue is full.
        """
        cart = self.consumer_carts[cart_id]
        for producer_id in range(self.producers_no):
            try:
                # Destructively read from the producer's queue.
                queue_head = self.producer_queues[producer_id].get_nowait()

                if queue_head == product:
                    # Found the product, add it to the cart.
                    cart.append(queue_head)
                    return True

                # If not the right product, attempt to put it back.
                # This is a dangerous pattern.
                while True:
                    try:
                        self.producer_queues[producer_id].put_nowait(queue_head)
                        break
                    except Full:
                        # This will spinlock if the queue is full, preventing
                        # the producer from ever adding to it.
                        continue
            except Empty:
                # Continue to the next producer if this one is empty.
                continue
        return False

    def remove_from_cart(self, cart_id: int, product) -> None:
        """
        Removes a product from a cart and returns it to a default queue.
        NOTE: The item is always returned to producer 0, which may not be the
        intended behavior.
        """
        try:
            self.consumer_carts[cart_id].remove(product)
            # All removed items go to producer 0's queue.
            self.publish(0, product)
        except ValueError:
            pass

    def place_order(self, cart_id: int) -> list:
        """Returns the list of items in the cart."""
        return self.consumer_carts[cart_id]


class Producer(Thread):
    """
    Represents a producer thread that creates products and publishes them.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes the Producer thread."""
        super().__init__(name=kwargs.get("name"), daemon=kwargs.get("daemon", False))
        self.products = products
        self.marketplace = marketplace
        self.republish_time = republish_wait_time
        # Each producer registers once upon creation.
        self.producer_id = marketplace.register_producer()

    def run(self):
        """
        Main loop for the producer. It continuously produces its assigned products.
        """
        while True:
            for product_info in self.products:
                product_item, quantity, production_time = product_info
                produced = 0
                waited = False

                while produced < quantity:
                    if not waited:
                        sleep(production_time)

                    # Attempt to publish the product.
                    if self.marketplace.publish(self.producer_id, product_item):
                        produced += 1
                        waited = False
                    else:
                        # If the queue is full, wait and retry.
                        sleep(self.republish_time)
                        waited = True


@dataclass(init=True, repr=True, order=False, frozen=True, eq=True)
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
