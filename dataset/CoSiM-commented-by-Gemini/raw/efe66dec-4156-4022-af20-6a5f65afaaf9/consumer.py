"""
This module simulates a marketplace with producers and consumers.
Producers create products and publish them to the marketplace.
Consumers add products to their carts and place orders.
The simulation is multi-threaded to handle concurrent producers and consumers.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    Represents a consumer that interacts with the marketplace.
    Each consumer runs in its own thread, processing a list of shopping carts.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.
        Args:
            carts: A list of carts, where each cart is a list of shopping events.
            marketplace: The Marketplace object to interact with.
            retry_wait_time: Time to wait before retrying to add a product.
            **kwargs: Arguments for the Thread base class.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.consumer_name = kwargs["name"]

    def run(self):
        """
        The main logic for the consumer. It processes each cart by adding and
        removing products and finally placing an order.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()

            for event in cart:
                count = 0
                while count < event["quantity"]:
                    if event["type"] == "add":
                        # Retry adding to cart if the product is not available.
                        if self.marketplace.add_to_cart(cart_id, event["product"]):
                            count += 1
                        else:
                            time.sleep(self.retry_wait_time)

                    if event["type"] == "remove":
                        self.marketplace.remove_from_cart(cart_id, event["product"])
                        count += 1
            products_list = self.marketplace.place_order(cart_id)

            for product in products_list:
                print(self.consumer_name + " bought " + str(product))

from threading import Lock

class Marketplace:
    """
    The central marketplace where producers and consumers interact.
    It manages producer registrations, product queues, and customer carts.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.
        Args:
            queue_size_per_producer: The maximum number of products a producer
                                     can have in their queue.
        """
        self.register_lock = Lock()
        self.add_remove_lock = Lock()
        self.cart_lock = Lock()

        self.queue_capacity = queue_size_per_producer
        self.nr_producers = 0
        self.nr_carts = 0
        self.producer_queues = {}
        self.carts = []

    def register_producer(self):
        """
        Registers a new producer with the marketplace.
        Returns a unique producer ID. This method is thread-safe.
        """
        with self.register_lock:
            self.nr_producers += 1
            producer_id = "prod" + str(self.nr_producers)
            self.producer_queues[producer_id] = []
        return producer_id

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product.
        The product is added to the producer's queue if there is space.
        Returns True on success, False otherwise.
        """
        if len(self.producer_queues[producer_id]) < self.queue_capacity:
            self.producer_queues[producer_id].append(product)
            return True
        return False

    def new_cart(self):
        """
        Creates a new empty cart for a consumer.
        Returns a unique cart ID. This method is thread-safe.
        """
        with self.cart_lock:
            cart_id = self.nr_carts
            self.carts.append([])
            self.nr_carts += 1
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a consumer's cart.
        It finds the product in any producer's queue, removes it, and adds it to the cart.
        """
        for (producer_id, products_queue) in self.producer_queues.items():
            if product in products_queue:
                products_queue.remove(product)
                self.carts[cart_id].append((product, producer_id))
                return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to the original producer's queue.
        This method is thread-safe.
        """
        with self.add_remove_lock:
            index = 0
            for (cart_product, producer_id) in self.carts[cart_id]:
                if cart_product == product:
                    self.producer_queues[producer_id].append(product)
                    break
                index += 1
        self.carts[cart_id].pop(index)

    def place_order(self, cart_id):
        """
        Finalizes an order and returns the list of products in the cart.
        """
        return [elem[0] for elem in self.carts[cart_id]]


from threading import Thread
import time


class Producer(Thread):
    """
    Represents a producer that creates and publishes products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.
        Args:
            products: A list of products to be published by the producer.
            marketplace: The Marketplace object.
            republish_wait_time: Time to wait before retrying to publish.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        The main logic for the producer. It continuously tries to publish its
        products to the marketplace.
        """
        while True:
            for product in self.products:
                count = 0
                while count < product[1]:
                    if self.marketplace.publish(self.producer_id, product[0]):
                        time.sleep(product[2])
                        count += 1
                    else:
                        # Wait before retrying if the producer's queue is full.
                        time.sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base class for products using dataclasses for convenience."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A specific type of Product: Tea."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A specific type of Product: Coffee."""
    acidity: str
    roast_level: str
