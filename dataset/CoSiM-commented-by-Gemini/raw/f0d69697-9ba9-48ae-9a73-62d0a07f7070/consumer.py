"""
This module simulates a marketplace with producers and consumers.
This version uses a centralized locking strategy in the Marketplace to manage
product inventory and producer queues.
"""
from threading import Thread
import time

class Consumer(Thread):
    """
    Represents a consumer that buys products from the marketplace.
    """
    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes the consumer."""
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        Main logic for the consumer. It processes a list of shopping carts,
        executing add/remove operations for each product.
        """
        num_of_carts = len(self.carts)
        my_carts = []

        for _ in range(0, num_of_carts):
            new_cart_id = self.marketplace.new_cart()
            my_carts.append(new_cart_id)

        for current_cart in self.carts:
            current_cart_id = my_carts.pop(0)

            for current_operation in current_cart:
                desired_quantity = current_operation["quantity"]
                current_quantity = 0

                while current_quantity < desired_quantity:
                    current_operation_type = current_operation["type"]
                    current_operation_product = current_operation["product"]

                    if current_operation_type == "add":
                        current_operation_status = self.marketplace\
                            .add_to_cart(current_cart_id, current_operation_product)
                    else:
                        current_operation_status = self.marketplace \
                            .remove_from_cart(current_cart_id, current_operation_product)

                    if current_operation_status is True or current_operation_status is None:
                        current_quantity = current_quantity + 1
                    else:
                        # If an operation fails (e.g., product not available), wait and retry.
                        time.sleep(self.retry_wait_time)

            bought_products = self.marketplace.place_order(current_cart_id)
            for bought_product in bought_products:
                print(self.kwargs["name"] + " bought " + str(bought_product))

from threading import Lock

class Marketplace:
    """
    The central marketplace, managing all interactions between producers and consumers.
    """
    def __init__(self, queue_size_per_producer):
        """Initializes the marketplace."""
        # Maps a product to the ID of the producer who made it.
        self.producer_of_product = {}

        # A dictionary to track the number of products each producer has in the market.
        self.queue_size_of_producer = {}
        # A single lock to protect shared data structures related to products and producers.
        self.queue_size_of_producer_lock = Lock()

        self.queue_size_per_producer = queue_size_per_producer

        self.carts = {}
        self.carts_lock = Lock()

        # A global list of all available products in the marketplace.
        self.all_products = []

    def register_producer(self):
        """Registers a new producer, returning a unique ID."""
        with self.queue_size_of_producer_lock:
            current_producers_number = len(self.queue_size_of_producer)
            self.queue_size_of_producer[current_producers_number] = 0
        return current_producers_number

    def publish(self, producer_id, product):
        """
        Publishes a product to the marketplace from a given producer.
        Returns False if the producer's queue is full.
        """
        if self.queue_size_of_producer[producer_id] >= self.queue_size_per_producer:
            return False

        with self.queue_size_of_producer_lock:
            self.queue_size_of_producer[producer_id] += 1
            self.producer_of_product[product] = producer_id
            self.all_products.append(product)
        return True

    def new_cart(self):
        """Creates a new shopping cart and returns its ID."""
        with self.carts_lock:
            current_carts_number = len(self.carts)
            self.carts[current_carts_number] = []
        return current_carts_number

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart if it's available in the marketplace.
        This operation is protected by the global producer lock.
        """
        with self.queue_size_of_producer_lock:
            if product not in self.all_products:
                return False

            producer_id = self.producer_of_product[product]
            self.queue_size_of_producer[producer_id] -= 1
            self.all_products.remove(product)

        self.carts[cart_id].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a cart and returns it to the marketplace."""
        self.carts[cart_id].remove(product)
        self.all_products.append(product)
        producer_id = self.producer_of_product[product]

        with self.queue_size_of_producer_lock:
            self.queue_size_of_producer[producer_id] += 1

    def place_order(self, cart_id):
        """Finalizes the order and returns the products from the cart."""
        bought_products = self.carts[cart_id]
        self.carts[cart_id] = []
        return bought_products


class Producer(Thread):
    """Represents a producer that creates and publishes products."""
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes the producer."""
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        Main loop for the producer. It continuously tries to publish its products.
        """
        while True:
            for current_product in self.products:
                current_product_type = current_product[0]
                current_product_quantity_desired = current_product[1]
                current_product_quantity = 0
                current_product_time_to_create = current_product[2]

                while current_product_quantity < current_product_quantity_desired:
                    current_transaction_status = self.marketplace\
                        .publish(self.producer_id, current_product_type)

                    if current_transaction_status is True:
                        time.sleep(current_product_time_to_create)
                        current_product_quantity += 1
                    else:
                        # Wait and retry if publishing fails (e.g., queue is full).
                        time.sleep(self.republish_wait_time)
