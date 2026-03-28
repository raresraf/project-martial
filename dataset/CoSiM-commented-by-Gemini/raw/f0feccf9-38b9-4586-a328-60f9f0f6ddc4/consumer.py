"""
This module simulates a marketplace with producers and consumers, using a
multi-semaphore approach for synchronization and random IDs for entities.
"""
from threading import Thread, Semaphore
import time
import random

class Consumer(Thread):
    """
    Represents a consumer that interacts with the marketplace by processing
    a list of shopping carts.
    """
    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes the consumer."""
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        Main logic for the consumer. It creates a cart and processes a list
        of add/remove operations.
        """
        customer_id = self.marketplace.new_cart()
        for cart in self.carts:
            for action in cart:
                if action["type"] == "add":
                    i = 0
                    while i < action["quantity"]:
                        if self.marketplace.add_to_cart(customer_id, action["product"]):
                            i += 1
                        else:
                            time.sleep(self.retry_wait_time)
                else:
                    for i in range(action["quantity"]):
                        self.marketplace.remove_from_cart(customer_id, action["product"])
        order = self.marketplace.place_order(customer_id)
        for product in order:
            print(self.name, "bought", product)

class Marketplace:
    """
    The central marketplace that manages producers, consumers, and products.
    This implementation uses multiple semaphores for synchronization.
    """
    def __init__(self, queue_size_per_producer):
        """Initializes the marketplace."""
        self.queue_size_per_producer = queue_size_per_producer
        self.producers = []
        self.customers = []
        self.carts = {}
        self.products = {}
        # Multiple semaphores are used for locking different operations.
        # This is an unconventional design that can be hard to manage.
        self.sem = Semaphore(1)
        self.sem1 = Semaphore(1)
        self.sem2 = Semaphore(1)
        self.sem3 = Semaphore(1)
        self.sem4 = Semaphore(1)

    def register_producer(self):
        """
        Registers a new producer with a random ID.
        """
        while True:
            rand = random.randint(0, 5000)
            if rand not in self.producers:
                self.producers.append(rand)
                self.products[rand] = []
                return rand

    def publish(self, producer_id, product):
        """
        Publishes a product to the marketplace. Protected by `self.sem`.
        """
        self.sem.acquire()
        length = len(self.products[producer_id])
        if length < self.queue_size_per_producer:
            self.products[producer_id].append(product)
            self.sem.release()
            return True
        self.sem.release()
        return False

    def new_cart(self):
        """
        Creates a new cart with a random ID.
        """
        while True:
            rand = random.randint(0, 5000)
            if rand not in self.producers:
                self.carts[rand] = []
                return rand

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart. Protected by `self.sem1`.
        """
        self.sem1.acquire()
        for producer in self.producers:
            products = self.products[producer]
            if product in products:
                self.carts[cart_id].append((product, producer))
                self.products[producer].remove(product)
                self.sem1.release()
                return True
        self.sem1.release()
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart. Protected by `self.sem3`.
        """
        self.sem3.acquire()
        producer_id = -1
        for aux in self.carts[cart_id]:
            if aux[0] == product:
                producer_id = aux[1]
                self.carts[cart_id].remove((product, producer_id))
                self.products[producer_id].append(product)
                self.sem3.release()
                return

    def place_order(self, cart_id):
        """
        Finalizes an order. Protected by `self.sem4`.
        """
        self.sem4.acquire()
        order = []
        for aux in self.carts[cart_id]:
            order.append(aux[0])
        self.sem4.release()
        return order


class Producer(Thread):
    """
    Represents a producer that creates and publishes products.
    """
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes the producer."""
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        Main loop for the producer. It registers itself and then continuously
        publishes its products.
        """
        while True:
            producer_id = self.marketplace.register_producer()
            for [prod, quantity, timee] in self.products:
                for i in range(quantity):
                    sleep_time = self.marketplace.publish(producer_id, prod)
                    if sleep_time:
                        time.sleep(timee)
                    else:
                        time.sleep(self.republish_wait_time)
                        i = i - 1
