"""
This module implements a multi-threaded producer-consumer simulation for a marketplace.

It defines classes for Consumers, Producers, and a central Marketplace where they
interact. The simulation uses threading to model concurrent producers adding products
and consumers purchasing them.
"""

from threading import Thread
from time import sleep


class Consumer(Thread):
    """
    Represents a consumer thread that simulates purchasing products from the marketplace.

    Each consumer processes a list of carts, with each cart containing a series of
    add or remove operations.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer instance.

        :param carts: A list of carts, where each cart is a list of purchase operations.
        :param marketplace: The Marketplace instance to interact with.
        :param retry_wait_time: Time in seconds to wait before retrying to add a product.
        :param kwargs: Additional keyword arguments for the Thread base class.
        """
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        Thread.__init__(self, **kwargs)

    def run(self):
        """
        The main execution logic for the consumer thread.

        Iterates through its assigned carts, executes the purchase operations for each,
        places the order, and prints the items bought.
        """
        for cart in self.carts:
            id_cart = self.marketplace.new_cart()
            for purchase in cart:
                if purchase["type"] == 'add':
                    for _ in range(purchase["quantity"]):
                        cart_new_product = self.marketplace.add_to_cart(id_cart,
                                                                        purchase["product"])

                        # Retry mechanism if a product cannot be added immediately.
                        while not cart_new_product:
                            sleep(self.retry_wait_time)
                            cart_new_product = self.marketplace.add_to_cart(id_cart,
                                                                            purchase["product"])
                else:
                    for _ in range(purchase["quantity"]):
                        self.marketplace.remove_from_cart(id_cart, purchase["product"])
            order = self.marketplace.place_order(id_cart)
            for buy in order:
                print(self.name + ' bought ' + str(buy))


import threading


class Marketplace:
    """
    Manages the inventory and all interactions between producers and consumers.

    This class is the synchronized core of the simulation, using locks to handle
    concurrent access to its data structures.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        :param queue_size_per_producer: The maximum number of products a single
                                        producer can have in the marketplace at one time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.contor_producer = -1
        self.contor_consumer = -1
        # Data structures to manage products and carts.
        self.product_queue = [[]]
        self.cart_queue = [[]]
        self.producer_cart = [[]]
        # Synchronization primitives.
        self.lock = threading.Lock()
        self.producer_locks = []

    def register_producer(self):
        """
        Registers a new producer, allocating necessary data structures and a lock.

        :return: The ID assigned to the new producer.
        """
        with self.lock:
            self.contor_producer += 1
            tmp = self.contor_producer
            self.product_queue.append([])
            self.producer_cart.append([])
            self.producer_locks.append(threading.Lock())
        return tmp

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.

        :param producer_id: The ID of the producer publishing the product.
        :param product: The product to be published.
        :return: True if publishing was successful, False otherwise (e.g., queue is full).
        """
        self.producer_locks[producer_id].acquire()
        if self.queue_size_per_producer > len(self.product_queue[producer_id]):
            self.product_queue[producer_id].append(product)
            self.producer_locks[producer_id].release()
            return True
        self.producer_locks[producer_id].release()
        return False

    def new_cart(self):
        """
        Creates a new, empty cart for a consumer.

        :return: The ID of the newly created cart.
        """
        self.lock.acquire()
        self.contor_consumer += 1
        self.cart_queue.append([])
        tmp = self.contor_consumer
        self.lock.release()
        return tmp

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specified consumer's cart.

        This method performs a linear search for the product across all producer
        queues, which is inefficient. The coarse-grained lock during this
        operation can also become a performance bottleneck.
        
        :param cart_id: The ID of the cart to add the product to.
        :param product: The product to add.
        :return: True if the product was found and added, False otherwise.
        """
        if any(product in list_products for list_products in self.product_queue):
            for products in self.product_queue:
                for prod in products:
                    if prod == product:
                        self.lock.acquire()
                        # The following operations are performed under a single lock,
                        # which can be a contention point.
                        tmp = self.product_queue.index(products)
                        self.producer_cart[tmp].append((product, cart_id))
                        self.cart_queue[cart_id].append(product)
                        self.product_queue[tmp].remove(product)
                        self.lock.release()
                        return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a consumer's cart and returns it to the producer.
        
        :param cart_id: The ID of the cart.
        :param product: The product to be removed.
        """
        self.cart_queue[cart_id].remove(product)
        # This nested loop to find the original producer is inefficient.
        for producer in self.producer_cart:
            if (cart_id, product) in producer:
                tmp = self.producer_cart.index(producer)
                self.producer_cart.remove((cart_id, product))
                # Re-acquires a specific producer lock to return the product.
                self.producer_locks[tmp].acquire()
                self.product_queue[tmp].append(product)
                self.producer_locks[tmp].release()

    def place_order(self, cart_id):
        """
        Finalizes an order by returning the contents of the cart.

        :param cart_id: The ID of the cart to place an order for.
        :return: A list of products in the cart.
        """
        return self.cart_queue[cart_id]


from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Represents a producer thread that generates products and publishes them to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer instance.

        :param products: A list of products that the producer can generate.
        :param marketplace: The Marketplace instance to interact with.
        :param republish_wait_time: Time in seconds to wait before retrying to publish.
        :param kwargs: Additional keyword arguments for the Thread base class.
        """
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        Thread.__init__(self, **kwargs)

    def run(self):
        """
        The main execution logic for the producer thread.

        Registers with the marketplace and then enters an infinite loop, continuously
        producing and publishing its products according to specified quantities and delays.
        """
        id_producer = self.marketplace.register_producer()
        while True:
            for product in self.products:
                for _ in range(product[1]):
                    sleep(product[2])
                    market_confirm = self.marketplace.publish(id_producer, product[0])
                    
                    # Retry logic if the marketplace queue for this producer is full.
                    while not market_confirm:


                        sleep(self.republish_wait_time)
                        market_confirm = self.marketplace.publish(id_producer, product[0])


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a product, containing a name and a price."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass for Tea, inheriting from Product and adding a 'type' attribute."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass for Coffee, inheriting from Product and adding acidity and roast level."""
    acidity: str
    roast_level: str
