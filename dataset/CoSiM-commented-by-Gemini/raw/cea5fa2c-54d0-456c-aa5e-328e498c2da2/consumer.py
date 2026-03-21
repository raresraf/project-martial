"""
This module implements a producer-consumer simulation of an online marketplace.

It defines classes for Producers, Consumers, and the central Marketplace.
Producers create products and publish them. Consumers have shopping lists and
attempt to add items to carts, which reserves them. Once a shopping list is
fulfilled, the consumer places the order. The code is notable for using
Romanian names for many variables and methods, which are translated and
explained in the documentation.
"""

from threading import Thread, Lock, current_thread
from time import sleep


class Consumer(Thread):
    """
    Represents a consumer thread that attempts to purchase items from a list.

    Each consumer processes a list of shopping carts. For each cart, it
    attempts to add or remove products. If any operation fails (e.g., product
    is out of stock), it retries the entire shopping list after a delay.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer thread.

        Args:
            carts (list): A list of shopping lists for the consumer to process.
            marketplace (Marketplace): The marketplace instance to interact with.
            retry_wait_time (float): Time in seconds to wait before retrying a
                                     failed shopping list.
            **kwargs: Keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, name=kwargs["name"], kwargs=kwargs)
        
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        
    def run(self):
        """The main logic for the consumer thread."""
        # lista_cumparaturi = shopping_list
        for lista_cumparaturi in self.carts:
            cart_id = self.marketplace.new_cart()
            
            failed = True
            # This loop retries the entire shopping list if any single operation fails.
            while failed:
                failed = False
                # operatie = operation
                for operatie in lista_cumparaturi:
                    # cantitate = quantity
                    cantitate = operatie["quantity"]
                    for _ in range(cantitate):
                        if operatie["type"] == "add":
                            if self.marketplace.add_to_cart(cart_id, operatie["product"]):
                                # Decrement quantity to track progress for the current operation.
                                operatie["quantity"] = operatie["quantity"] - 1
                            else:
                                # If adding to cart fails, mark the whole list as failed and retry.
                                failed = True
                                break
                        elif operatie["type"] == "remove":
                            if self.marketplace.remove_from_cart(cart_id, operatie["product"]):
                                operatie["quantity"] = operatie["quantity"] - 1
                            else:
                                failed = True
                                break
            if failed:
                # Wait before retrying the entire shopping list.
                sleep(self.retry_wait_time)
            else:
                # If all operations were successful, place the order.
                self.marketplace.place_order(cart_id)


class PublishedProduct:
    """
    A wrapper class for a product, adding a reservation status.
    This allows items to be "held" in a cart before being officially sold.
    """
    def __init__(self, product):
        self.product = product
        self.reserved = False
    
    def __eq__(self, obj):
        """Checks equality based on product and reservation status."""
        ret = isinstance(obj, PublishedProduct) and self.reserved == obj.reserved
        return ret and obj.product == self.product

class ProductsList:
    """
    A thread-safe inventory list for a single producer.
    It has a fixed capacity and supports reserving items.
    """
    def __init__(self, maxsize):
        self.lock = Lock()
        self.list = []
        self.maxsize = maxsize

    def put(self, item):
        """Adds an item to the inventory if there is space."""
        with self.lock:
            if self.maxsize == len(self.list):
                return False
            self.list.append(item)
        return True

    def rezerva(self, item):
        """
        Reserves an available (unreserved) item in the inventory.
        'rezerva' is Romanian for 'reserve'.
        """
        item = PublishedProduct(item)
        with self.lock:
            if item in self.list:
                self.list[self.list.index(item)].reserved = True
                return True
        return False

    def anuleaza_rezervarea(self, item):
        """
        Cancels the reservation for an item, making it available again.
        'anuleaza_rezervarea' is Romanian for 'cancel the reservation'.
        """
        item = PublishedProduct(item)
        item.reserved = True
        with self.lock:
            self.list[self.list.index(item)].reserved = False

    def remove(self, item):
        """Removes a reserved item from inventory upon final sale."""
        product = PublishedProduct(item)
        product.reserved = True
        with self.lock:
            self.list.remove(product)
            return item

class Cart:
    """Represents a consumer's shopping cart, holding products to be purchased."""
    def __init__(self):
        self.products = []

    def add_product(self, product, producer_id):
        """Adds a product and the ID of the producer it came from."""
        self.products.append((product, producer_id))

    def remove_product(self, product):
        """Removes a product from the cart and returns the producer's ID."""
        for item in self.products:
            if item[0] == product:
                self.products.remove(item)
                return item[1]
        return None

    def get_products(self):
        """Returns all products currently in the cart."""
        return self.products

class Marketplace:
    """
    The central marketplace that coordinates producers and consumers.
    This class is thread-safe.
    """
    def __init__(self, queue_size_per_producer):
        self.print_lock = Lock()
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_queues = {}
        
        # 'generator_id_producator' = producer ID generator
        self.generator_id_producator = 0
        self.generator_id_producator_lock = Lock()

        self.carts = {}
        self.cart_id_generator = 0
        self.cart_id_generator_lock = Lock()

    def register_producer(self):
        """
        Registers a new producer, giving them a unique ID and an inventory list.
        """
        # id_producator = producer_id
        id_producator = None
        with self.generator_id_producator_lock:
            id_producator = self.generator_id_producator
            self.generator_id_producator += 1
            self.producer_queues[id_producator] = ProductsList(self.queue_size_per_producer)
        return id_producator

    def publish(self, producer_id, product):
        """Allows a producer to publish a new product to their inventory."""
        return self.producer_queues[producer_id].put(PublishedProduct(product))

    def new_cart(self):
        """Creates a new, unique cart for a consumer."""
        with self.cart_id_generator_lock:
            current_cart_id = self.cart_id_generator
            self.cart_id_generator += 1
            self.carts[current_cart_id] = Cart()
            return current_cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart by finding an available one from any producer
        and reserving it.
        """
        producers_num = 0
        with self.generator_id_producator_lock:
            producers_num = self.generator_id_producator

        for producer_id in range(producers_num):
            # Tries to reserve the product from a producer's queue.
            if self.producer_queues[producer_id].rezerva(product):
                self.carts[cart_id].add_product(product, producer_id)
                return True
        return False

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a cart and cancels its reservation."""
        producer_id = self.carts[cart_id].remove_product(product)
        if producer_id is None:
            return False
        self.producer_queues[producer_id].anuleaza_rezervarea(product)
        return True

    def place_order(self, cart_id):
        """
        Finalizes an order, removing products from inventory and printing a
        confirmation message.
        """
        # lista = list
        lista = list()
        # produs = product
        for (produs, producer_id) in self.carts[cart_id].get_products():
            lista.append(self.producer_queues[producer_id].remove(produs))
            with self.print_lock:
                print(f"{current_thread().getName()} bought {produs}")
        return lista


class Producer(Thread):
    """
    Represents a producer thread that creates products and publishes them to
    the marketplace.
    """
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes the Producer thread.
        Args:
            products (list): A list of products the producer can create,
                             including quantity and production time.
            marketplace (Marketplace): The marketplace instance.
            republish_wait_time (float): Time to wait if publishing fails
                                         (e.g., inventory is full).
            **kwargs: Keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, name=kwargs["name"], daemon=kwargs["daemon"], kwargs=kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """The main logic for the producer thread."""
        producer_id = self.marketplace.register_producer()
        
        while True:
            # (product, cantitate, production_time) = (product, quantity, production_time)
            for (product, cantitate, production_time) in self.products:
                sleep(production_time)
                for _ in range(cantitate):
                    # Keep trying to publish the product until successful.
                    while not self.marketplace.publish(producer_id, product):
                        sleep(self.republish_wait_time)
