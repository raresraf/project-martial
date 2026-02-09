"""
A multi-threaded producer-consumer marketplace simulation.

This script defines a system with three main components:
- Producer: A thread that generates products and makes them available in the marketplace.
- Consumer: A thread that acquires products from the marketplace by assembling a cart and placing an order.
- Marketplace: A central, shared resource that facilitates the exchange, managing product inventory
  and customer carts with basic locking for thread safety.

The simulation uses standard Python threading and locking primitives to manage concurrent operations.
"""

from threading import Thread, current_thread
from time import sleep


class Consumer(Thread):
    """
    Represents a consumer that processes a list of shopping carts.
    
    Each consumer thread is initialized with a set of carts, where each cart is a list of
    add/remove operations. The consumer executes these operations against the marketplace,
    retrying if a product is temporarily unavailable, and finally places the order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer instance.

        Args:
            carts (list): A list of shopping carts. Each cart is a list of dictionaries,
                          with each dictionary specifying a product, quantity, and operation type
                          ('add' or 'remove').
            marketplace (Marketplace): The shared marketplace instance where products are exchanged.
            retry_wait_time (float): The time in seconds to wait before retrying to add a
                                     product that is not available.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        super().__init__(**kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        The main execution logic for the consumer thread.

        Iterates through each assigned cart, creates a new cart session in the marketplace,
        and processes each operation (add/remove). If an 'add' operation cannot be fulfilled
        immediately, it waits and retries. After processing all operations in a cart,
        it places the order and prints the acquired products.
        """
        for cart in self.carts:
            # Pre-condition: A new, empty cart is required for each set of transactions.
            # Invariant: cart_id uniquely identifies the consumer's current session.
            cart_id = self.marketplace.new_cart()
            for operation in cart:
                # Block Logic: Executes a series of 'add' or 'remove' operations for a single product.
                for _ in range(operation['quantity']):
                    if operation['type'] == 'add':
                        # Block Logic: Persistently tries to add a product to the cart.
                        # Invariant: The loop continues until the marketplace confirms the product has been
                        # successfully added, ensuring the operation completes.
                        while not self.marketplace.add_to_cart(cart_id, operation['product']):
                            sleep(self.retry_wait_time)
                    elif operation['type'] == 'remove':
                        self.marketplace.remove_from_cart(cart_id, operation['product'])
            
            # Finalizes the transaction and retrieves the list of products.
            products = self.marketplace.place_order(cart_id)
            for product in products:
                print(f"{current_thread().name} bought {product}")

from threading import Lock


class Marketplace:
    """
    Manages the inventory and transactions between producers and consumers.
    
    This class acts as the central shared resource, handling product queues from multiple
    producers and managing active shopping carts for consumers. It uses locks to ensure
    thread-safe modifications of its internal state during concurrent access.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace instance.

        Args:
            queue_size_per_producer (int): The maximum number of products that each
                                           producer can have in their published queue at one time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.current_producer_id = -1
        self.producer_queues = {}
        self.producers_lock = Lock()
        self.current_cart_id = -1
        self.carts = {}
        self.consumers_lock = Lock()

    def register_producer(self):
        """
        Assigns a unique ID to a new producer and initializes their product queue.
        
        This method is a critical section, protected by a lock to prevent race conditions
        when multiple producers register simultaneously.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        with self.producers_lock:
            self.current_producer_id += 1
            self.producer_queues[self.current_producer_id] = []
            aux = self.current_producer_id
            return aux

    def publish(self, producer_id, product):
        """
        Adds a product to a specific producer's public inventory queue.

        The operation will fail if the producer's queue is already at its maximum capacity.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (Product): The product to be published.

        Returns:
            bool: True if the product was successfully published, False otherwise.
        """
        # Pre-condition: The producer's queue must not be full.
        if len(self.producer_queues[producer_id]) >= self.queue_size_per_producer:
            return False

        self.producer_queues[producer_id].append(product)
        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart and assigns it a unique ID.
        
        This is a critical section protected by a lock to ensure unique cart ID generation.

        Returns:
            int: The unique ID for the newly created cart.
        """
        with self.consumers_lock:
            self.current_cart_id += 1
            self.carts[self.current_cart_id] = []
            return self.current_cart_id

    def add_to_cart(self, cart_id, product):
        """
        Transfers a product from any available producer's queue to the consumer's cart.

        This method iterates through all producer queues to find the requested product.
        Once found, it moves the product from the producer's queue to the consumer's cart.
        Note: The iteration over producer queues is not atomic and could be subject to
        race conditions if queues were modified by other threads without a lock.

        Args:
            cart_id (int): The ID of the consumer's cart.
            product (Product): The product to add.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        # Block Logic: Searches for the specified product across all producer inventories.
        # Invariant: The product is sourced from the first available producer.
        for producer_id, producer_queue in self.producer_queues.items():
            if product in producer_queue:
                producer_queue.remove(product)
                self.carts[cart_id].append((producer_id, product))
                return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from the consumer's cart and returns it to the original producer.

        Args:
            cart_id (int): The ID of the consumer's cart.
            product (Product): The product to remove.
        """
        p_id = -1
        # Block Logic: Finds the product in the cart to identify its original producer.
        for producer_id, cart_product in self.carts[cart_id]:
            if cart_product == product:
                self.producer_queues[producer_id].append(product)
                p_id = producer_id
        self.carts[cart_id].remove((p_id, product))

    def place_order(self, cart_id):
        """
        Finalizes the shopping cart session and returns the list of products.

        Args:
            cart_id (int): The ID of the cart to finalize.

        Returns:
            list: A list of the products that were in the cart.
        """
        return [cart_product for producer_id, cart_product in self.carts[cart_id]]

from threading import Thread
from time import sleep


class Producer(Thread):
    """
    Represents a producer that generates and publishes products to the marketplace.
    
    The producer runs in an infinite loop, periodically publishing a predefined list
    of products. It retries publishing if the marketplace queue is full.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer instance.

        Args:
            products (list): A list of products to generate. Each element is a tuple
                             containing the product, the quantity to produce, and the
                             time to wait after producing this batch.
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): The time in seconds to wait before retrying to
                                         publish a product when the queue is full.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        super().__init__(**kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = marketplace.register_producer()

    def run(self):
        """
        The main execution logic for the producer thread.
        
        Enters an infinite loop to continuously produce items. For each product type,
        it publishes the specified quantity, waiting and retrying if the marketplace's
        queue is full. It then sleeps for a designated time before repeating the cycle.
        """
        while True:
            # Block Logic: Iterates through the predefined list of products to generate.
            for product, num, time in self.products:
                # Block Logic: Publishes a batch of a single product.
                for _ in range(num):
                    # Block Logic: Persistently tries to publish the product.
                    # Invariant: Loop continues until the product is successfully published.
                    while not self.marketplace.publish(self.producer_id, product):
                        sleep(self.republish_wait_time)
                sleep(time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass representing a generic product with a name and a price."""
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass representing a type of tea, inheriting from Product."""
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass representing a type of coffee, inheriting from Product."""
    acidity: str
    roast_level: str