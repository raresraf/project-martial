"""
This module implements a Producer-Consumer simulation for a marketplace.

It defines a `Marketplace` as the central shared resource, where `Producer`
threads can publish products and `Consumer` threads can add products to carts
and place orders. The simulation uses threading to model concurrent producers
and consumers interacting with the marketplace. Synchronization is handled via
a single lock within the Marketplace, though some operations have potential
race conditions.
"""

from threading import Thread, currentThread, Lock
import time

class Consumer(Thread):
    """
    Represents a consumer thread that interacts with the marketplace.

    Each consumer has a list of shopping carts, where each cart is a sequence
    of 'add' or 'remove' operations. The consumer processes each cart,
    executes the operations, and finally places the order.

    Attributes:
        carts (list): A list of carts, where each cart is a list of operations.
        marketplace (Marketplace): The shared marketplace instance.
        retry_wait_time (float): Time to wait before retrying a failed operation.
        name (str): The name of the consumer thread.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes the Consumer thread."""
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name']

    def get_name(self):
        """Returns the name of the consumer thread."""
        return self.name

    def run(self):
        """
        The main execution loop for the consumer.
        Processes each cart by performing add/remove operations and placing an order.
        """
        for cart in self.carts:
            # Create a new cart in the marketplace for this shopping session.
            cart_id = self.marketplace.new_cart()
            for operation in cart:
                quan_nr = 0
                # Perform the operation 'quantity' times.
                while quan_nr < operation['quantity']:
                    res = None
                    if operation['type'] == 'add':
                        res = self.marketplace.add_to_cart(cart_id,
                                                           operation['product'])
                    elif operation['type'] == 'remove':
                        res = self.marketplace.remove_from_cart(cart_id,
                                                                operation['product'])
                    
                    # If the operation was successful, increment the count.
                    if res is None or res is True:
                        quan_nr = quan_nr + 1
                    else:
                        # If the operation failed (e.g., product not available),
                        # wait before retrying.
                        time.sleep(self.retry_wait_time)
            
            # Finalize the purchase for the current cart.
            self.marketplace.place_order(cart_id)

class Marketplace:
    """
    The central marketplace, acting as the shared resource for producers and consumers.

    It manages product inventory, producer registration, and shopping carts.
    Synchronization is partially managed by a single master lock.

    Attributes:
        queue_size_per_producer (int): Max number of items a producer can publish.
        lock (Lock): A lock to protect shared state modifications.
    """
    def __init__(self, queue_size_per_producer):
        """Initializes the Marketplace."""
        self.queue_size_per_producer = queue_size_per_producer
        self.nr_of_producers = 0
        self.nr_of_carts = 0
        # `nr_of_items` tracks the number of published items per producer.
        self.nr_of_items = []
        # `carts` stores the contents of each active shopping cart.
        self.carts = {}
        # `producers` maps a product to the ID of the producer who published it.
        self.producers = {}
        self.lock = Lock()

    def register_producer(self):
        """
        Registers a new producer, assigning it a unique ID.

        Returns:
            int: The newly registered producer's ID.
        """
        with self.lock:
            producer_id = self.nr_of_producers
            self.nr_of_producers = self.nr_of_producers + 1
            # Initialize the item count for this new producer to 0.
            self.nr_of_items.insert(producer_id, 0)
        return producer_id

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.

        The operation fails if the producer has already reached its item limit.
        
        Returns:
            bool: True if publishing was successful, False otherwise.
        """
        # Check if the producer's queue is full.
        if self.nr_of_items[int(producer_id)] >= self.queue_size_per_producer:
            return False
        
        # This block is not protected by a lock and could be a race condition
        # if multiple threads from the same producer call publish concurrently.
        self.nr_of_items[int(producer_id)] += 1
        self.producers[product] = int(producer_id)
        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart and returns its ID.

        Returns:
            int: The ID of the new cart.
        """
        with self.lock:
            self.nr_of_carts = self.nr_of_carts + 1
            cart_id = self.nr_of_carts
        
        self.carts[cart_id] = []
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specific shopping cart.

        This "moves" the product from the general producer stock to the user's cart.
        
        Returns:
            bool: True on success, False if the product is not available.
        """
        with self.lock:
            # Check if the product exists in the marketplace.
            if self.producers.get(product) is None:
                return False
            
            # Decrement producer's published item count and remove from public listing.
            self.nr_of_items[self.producers[product]] -= 1
            producers_id = self.producers.pop(product)

        # These operations are outside the lock and could race with other
        # operations on the same cart_id.
        self.carts[cart_id].append(product)
        self.carts[cart_id].append(producers_id)
        return True

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a cart, returning it to the marketplace."""
        # Unsafe check outside of lock.
        if product in self.carts[cart_id]:
            # This block is unsynchronized.
            index = self.carts[cart_id].index(product)
            self.carts[cart_id].remove(product)
            producers_id = self.carts[cart_id].pop(index)
            
            # Put the product back into the marketplace under the original producer.
            self.producers[product] = producers_id
            with self.lock:
                self.nr_of_items[int(producers_id)] += 1

    def place_order(self, cart_id):
        """
        Finalizes an order, printing the items bought.
        
        This method is not fully synchronized and has known race conditions.
        """
        product_list = self.carts.pop(cart_id)
        
        for i in range(0, len(product_list), 2):
            with self.lock:
                print(currentThread().get_name() +" bought " + str(product_list[i]))
                # This item count decrement is documented as racy in the original context.
                self.nr_of_items[product_list[i + 1]] -= 1
        return product_list

class Producer(Thread):
    """
    Represents a producer thread that creates and publishes products.

    Attributes:
        products (list): A list of (product, quantity, sleep_time) tuples.
        marketplace (Marketplace): The shared marketplace instance.
        republish_wait_time (float): Time to wait before retrying to publish.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes the Producer thread."""
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = None

    def run(self):
        """
        Main execution loop for the producer.
        Continuously produces items from its product list and publishes them.
        """
        self.producer_id = self.marketplace.register_producer()
        while True:
            for product in self.products:
                produced = 0
                while produced < product[1]:
                    # Attempt to publish the product.
                    res = self.marketplace.publish(str(self.producer_id), product[0])
                    if res:
                        # If successful, wait for the specified "production time".
                        time.sleep(product[2])
                        produced = produced + 1
                    else:
                        # If unsuccessful (e.g., queue full), wait and retry.
                        time.sleep(self.republish_wait_time)

from dataclasses import dataclass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for a product with a name and price."""
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """A dataclass for Tea, inheriting from Product and adding a type."""
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """A dataclass for Coffee, adding acidity and roast level."""
    acidity: str
    roast_level: str
