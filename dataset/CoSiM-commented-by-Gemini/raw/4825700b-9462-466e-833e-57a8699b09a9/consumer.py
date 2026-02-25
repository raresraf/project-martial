"""
Models a multi-producer, multi-consumer marketplace simulation.

This module uses threading to simulate the behavior of producers who publish
products and consumers who purchase them. The Marketplace class acts as the
central, thread-safe intermediary for all transactions.

Classes:
    Consumer: A thread that simulates a customer adding and removing products
              from shopping carts and placing orders.
    Marketplace: A thread-safe class that manages product inventories, shopping
                 carts, and all interactions between producers and consumers.
    Producer: A thread that simulates a supplier publishing products to the
              marketplace.
"""
from threading import Thread
import time

class Consumer(Thread):
    """
    Represents a consumer thread that interacts with the marketplace.
    
    Each consumer processes a list of shopping carts, where each cart is a
    sequence of 'add' and 'remove' commands.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer thread.

        Args:
            carts (list): A list of shopping lists for the consumer to process.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time in seconds to wait before retrying to
                                     add a product if it's unavailable.
            **kwargs: Keyword arguments for the Thread constructor (e.g., name).
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs["name"]

    def add_product(self, cart_id, product):
        """
        Persistently tries to add a product to a specific cart.
        
        If the marketplace cannot add the product immediately (e.g., it is
        out of stock), this method will wait and retry.
        """
        added = False
        while not added:
            added = self.marketplace.add_to_cart(cart_id, product)
            if not added:
                time.sleep(self.retry_wait_time)

    def run(self):
        """The main execution logic for the consumer thread."""
        carts_id = []
        
        # Process each shopping list assigned to this consumer.
        for cart in self.carts:
            cart_id = self.marketplace.new_cart()
            carts_id.append(cart_id)
            
            # Execute the commands ('add' or 'remove') in the current shopping list.
            for command in cart:
                if command["type"] == "add":
                    for _ in range(command["quantity"]):
                        self.add_product(cart_id, command["product"])
                else:
                    for _ in range(command["quantity"]):
                        self.marketplace.remove_from_cart(cart_id, command["product"])

        # After filling the carts, place the orders.
        for cart_id in carts_id:
            products = self.marketplace.place_order(cart_id)
            for product in products:
                print(f'{self.name} bought {product}', flush=True)

from threading import Lock

class Marketplace:
    """
    A thread-safe marketplace that manages producers, products, and carts.
    
    This class is the central hub for the simulation, providing synchronized
    methods for all producer and consumer actions.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a
                                           single producer can have listed at one time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        
        # Lock for generating unique producer IDs.
        self.register_lock = Lock()
        self.producer_id = 0

        # Lock for generating unique cart IDs.
        self.cart_lock = Lock()
        self.cart_id = 0
        
        # Data structures for managing state.
        self.products = []  # List of dicts, one per producer's inventory.
        self.carts = []     # List of dicts, one per shopping cart.
        self.sizes = []     # List of current inventory sizes per producer.
        self.producers_lock = [] # List of locks, one per producer's inventory.

    def register_producer(self):
        """
        Registers a new producer, providing a unique ID and initializing storage.
        
        Returns:
            int: The unique ID for the new producer.
        """
        self.register_lock.acquire()
        id_copy = self.producer_id
        self.producer_id = self.producer_id + 1
        self.register_lock.release()
        
        # Initialize data structures for the new producer.
        self.products.append({})
        self.sizes.append(0)
        self.producers_lock.append(Lock())

        return id_copy

    def publish(self, producer_id, product):
        """
        Allows a producer to list a product for sale.
        
        Returns:
            bool: True if the product was published successfully, False otherwise.
        """
        self.producers_lock[producer_id].acquire()
        if self.sizes[producer_id] < self.queue_size_per_producer:
            # Logic for adding a product to inventory.
            # NOTE: If a product is new, its count is initialized to 0, which might be a bug.
            if product in self.products[producer_id]:
                self.products[producer_id][product] += 1
            else:
                self.products[producer_id][product] = 0

            self.sizes[producer_id] += 1
            self.producers_lock[producer_id].release()
            return True

        self.producers_lock[producer_id].release()
        return False

    def new_cart(self):
        """Creates a new, empty shopping cart and returns its ID."""
        self.cart_lock.acquire()
        id_copy = self.cart_id
        self.cart_id = self.cart_id + 1
        self.cart_lock.release()
        
        self.carts.append({})

        return id_copy

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a shopping cart by taking it from a producer's inventory.
        
        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        for producer_id in range(len(self.products)):
            
            self.producers_lock[producer_id].acquire()
            if product in self.products[producer_id]:

                # Move product from producer stock to consumer cart.
                self.products[producer_id][product] -= 1
                self.sizes[producer_id] -= 1

                if self.products[producer_id][product] == 0:
                    self.products[producer_id].pop(product)
                self.producers_lock[producer_id].release()

                # Add product to cart, tracking its source producer.
                if (product, producer_id) in self.carts[cart_id]:
                    new_quantity = self.carts[cart_id].get((product, producer_id)) + 1
                    self.carts[cart_id].update({(product, producer_id): new_quantity})
                else:
                    self.carts[cart_id].update({(product, producer_id): 1})

                return True

            self.producers_lock[producer_id].release()
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a cart and returns it to the original producer's inventory.
        """
        producer_id = -1
        # Find which producer the product in the cart came from.
        for product_tuple in self.carts[cart_id].keys():
            if product == product_tuple[0]:
                producer_id = product_tuple[1]
                break

        # NOTE: If the product is not found, producer_id remains -1, which will
        # cause an IndexError on the `producers_lock` access below.
        
        # Return the product to the producer's inventory.
        self.producers_lock[producer_id].acquire()
        # NOTE: Initializing a returned product's count to 0 might be a bug.
        if product in self.products[producer_id]:
            self.products[producer_id][product] += 1
        else:
            self.products[producer_id][product] = 0

        self.sizes[producer_id] += 1
        self.producers_lock[producer_id].release()

        # Remove the product from the cart data structure.
        new_quantity = self.carts[cart_id].get((product, producer_id)) - 1
        self.carts[cart_id].update({(product, producer_id): new_quantity})
        if self.carts[cart_id].get((product, producer_id)) == 0:
            # This is an inefficient way to remove a key from a dictionary.
            self.carts[cart_id] = {key: val for key, val in self.carts[cart_id].items()
                                   if key != (product, producer_id)}

    def place_order(self, cart_id):
        """Finalizes an order and returns a simple list of products bought."""
        simple_list = []

        for product_tuple in self.carts[cart_id]:
            for _ in range(self.carts[cart_id][product_tuple]):
                simple_list.append(product_tuple[0])

        return simple_list

from threading import Thread
import time

class Producer(Thread):
    """Represents a producer thread that publishes products to the marketplace."""

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes the Producer thread.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = marketplace.register_producer()

    def run(self):
        """
        The main execution logic for the producer thread. In an infinite loop,
        it continuously tries to publish its products.
        """
        while True:
            for product in self.products:
                published = False
                while not published:
                    published = self.marketplace.publish(self.producer_id, product[0])
                    
                    if not published:
                        time.sleep(self.republish_wait_time)
                    else:
                        # After a successful publication, waits for a product-specific time.
                        time.sleep(product[2])