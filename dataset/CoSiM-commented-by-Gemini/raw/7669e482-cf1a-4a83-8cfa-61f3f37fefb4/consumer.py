"""
This module implements a multi-producer, multi-consumer marketplace simulation.

It models a system where multiple Producer threads create and publish products,
and multiple Consumer threads attempt to purchase them according to a shopping list.
The Marketplace class acts as a central broker, managing inventory and carts.

NOTE: The variable and method names are in Romanian. The comments and docstrings
below will provide English translations and explanations. The implementation also
contains some inefficient design patterns, particularly in the Consumer's retry logic.
"""


from threading import Thread, Lock, current_thread
from time import sleep


class Consumer(Thread):
    """
    Represents a consumer that processes shopping lists.
    
    Each consumer is a thread that takes a list of "carts" (shopping lists)
    and attempts to acquire all items in each list from the marketplace.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes the Consumer thread.

        Args:
            carts (list): A list of shopping lists. Each shopping list is a list of
                          operations (add/remove product).
            marketplace (Marketplace): The central marketplace object.
            retry_wait_time (float): Time to wait before retrying a failed shopping attempt.
        """
        Thread.__init__(self, name=kwargs["name"])
        
        self.carts = carts # A list of shopping lists (`lista_cumparaturi`).
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """The main logic for a consumer thread."""
        # Process each shopping list one by one.
        for shopping_list in self.carts: # `lista_cumparaturi` = "shopping list"
            
            # Get a new cart ID from the marketplace for this shopping attempt.
            cart_id = self.marketplace.new_cart()
            
            # --- Inefficient Retry Loop ---
            # This loop retries the *entire* shopping list if any single operation fails.
            failed = True
            while failed:
                failed = False
                for operation in shopping_list: # `operatie` = "operation"
                    
                    quantity = operation["quantity"] # `cantitate` = "quantity"
                    for _ in range(quantity):
                        if operation["type"] == "add":
                            # Attempt to add one item to the cart.
                            if not self.marketplace.add_to_cart(cart_id, operation["product"]):
                                # If adding fails, mark the whole attempt as failed and break.
                                failed = True
                                break
                        elif operation["type"] == "remove":
                            # Attempt to remove one item from the cart.
                            if not self.marketplace.remove_from_cart(cart_id, operation["product"]):
                                failed = True
                                break
                    if failed:
                        break # Break from the outer loop over the shopping list as well.

                if failed:
                    # If the attempt failed, wait before retrying the whole shopping list.
                    sleep(self.retry_wait_time)
                else:
                    # If all operations succeeded, place the order.
                    self.marketplace.place_order(cart_id)


class PublishedProduct:
    """A wrapper for a product in the marketplace, including a reservation flag."""
    def __init__(self, product):
        self.product = product
        self.reserved = False

    def __eq__(self, obj):
        """Custom equality check for finding unreserved products."""
        if isinstance(obj, PublishedProduct):
            # Two PublishedProduct objects are equal if they represent the same product
            # and have the same reservation status.
            return self.reserved == obj.reserved and self.product == obj.product
        return False

class ProductsList:
    """
    A thread-safe list representing a single producer's inventory.
    
    It supports reserving products before they are finally removed.
    """
    def __init__(self, maxsize):
        self.lock = Lock()
        self.list = []
        self.maxsize = maxsize

    def put(self, item):
        """Adds a new item to the inventory if there is space."""
        with self.lock:
            if len(self.list) < self.maxsize:
                self.list.append(item)
                return True
            return False

    def rezerva(self, item):
        """
        'rezerva' = 'reserve'. Finds an unreserved product and marks it as reserved.
        """
        # Create a temporary object representing an *unreserved* product to search for.
        item_to_find = PublishedProduct(item)
        with self.lock:
            try:
                # Find the index of the first matching unreserved product.
                index = self.list.index(item_to_find)
                # Mark it as reserved.
                self.list[index].reserved = True
                return True
            except ValueError:
                # The item was not found.
                return False

    def anuleaza_rezervarea(self, item):
        """
        'anuleaza_rezervarea' = 'cancel_reservation'. Finds a reserved product and
        marks it as unreserved.
        """
        # Create a temporary object representing a *reserved* product to search for.
        item_to_find = PublishedProduct(item)
        item_to_find.reserved = True
        with self.lock:
            try:
                # Find the reserved product and un-reserve it.
                index = self.list.index(item_to_find)
                self.list[index].reserved = False
            except ValueError:
                # This case might indicate a logic error, as we are trying to
                # cancel a reservation that doesn't exist.
                pass

    def remove(self, item):
        """Removes a reserved item from the inventory, finalizing the purchase."""
        product_to_remove = PublishedProduct(item)
        product_to_remove.reserved = True
        with self.lock:
            self.list.remove(product_to_remove)
            return item

class Cart:
    """Represents a consumer's shopping cart."""
    def __init__(self):
        self.products = []

    def add_product(self, product, producer_id):
        """Adds a product and the ID of the producer it was reserved from."""
        self.products.append((product, producer_id))

    def remove_product(self, product):
        """Removes a product and returns the ID of the producer it belonged to."""
        for item in self.products:
            if item[0] == product:
                self.products.remove(item)
                return item[1] # Return producer_id
        return None

    def get_products(self):
        """Returns the list of (product, producer_id) tuples in the cart."""
        return self.products

class Marketplace:
    """
    The central marketplace that manages producers and consumer carts.
    Acts as the main synchronization point between producers and consumers.
    """
    def __init__(self, queue_size_per_producer):
        self.print_lock = Lock()
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_queues = {}

        # Thread-safe generator for producer IDs.
        self.generator_id_producator = 0 # `producator` = "producer"
        self.generator_id_producator_lock = Lock()

        # Thread-safe generator for cart IDs.
        self.carts = {}
        self.cart_id_generator = 0
        self.cart_id_generator_lock = Lock()

    def register_producer(self):
        """Assigns a unique ID to a new producer and creates their inventory list."""
        with self.generator_id_producator_lock:
            producer_id = self.generator_id_producator
            self.generator_id_producator += 1
            self.producer_queues[producer_id] = ProductsList(self.queue_size_per_producer)
        return producer_id

    def publish(self, producer_id, product):
        """Allows a producer to publish a product to their inventory."""
        return self.producer_queues[producer_id].put(PublishedProduct(product))

    def new_cart(self):
        """Creates a new, empty shopping cart and returns its unique ID."""
        with self.cart_id_generator_lock:
            current_cart_id = self.cart_id_generator
            self.cart_id_generator += 1
            self.carts[current_cart_id] = Cart()
            return current_cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a cart by finding and reserving it from any producer.
        
        NOTE: This linearly scans all producers, which can be a bottleneck.
        """
        with self.generator_id_producator_lock:
            producers_num = self.generator_id_producator

        for producer_id in range(producers_num):
            # Attempt to reserve the product from the current producer.
            if self.producer_queues[producer_id].rezerva(product):
                # If successful, add it to the cart and record which producer it came from.
                self.carts[cart_id].add_product(product, producer_id)
                return True
        # The product could not be found or reserved from any producer.
        return False

    def remove_from_cart(self, cart_id, product):
        """Removes a product from a cart and cancels its reservation."""
        producer_id = self.carts[cart_id].remove_product(product)
        if producer_id is not None:
            self.producer_queues[producer_id].anuleaza_rezervarea(product)
            return True
        return False

    def place_order(self, cart_id):
        """Finalizes the purchase for all items in a cart."""
        purchased_items = []
        for (product, producer_id) in self.carts[cart_id].get_products():
            # Permanently remove the product from the producer's inventory.
            item = self.producer_queues[producer_id].remove(product)
            purchased_items.append(item)
            with self.print_lock:
                print(f"{current_thread().getName()} bought {product}")
        return purchased_items


class Producer(Thread):
    """Represents a producer that creates products and publishes them to the marketplace."""

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes the Producer thread.
        
        Args:
            products (list): A list of (product, quantity, production_time) tuples.
            marketplace (Marketplace): The central marketplace object.
            republish_wait_time (float): Time to wait if publishing fails (e.g., inventory is full).
        """
        Thread.__init__(self, name=kwargs.get("name"), daemon=kwargs.get("daemon"))
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """Main logic for the producer thread."""
        producer_id = self.marketplace.register_producer()
        
        while True:
            # Cycle through the list of products this producer can make.
            for (product, quantity, production_time) in self.products:
                # Simulate the time it takes to produce the item.
                sleep(production_time)
                
                # Produce the specified quantity of the item.
                for _ in range(quantity):
                    # Attempt to publish the product to the marketplace.
                    # If the producer's inventory is full, this will fail.
                    while not self.marketplace.publish(producer_id, product):
                        # Wait and retry if publishing fails.
                        sleep(self.republish_wait_time)
