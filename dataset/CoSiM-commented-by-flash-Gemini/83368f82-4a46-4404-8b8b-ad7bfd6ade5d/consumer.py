"""
@file consumer.py
@brief Implements a multi-threaded simulated e-commerce marketplace with producers, consumers, and product management.

This module sets up a concurrent environment where `Producer` threads supply products to a `Marketplace`,
and `Consumer` threads interact with the marketplace to add/remove products from carts and place orders.
Synchronization mechanisms (Locks, Events) are used to manage shared resources and coordinate thread activities.

Key Components:
- `Consumer`: Represents a buyer, executing a list of shopping actions on the marketplace.
- `Marketplace`: Manages product inventory, producer registration, product publication, and consumer cart operations, handling all transactional logic.
- `Producer`: Represents a seller, continuously publishing products to the marketplace.
- `Product`: A base class for items sold, with specific subclasses like `Tea` and `Coffee` demonstrating product attributes.
"""


from threading import Thread, currentThread
from time import sleep

class Consumer(Thread):
    """
    Represents a consumer thread in the marketplace simulation.
    Each consumer executes a predefined sequence of shopping actions (adding/removing products)
    and ultimately places an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping cart dictionaries, each detailing products and quantities.
            marketplace (Marketplace): The shared marketplace instance to interact with.
            retry_wait_time (float): Time in seconds to wait before retrying an action if the marketplace is busy.
            **kwargs: Arbitrary keyword arguments passed to the Thread.__init__.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        The main execution loop for the Consumer thread.
        It iterates through the assigned carts, performs add/remove actions,
        and places orders on the marketplace.
        """
        # Block Logic: Iterates through each shopping cart assigned to this consumer.
        for cart in self.carts:
            # Functional Utility: Requests a new shopping cart ID from the marketplace.
            cart_id = self.marketplace.new_cart()

            # Block Logic: Processes each action (add or remove product) within the current cart.
            for action in cart:
                count = 0
                # Invariant: Continues until the desired quantity for the current action is fulfilled.
                while count < action['quantity']:
                    # Conditional Logic: Determines whether to add or remove a product.
                    if action['type'] == 'add':
                        # Functional Utility: Attempts to add a product to the cart.
                        if self.marketplace.add_to_cart(cart_id, action['product']) is False:
                            # Functional Utility: Waits for a specified duration before retrying if the add operation fails.
                            sleep(self.retry_wait_time)
                        else:
                            # Functional Utility: Increments the count upon successful addition.
                            count += 1
                    elif action['type'] == 'remove':
                        # Functional Utility: Removes a product from the cart.
                        self.marketplace.remove_from_cart(cart_id, action['product'])
                        # Functional Utility: Increments the count upon successful removal.
                        count += 1

            # Functional Utility: Places the order for all products currently in the cart.
            products_in_cart = self.marketplace.place_order(cart_id)
            # Block Logic: Prints a confirmation for each product bought.
            for product in products_in_cart:
                print(currentThread().getName() + " bought " + str(product))

from threading import Lock

class Marketplace:
    """
    Simulates an e-commerce marketplace, managing product inventory,
    producer registration, product publication, and consumer cart operations.
    It uses locks for thread-safe access to shared data structures.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single producer
                                           can have in the marketplace at any given time.
        """
        
        self.queue_size_per_producer = queue_size_per_producer
        # Dictionary to store all active shopping carts, keyed by cart_id.
        self.all_carts = {}
        # Lock to protect `cart_id` generation.
        self.carts_id_lock = Lock()
        # Counter for unique cart IDs.
        self.cart_id = -1
        # Counter for unique producer IDs.
        self.producer_id = -1
        # Lock to protect `producer_id` generation.
        self.producer_id_lock = Lock()
        # List of all products currently available in the marketplace.
        self.products_in_marketplace = []
        # Dictionary to track the number of products each producer has published.
        self.producers_queues = {}
        # Dictionary to store products published by each producer.
        self.producers_products = {}
        # Lock to protect add/remove operations in the marketplace.
        self.add_remove_lock = Lock()

    def register_producer(self):
        """
        Registers a new producer with the marketplace, assigning a unique ID
        and initializing its product queues.

        Returns:
            int: The newly assigned unique producer ID.
        """
        # Functional Utility: Acquires a lock to ensure atomic increment of `producer_id`.
        self.producer_id_lock.acquire()
        self.producer_id += 1   
        # Functional Utility: Releases the lock after updating `producer_id`.
        self.producer_id_lock.release()

        # Functional Utility: Initializes an empty list for the new producer's products.
        self.producers_products[self.producer_id] = []    
        # Functional Utility: Initializes the product count for the new producer to 0.
        self.producers_queues[self.producer_id] = 0    

        return self.producer_id

    def publish(self, producer_id, product):
        """
        Publishes a product from a producer to the marketplace.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (str): The name of the product to publish.

        Returns:
            bool: True if the product was successfully published, False otherwise
                  (e.g., if the producer's queue is full).
        """
        
        # Conditional Logic: Checks if the producer's queue has space for a new product.
        if self.producers_queues[int(producer_id)] < self.queue_size_per_producer:
            # Functional Utility: Increments the producer's queue size.
            self.producers_queues[int(producer_id)] += 1
            # Functional Utility: Adds the product to the global marketplace inventory.
            self.products_in_marketplace.append(product)   
            # Functional Utility: Adds the product to the producer's individual product list.
            self.producers_products[int(producer_id)].append(product)
            return True

        return False

    def new_cart(self):
        """
        Creates a new, empty shopping cart and assigns it a unique cart ID.

        Returns:
            int: The newly assigned unique cart ID.
        """
        # Functional Utility: Acquires a lock to ensure atomic increment of `cart_id`.
        self.carts_id_lock.acquire()
        self.cart_id += 1   
        # Functional Utility: Releases the lock after updating `cart_id`.
        self.carts_id_lock.release()
        # Functional Utility: Initializes an empty list for the new cart in `all_carts`.
        self.all_carts[self.cart_id] = []

        return self.cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specified shopping cart.
        If the product is available in the marketplace, it is removed from the global
        inventory and the corresponding producer's queue.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (str): The name of the product to add.

        Returns:
            bool: True if the product was successfully added, False if not found in marketplace.
        """
        
        # Functional Utility: Acquires a lock to ensure atomicity of add/remove operations in the marketplace.
        with self.add_remove_lock:
            # Conditional Logic: Checks if the product is currently available in the marketplace.
            if product in self.products_in_marketplace:
                # Functional Utility: Removes the product from the global marketplace inventory.
                self.products_in_marketplace.remove(product)

                # Block Logic: Locates the producer who supplied this product to update its queue.
                for producer in self.producers_products:
                    # Conditional Logic: Checks if the product belongs to the current producer.
                    if product in self.producers_products[producer]: 
                        # Functional Utility: Decrements the producer's active product count.
                        self.producers_queues[producer] -= 1
                        # Functional Utility: Removes the product from the producer's specific product list.
                        self.producers_products[producer].remove(product)
                        break
            else: return False # Functional Utility: Returns False if the product is not available.

        # Functional Utility: Adds the product to the consumer's cart.
        self.all_carts[cart_id].append(product)
        return True


    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a specified shopping cart and returns it to the marketplace inventory.

        Args:
            cart_id (int): The ID of the cart to remove the product from.
            product (str): The name of the product to remove.

        Pre-condition: The product must exist in the specified cart.
        Post-condition: The product is removed from the cart and added back to the marketplace's global inventory.
        """
        # Functional Utility: Removes the product from the specified cart.
        self.all_carts[cart_id].remove(product)
        # Functional Utility: Returns the product to the global marketplace inventory.
        self.products_in_marketplace.append(product)

    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart, returning the list of products in it.
        Note: In this simulation, placing an order does not empty the cart or
        remove items from the marketplace (as they were already removed when added to cart).

        Args:
            cart_id (int): The ID of the cart to place the order for.

        Returns:
            list: A list of products that were in the placed order.
        """
        return self.all_carts[cart_id]  


from threading import Thread
from time import sleep

class Producer(Thread):
    """
    Represents a producer thread that continuously publishes products to the marketplace.
    Each producer has a specific list of products it offers and a wait time between republishing.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of tuples, each containing (product_name, quantity, publish_interval).
            marketplace (Marketplace): The shared marketplace instance to publish to.
            republish_wait_time (float): Time in seconds to wait before retrying to publish if the queue is full.
            **kwargs: Arbitrary keyword arguments passed to the Thread.__init__.
        """
        
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs
        # Functional Utility: Registers with the marketplace to get a unique producer ID.
        self.producer_id = self.marketplace.register_producer()


    def run(self):
        """
        The main execution loop for the Producer thread.
        It continuously attempts to publish its products to the marketplace,
        respecting its queue size limit and specified intervals.
        """
        # Invariant: The producer continuously attempts to publish products.
        while True:
            # Block Logic: Iterates through each product defined for this producer.
            for product in self.products:
                quantity = 0
                # Invariant: Continues publishing the current product until its desired quantity is met.
                while quantity < product[1]:
                    # Functional Utility: Attempts to publish a product to the marketplace.
                    if self.marketplace.publish(str(self.producer_id), product[0]):
                        # Functional Utility: Waits for the specified interval after a successful publish.
                        sleep(product[2])
                        quantity += 1
                    else:
                        # Functional Utility: Waits for a specified duration before retrying if publishing fails (e.g., queue full).
                        sleep(self.republish_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Base class for a product in the marketplace.
    Uses Python's `dataclasses` decorator for automatic __init__, __repr__, etc.
    """
    
    name: str  # The name of the product.
    price: int # The price of the product.


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Represents a specific type of Product: Tea.
    Inherits from `Product` and adds a 'type' attribute.
    """
    
    type: str  # The type of tea (e.g., "Green", "Black", "Herbal").


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Represents a specific type of Product: Coffee.
    Inherits from `Product` and adds 'acidity' and 'roast_level' attributes.
    """
    
    acidity: str      # The acidity level of the coffee.
    roast_level: str  # The roast level of the coffee.
