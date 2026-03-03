"""
This module implements a producer-consumer marketplace simulation using threading.

It defines:
- Consumer: Represents a buyer that interacts with the marketplace to add/remove products from a cart and place orders.
- Marketplace: Manages product inventories from producers, handles cart creation, and processes add/remove operations with thread-safe mechanisms.
- Producer: Represents a seller that publishes products to the marketplace.
- Product, Tea, Coffee: Dataclasses for defining various product types.
"""

from threading import Thread, Lock
from time import sleep

# The Queue module is implicitly used in Marketplace for product management.
# from Queue import Queue # Not explicitly imported here, but conceptually part of the marketplace


class Consumer(Thread):
    """
    Represents a consumer (buyer) in the marketplace simulation.

    Each consumer operates as a separate thread, executing a series of shopping tasks
    (adding and removing products from a cart) and eventually placing an order.
    It handles retries if a marketplace operation fails.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a new Consumer thread.

        Args:
            carts (list): A list of cart definitions. Each cart is a list of tasks,
                          where each task is a dictionary like {'type': 'add'/'remove',
                          'product': product_obj, 'quantity': int}.
            marketplace (Marketplace): The shared marketplace instance to interact with.
            retry_wait_time (float): The time (in seconds) to wait before retrying a failed operation.
            **kwargs: Arbitrary keyword arguments, primarily used to set the thread's 'name'.
        """
        Thread.__init__(self) # Initialize the base Thread class.
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs.get('name') # Get thread name for printing, defaults to None if not provided.

    def run(self):
        """
        The main execution method for the Consumer thread.

        It registers itself by incrementing the consumer count in the marketplace.
        Then, for each defined shopping cart:
        1. Creates a new cart in the marketplace.
        2. Iterates through the tasks in the cart (add or remove products).
        3. If an operation (`add_to_cart` or `remove_from_cart`) fails, it retries
           after `retry_wait_time`.
        4. Once all tasks for a cart are processed, it places the order and prints
           the purchased items.
        Finally, it decrements the consumer count in the marketplace upon completion.
        """
        # Block Logic: Increment the count of active consumers in the marketplace.
        # This is used by producers to determine when to shut down.
        if self.marketplace.nr_of_consumers == -1: # First consumer initializes count
            self.marketplace.nr_of_consumers = 1
        else:
            self.marketplace.nr_of_consumers += 1

        # Block Logic: Process each shopping cart defined for this consumer.
        for cart in self.carts:
            new_cart_id = self.marketplace.new_cart() # Request a new cart ID from the marketplace.
            
            # Block Logic: Execute each task (add/remove product) within the current cart.
            for task in cart:
                i = 0 # Counter for quantity of product to be processed.
                while i < task.get('quantity'): # Loop until the desired quantity for the task is processed.
                    check = False # Flag to indicate if the operation was successful.
                    if task.get('type') == "add":   
                        # Attempt to add product to cart.
                        check = self.marketplace.add_to_cart(new_cart_id, task.get('product'))
                    elif task.get('type') == "remove":
                        # Attempt to remove product from cart.
                        check = self.marketplace.remove_from_cart(new_cart_id, task.get('product'))
                    
                    if check == False:
                        # If the operation failed (e.g., product not available), wait and retry.
                        sleep(self.retry_wait_time)
                    else:
                        # If successful, move to the next quantity unit.
                        i += 1

            # Block Logic: Place the order and print the purchased products.
            # Iterates through the products in the finalized cart.
            for prod in self.marketplace.place_order(new_cart_id):
                print("%s bought %s" % (self.name, prod))
        
        # Decrement the count of active consumers as this consumer thread finishes its tasks.
        self.marketplace.nr_of_consumers -= 1
        

class Marketplace:
    """
    Manages the central logic for product exchange between producers and consumers.

    It handles producer registration, product publishing, shopping cart creation,
    and thread-safe addition/removal of products from carts.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products each producer
                                           can have in its queue at any given time.
        """
        self.last_producer_id = -1          # Counter for assigning unique producer IDs.
        self.last_cart_id = -1              # Counter for assigning unique cart IDs.
        self.prod_queue = []                # List of lists, where each inner list is a product queue for a producer.
        self.all_carts = []                 # List of lists, where each inner list represents a consumer's shopping cart.
        self.producerAndProduct = []        # Stores (producer_id, product) for products currently in carts,
                                            # to facilitate returning products to the correct producer's queue on removal.
        
        # Locks to ensure thread-safe access to shared resources within the marketplace.
        self.addToCart_lock = Lock()        # Protects add_to_cart operations.
        self.removeFromCart_lock = Lock()   # Protects remove_from_cart operations.
        self.lastProdId_lock = Lock()       # Protects access to last_producer_id during registration.
        self.publish_lock = Lock()          # Protects publish operations.
        self.new_cart_lock = Lock()         # Protects new_cart operations.


        self.nr_of_consumers = -1           # Tracks the number of active consumer threads.
                                            # Initialized to -1 to distinguish from 0 active consumers.
        self.queue_size_per_producer = queue_size_per_producer # Max products per producer queue.

    def register_producer(self):
        """
        Registers a new producer with the marketplace and assigns it a unique ID.

        Returns:
            int: The unique ID assigned to the registered producer.
        """
        self.lastProdId_lock.acquire()      # Acquire lock to safely increment producer ID.
        self.last_producer_id += 1          # Increment the producer ID counter.
        self.prod_queue.append([])          # Create an empty product queue for the new producer.
        self.lastProdId_lock.release()      # Release lock.
        return self.last_producer_id

    def publish(self, producer_id, product):
        """
        Publishes a product from a producer to its respective queue in the marketplace.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (Product): The product object to be published.

        Returns:
            bool: True if the product was successfully published (queue was not full), False otherwise.
        """
        self.publish_lock.acquire()         # Acquire lock to safely modify product queues.
        # Check if the producer's queue has space.
        if(len(self.prod_queue[producer_id]) < self.queue_size_per_producer):
            self.prod_queue[producer_id].append(product) # Add product to the queue.
            self.publish_lock.release()     # Release lock.
            return True
        else:
            self.publish_lock.release()     # Release lock.
            return False

    def new_cart(self):
        """
        Creates a new empty shopping cart and assigns it a unique ID.

        Returns:
            int: The unique ID assigned to the new cart.
        """
        self.new_cart_lock.acquire()        # Acquire lock to safely increment cart ID.
        self.last_cart_id += 1              # Increment the cart ID counter.
        
        self.all_carts.append([])           # Add a new empty cart to the list of all carts.
        self.new_cart_lock.release()        # Release lock.
        return self.last_cart_id

    def add_to_cart(self, cart_id, product):
        """
        Attempts to add a specified product to a consumer's cart.

        It searches all producer queues for the product. If found, it removes
        the product from the producer's queue, adds it to the consumer's cart,
        and records the producer-product association for potential returns.

        Args:
            cart_id (int): The ID of the cart to which the product should be added.
            product (Product): The product object to add.

        Returns:
            bool: True if the product was found and added to the cart, False otherwise.
        """
        self.addToCart_lock.acquire()       # Acquire lock to safely modify carts and product queues.
        # Iterate through all producer queues.
        for i in range(len(self.prod_queue)):
            # Iterate through products in the current producer's queue.
            for j in range(len(self.prod_queue[i])):
                # If the product is found in a producer's queue.
                if self.prod_queue[i][j] == product:
                    self.all_carts[cart_id].append(product)     # Add product to the consumer's cart.
                    
                    self.producerAndProduct.append((i, product)) # Record the association for return tracking.
                    
                    self.prod_queue[i].remove(product)          # Remove product from the producer's queue.
                    self.addToCart_lock.release()               # Release lock.
                    return True # Successfully added.
        self.addToCart_lock.release()                           # Release lock.
        return False # Product not found in any queue.

    def remove_from_cart(self, cart_id, product):
        """
        Removes a specified product from a consumer's cart and returns it to its original producer's queue.

        Args:
            cart_id (int): The ID of the cart from which the product should be removed.
            product (Product): The product object to remove.
        """
        self.removeFromCart_lock.acquire()  # Acquire lock to safely modify carts and product queues.
        # Iterate through items in the specified cart.
        for i in range(len(self.all_carts[cart_id])):
            if self.all_carts[cart_id][i] == product:
                self.all_carts[cart_id].remove(product)     # Remove product from the cart.
                
                # Block Logic: Find the original producer and return the product to their queue.
                for j in range(len(self.producerAndProduct)):
                    (index, searchProduct) = self.producerAndProduct[j]
                    if(searchProduct == product): # If this is the matching product.
                        self.prod_queue[index].append(product)  # Return product to producer's queue.
                        
                        self.producerAndProduct.pop(j)          # Remove association.
                        break # Found and processed, exit inner loop.
                break # Found and processed, exit outer loop.
        self.removeFromCart_lock.release()                      # Release lock.
                    
    def place_order(self, cart_id):
        """
        Retrieves the contents of a specific shopping cart.

        Args:
            cart_id (int): The ID of the cart to retrieve.

        Returns:
            list: A list of products in the specified cart.
        """
        return self.all_carts[cart_id]


class Producer(Thread):
    """
    Represents a producer (seller) in the marketplace simulation.

    Each producer operates as a separate thread, continuously publishing products
    to the marketplace's queues. It handles delays for republishing.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a new Producer thread.

        Args:
            products (list): A list of product definitions. Each element is a tuple
                             `(product_obj, quantity_to_publish, publish_wait_time_per_unit)`.
            marketplace (Marketplace): The shared marketplace instance to interact with.
            republish_wait_time (float): The time (in seconds) to wait before retrying
                                         to publish if the queue is full.
            **kwargs: Arbitrary keyword arguments, used to set 'daemon' status and 'name'.
        """
        # Initialize the base Thread class. Set as daemon if specified.
        Thread.__init__(self, daemon=kwargs.get("daemon", False))
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.name = kwargs.get('name') # Get thread name for printing, defaults to None if not provided.

    def run(self):
        """
        The main execution method for the Producer thread.

        It registers itself with the marketplace to get a unique producer ID.
        Then, it continuously attempts to publish its defined products.
        For each product:
        1. It attempts to publish the specified quantity.
        2. If `marketplace.publish` returns `True`, it waits `publish_wait_time_per_unit`.
        3. If `marketplace.publish` returns `False` (queue full), it waits `republish_wait_time`.
        The producer breaks its loop and terminates if there are no active consumers.
        """
        id = self.marketplace.register_producer() # Register with the marketplace to get a producer ID.

        while True: # Continuous loop for publishing products.
            # Block Logic: Iterate through each product type this producer offers.
            for prod in self.products:
                nr_of_prod = 0 # Counter for the number of units of the current product published.
                while nr_of_prod < prod[1]: # Loop until the desired quantity for this product type is published.
                    check = self.marketplace.publish(id, prod[0]) # Attempt to publish one unit of the product.
                    if check:
                        # If publishing was successful, wait for the specified publish time per unit.
                        sleep(prod[2])
                        nr_of_prod += 1 # Increment published quantity.
                    else:
                        # If publishing failed (e.g., producer's queue is full), wait and retry.
                        sleep(self.republish_wait_time)
            
            # Block Logic: Check for consumer activity and terminate if no consumers are left.
            if self.marketplace.nr_of_consumers == 0: # If there are no more active consumers.
                break # Exit the main publishing loop and terminate the producer thread.
                        
"""
This section defines dataclasses for product types.
These classes are typically found in a separate file (e.g., product.py)
but are included here as part of the content.
"""
from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Base dataclass representing a generic product.

    Attributes:
        name (str): The name of the product.
        price (int): The price of the product.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Dataclass representing a type of tea, inheriting from Product.

    Attributes:
        type (str): The type of tea (e.g., "Green", "Black", "Herbal").
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Dataclass representing a type of coffee, inheriting from Product.

    Attributes:
        acidity (str): The acidity level of the coffee.
        roast_level (str): The roast level of the coffee.
    """
    acidity: str
    roast_level: str
