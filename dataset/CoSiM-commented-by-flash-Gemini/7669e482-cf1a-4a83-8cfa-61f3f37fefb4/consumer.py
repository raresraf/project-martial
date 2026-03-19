"""
This module implements a multi-threaded e-commerce simulation, defining
`Consumer` and `Producer` roles that interact with a `Marketplace`.
It includes classes for managing product listings (`PublishedProduct`, `ProductsList`)
and shopping carts (`Cart`), demonstrating concurrent operations with
thread-safe mechanisms like locks and queues.
"""

from threading import Thread, Lock, current_thread
from time import sleep


class Consumer(Thread):
    """
    Represents a consumer in the e-commerce simulation.

    Each consumer operates as a separate thread, managing a list of shopping carts.
    It attempts to add/remove products to/from its carts via the marketplace
    and retries failed operations after a specified wait time.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a new Consumer instance.

        Args:
            carts (list): A list of shopping lists, where each list contains
                          operations (add/remove) for products.
            marketplace (Marketplace): The marketplace instance to interact with.
            retry_wait_time (int): The time in seconds to wait before retrying
                                   failed cart operations.
            **kwargs: Arbitrary keyword arguments to be passed to the Thread constructor.
        """
        Thread.__init__(self, name=kwargs["name"], kwargs=kwargs)
        
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
    def run(self):
        """
        The main execution loop for the Consumer thread.

        Iterates through each shopping list (cart) provided, attempts to perform
        the specified add/remove operations for products, retries failed
        operations, and finally places the order for completed carts.
        """
        for lista_cumparaturi in self.carts: # Iterates through each shopping list (cart) assigned to this consumer.
            
            cart_id = self.marketplace.new_cart() # Creates a new cart in the marketplace for the current shopping list.
            
            failed = True
            # Block Logic: Retries cart operations until all products in the current shopping list are successfully processed.
            while failed:
                failed = False
                for operatie in lista_cumparaturi: # Iterates through each operation (add/remove) in the current shopping list.
                    
                    # 'cantitate' is the quantity of the product for the current operation.
                    cantitate = operatie["quantity"]
                    # Block Logic: Processes each unit of product for the current operation.
                    for _ in range(cantitate):
                        # Block Logic: Handles "add" operations.
                        if operatie["type"] == "add":
                            # Pre-condition: The product can be added to the cart.
                            if self.marketplace.add_to_cart(cart_id, operatie["product"]):
                                # If successful, decrements the remaining quantity for this operation.
                                operatie["quantity"] = operatie["quantity"] - 1
                            else:
                                # If adding fails, marks the operation as failed and breaks to retry the entire cart.
                                failed = True
                                break
                        # Block Logic: Handles "remove" operations.
                        elif operatie["type"] == "remove":
                            # Pre-condition: The product can be removed from the cart.
                            if self.marketplace.remove_from_cart(cart_id, operatie["product"]):
                                # If successful, decrements the remaining quantity for this operation.
                                operatie["quantity"] = operatie["quantity"] - 1
                            else:
                                # If removing fails, marks the operation as failed and breaks to retry the entire cart.
                                failed = True
                                break
            # Block Logic: If any operation in the cart failed, waits for a specified time before retrying.
            if failed:
                sleep(self.retry_wait_time)
            # Block Logic: If all operations in the cart were successful, places the order.
            else:
                self.marketplace.place_order(cart_id)
                
                
                


class PublishedProduct:
    """
    A wrapper class for a product that includes a reservation status.

    This is used within the Marketplace to track products that have been
    reserved by a consumer but not yet purchased.
    """
    def __init__(self, product):
        """
        Initializes a new PublishedProduct instance.

        Args:
            product (Any): The product data being wrapped.
        """
        self.product = product
        self.reserved = False

    
    def __eq__(self, obj):
        """
        Determines if two PublishedProduct objects are equal.

        Equality is based on both the wrapped product data and the
        reservation status.

        Args:
            obj (object): The other object to compare against.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        ret = isinstance(obj, PublishedProduct) and self.reserved == obj.reserved
        return ret and obj.product == self.product

class ProductsList:
    """
    A thread-safe list for managing `PublishedProduct` instances, typically
    representing products available from a producer.

    It enforces a maximum size and provides methods for adding, reserving,
    cancelling reservations, and removing products.
    """
    def __init__(self, maxsize):
        """
        Initializes a new ProductsList instance.

        Args:
            maxsize (int): The maximum number of products this list can hold.
        """
        self.lock = Lock() # A lock to ensure thread-safe access to the internal list.
        self.list = [] # The internal list storing PublishedProduct objects.
        self.maxsize = maxsize

    def put(self, item):
        """
        Adds an item to the products list.

        Args:
            item (PublishedProduct): The item to add to the list.

        Returns:
            bool: True if the item was added successfully, False if the
                  list is at its maximum capacity.
        """
        with self.lock: # Ensures thread-safe access to the list.
            # Block Logic: Checks if the list has reached its maximum capacity.
            if self.maxsize == len(self.list):
                return False # Cannot add item if at max capacity.
            self.list.append(item) # Appends the item to the list.
        return True # Returns True if the item was successfully added.

    def rezerva(self, item):
        """
        Attempts to reserve a product in the list.

        Args:
            item (Any): The product data to reserve.

        Returns:
            bool: True if the product was found and successfully reserved,
                  False otherwise (e.g., product not found or already reserved).
        """
        item = PublishedProduct(item) # Wraps the product data into a PublishedProduct object.
        with self.lock: # Ensures thread-safe access to the list.
            # Block Logic: Checks if the product exists in the list.
            if item in self.list:
                # If found, marks the product as reserved and returns True.
                self.list[self.list.index(item)].reserved = True
                return True
        return False # Returns False if the product was not found or could not be reserved.

    def anuleaza_rezervarea(self, item):
        """
        Cancels the reservation for a previously reserved product.

        Args:
            item (Any): The product data for which to cancel the reservation.
        """
        item = PublishedProduct(item) # Wraps the product data into a PublishedProduct object.
        item.reserved = True # Sets reserved to True to match the reserved item in the list for lookup.
        with self.lock: # Ensures thread-safe access to the list.
            # Block Logic: Finds the reserved product in the list and sets its reserved status to False.
            self.list[self.list.index(item)].reserved = False

    def remove(self, item):
        """
        Removes a reserved product from the list.

        Args:
            item (Any): The product data to remove.

        Returns:
            Any: The product data of the removed item.
        """
        product = PublishedProduct(item) # Wraps the product data into a PublishedProduct object.
        product.reserved = True # Sets reserved to True to match the reserved item in the list for lookup.
        with self.lock: # Ensures thread-safe access to the list.
            # Block Logic: Finds and removes the specified reserved product from the list.
            self.list.remove(product)
            return item # Returns the original product data.

class Cart:
    """
    Represents a shopping cart, holding products added by a consumer.

    Each item in the cart is stored as a tuple containing the product and
    the ID of the producer from which it was reserved.
    """

    def __init__(self):
        """
        Initializes an empty shopping cart.
        """
        self.products = [] # List to store products in the cart, each as (product_data, producer_id).

    def add_product(self, product, producer_id):
        """
        Adds a product to the cart along with its producing entity's ID.

        Args:
            product (Any): The product data to add.
            producer_id (int): The ID of the producer that published this product.
        """
        self.products.append((product, producer_id))

    def remove_product(self, product):
        """
        Removes a specific product from the cart.

        Args:
            product (Any): The product data to remove from the cart.

        Returns:
            int or None: The producer ID of the removed product if found,
                         otherwise None.
        """
        # Block Logic: Iterates through the products in the cart to find a match.
        for item in self.products:
            if item[0] == product: # Compares the product data.
                self.products.remove(item) # Removes the found product.
                return item[1] # Returns the producer ID of the removed product.

        return None # Returns None if the product was not found in the cart.

    def get_products(self):
        """
        Retrieves all products currently in the cart.

        Returns:
            list: A list of (product_data, producer_id) tuples.
        """
        return self.products

class Marketplace:
    """
    Acts as the central hub for producers and consumers in the e-commerce simulation.

    It manages product listings from various producers, handles cart creation,
    adding/removing items from carts, and processing orders. All operations
    are designed to be thread-safe.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes a new Marketplace instance.

        Args:
            queue_size_per_producer (int): The maximum number of products
                                           a single producer can have listed
                                           in the marketplace at any time.
        """
        self.print_lock = Lock() # Lock to synchronize print statements to avoid interleaving.
        self.queue_size_per_producer = queue_size_per_producer
        self.producer_queues = {} # Dictionary to store ProductsList for each producer.

        self.generator_id_producator = 0 # Counter for generating unique producer IDs.
        self.generator_id_producator_lock = Lock() # Lock for thread-safe access to producer ID generator.

        self.carts = {} # Dictionary to store Cart objects, mapped by cart_id.
        self.cart_id_generator = 0 # Counter for generating unique cart IDs.
        self.cart_id_generator_lock = Lock() # Lock for thread-safe access to cart ID generator.

    def register_producer(self):
        """
        Registers a new producer with the marketplace and assigns a unique ID.

        This also initializes a `ProductsList` for the new producer.

        Returns:
            int: The unique ID assigned to the registered producer.
        """
        id_producator = None
        with self.generator_id_producator_lock: # Ensures thread-safe generation of producer IDs.
            # Block Logic: Assigns a new unique producer ID.
            id_producator = self.generator_id_producator
            self.generator_id_producator += 1 # Increments the counter for the next producer.
            # Block Logic: Initializes a new ProductsList for the registered producer.
            self.producer_queues[id_producator] = ProductsList(self.queue_size_per_producer)

        return id_producator

    def publish(self, producer_id, product):
        """
        Publishes a product from a specific producer to the marketplace.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (Any): The product data to publish.

        Returns:
            bool: True if the product was published successfully, False if
                  the producer's queue is full.
        """
        return self.producer_queues[producer_id].put(PublishedProduct(product))

    def new_cart(self):
        """
        Creates a new, empty shopping cart and assigns it a unique ID.

        Returns:
            int: The unique ID of the newly created cart.
        """
        with self.cart_id_generator_lock: # Ensures thread-safe generation of cart IDs.
            # Block Logic: Assigns a new unique cart ID.
            current_cart_id = self.cart_id_generator
            self.cart_id_generator += 1 # Increments the counter for the next cart.
            
            # Block Logic: Initializes a new Cart object and stores it with its ID.
            self.carts[current_cart_id] = Cart()

            return current_cart_id

    def add_to_cart(self, cart_id, product):
        """
        Attempts to add a product to a specified shopping cart.

        This involves finding an available instance of the product from any
        producer and reserving it.

        Args:
            cart_id (int): The ID of the cart to which the product should be added.
            product (Any): The product data to add to the cart.

        Returns:
            bool: True if the product was successfully added (reserved),
                  False otherwise.
        """
        producers_num = 0
        with self.generator_id_producator_lock: # Ensures thread-safe access to the total number of producers.
            producers_num = self.generator_id_producator

        # Block Logic: Iterates through all registered producers to find an available instance of the product.
        for producer_id in range(producers_num):
            # Pre-condition: The producer has the product and it can be reserved.
            if self.producer_queues[producer_id].rezerva(product):
                self.carts[cart_id].add_product(product, producer_id) # Adds the product to the cart with the producer's ID.
                return True # Returns True if successfully reserved and added.

        return False # Returns False if the product could not be found or reserved from any producer.

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a specified shopping cart and cancels its reservation.

        Args:
            cart_id (int): The ID of the cart from which the product should be removed.
            product (Any): The product data to remove from the cart.

        Returns:
            bool: True if the product was successfully removed and its reservation
                  cancelled, False otherwise (e.g., product not in cart).
        """
        # Block Logic: Attempts to remove the product from the cart and retrieve its producer ID.
        producer_id = self.carts[cart_id].remove_product(product)
        # Pre-condition: The product was successfully removed from the cart.
        if producer_id is None:
            return False # Product not found in cart.
        # Block Logic: Cancels the reservation of the product at the corresponding producer's queue.
        self.producer_queues[producer_id].anuleaza_rezervarea(product)
        return True

    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart.

        This involves permanently removing the reserved products from their
        respective producers' queues and printing a confirmation message.

        Args:
            cart_id (int): The ID of the cart for which to place the order.

        Returns:
            list: A list of the products that were successfully purchased.
        """
        lista = list() # List to hold the products that are successfully purchased.
        # Block Logic: Iterates through each product in the specified cart.
        for (produs, producer_id) in self.carts[cart_id].get_products():
            # Block Logic: Removes the product from the producer's queue, marking it as sold.
            lista.append(self.producer_queues[producer_id].remove(produs))
            with self.print_lock: # Ensures thread-safe printing to avoid interleaved output.
                print(f"{current_thread().getName()} bought {produs}")
        return lista


class Producer(Thread):
    """
    Represents a producer in the e-commerce simulation.

    Each producer operates as a separate thread, continuously producing
    and publishing products to the marketplace. It waits for a specified
    time between productions and retries publishing if the marketplace
    queue for its products is full.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a new Producer instance.

        Args:
            products (list): A list of tuples, where each tuple contains
                             (product_data, quantity_to_produce, production_time).
            marketplace (Marketplace): The marketplace instance to interact with.
            republish_wait_time (int): The time in seconds to wait before retrying
                                       to publish a product if the marketplace queue is full.
            **kwargs: Arbitrary keyword arguments to be passed to the Thread constructor.
        """
        Thread.__init__(self, name=kwargs["name"], daemon=kwargs["daemon"], kwargs=kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        The main execution loop for the Producer thread.

        It continuously registers itself with the marketplace, then iterates
        through its product list, producing and publishing each product.
        If publishing fails (e.g., due to a full queue), it retries after a delay.
        """
        producer_id = self.marketplace.register_producer() # Registers with the marketplace to get a unique producer ID.
        
        # Block Logic: Main production loop, runs indefinitely.
        while True:
            # Block Logic: Iterates through each type of product this producer is responsible for.
            for (product, cantitate, production_time) in self.products:
                
                sleep(production_time) # Simulates the time taken to produce one batch of a product.
                
                # Block Logic: Attempts to publish the specified quantity of the current product.
                for _ in range(cantitate):
                    # Block Logic: Continuously attempts to publish the product until successful.
                    while not self.marketplace.publish(producer_id, product):
                        sleep(self.republish_wait_time) # Waits before retrying if publishing fails.