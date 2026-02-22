

"""
@brief This module implements a simulation of a producer-consumer system using a `Marketplace` to mediate transactions.
@details It defines `Consumer` threads that purchase products, `Producer` threads that supply products,
and a `Marketplace` class that manages product queues, carts, and order placements with thread-safe operations.
This simulation demonstrates concurrent access to shared resources and synchronization challenges
in a multi-threaded environment.
"""

from threading import Thread
from time import sleep

class Consumer(Thread):
    """
    @brief Simulates a buyer in the marketplace, extending `threading.Thread`.
    @details Each Consumer instance operates independently, creating shopping carts,
    adding and removing products based on its defined `carts` list, and finally
    placing orders. It interacts with the `Marketplace` to perform these actions,
    handling retries if products are not immediately available.
    @architectural_intent Represents an active entity in the marketplace, driving demand
    for products and interacting with the `Marketplace`'s cart management and ordering system.
    """
    
    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a new Consumer thread.
        @param carts (list): A list of shopping carts, where each cart is a list of operations
                             (add/remove product, quantity).
        @param marketplace (Marketplace): A reference to the shared Marketplace instance.
        @param retry_wait_time (float): The time (in seconds) to wait before retrying an operation
                                        if a product is not available.
        @param kwargs: Arbitrary keyword arguments to be passed to the `Thread` constructor (e.g., `name`).
        """
        Thread.__init__(self, **kwargs) # Initialize the base Thread class.
        self.carts = carts # List of shopping carts this consumer will process.
        self.name = kwargs['name'] # The name of this consumer thread.
        self.marketplace = marketplace # Reference to the shared Marketplace.
        self.retry_wait_time = retry_wait_time # Time to wait before retrying an operation.

    def run(self):
        """
        @brief The main execution logic for the Consumer thread.
        @details This method iterates through each shopping cart assigned to the consumer.
        For each cart, it performs a series of 'add' or 'remove' operations for products.
        'Add' operations involve retries if the product is not immediately available.
        Once all operations for a cart are complete, an order is placed, and the purchased
        products are printed.
        @block_logic Processes a sequence of shopping carts, performing product transactions.
        @pre_condition `self.carts` contains valid cart definitions, `self.marketplace` is
                       an active `Marketplace` instance.
        @invariant Each cart is processed sequentially, and operations are performed with
                   potential retries until successful.
        """
        # Block Logic: Iterate through each cart assigned to this consumer.
        # Invariant: Each cart will be processed from creation to order placement.
        for cart in self.carts:
            # Functional Utility: Create a new cart in the marketplace and obtain its ID.
            cart_id = self.marketplace.new_cart()

            # Block Logic: Process each operation (add or remove product) within the current cart.
            # Invariant: All operations in a cart are attempted, with 'add' operations retried until success.
            for operation in cart:
                # Block Logic: Perform the specified quantity of an operation.
                # Invariant: The product operation is attempted 'quantity' times.
                for _ in range(operation['quantity']):
                    if operation['type'] == 'add':
                        # Block Logic: Attempt to add a product to the cart, retrying if unavailable.
                        # Invariant: The product is eventually added to the cart, or the loop continues indefinitely if always unavailable.
                        while not self.marketplace.add_to_cart(cart_id, operation['product']):
                            sleep(self.retry_wait_time) # Wait before retrying.
                    elif operation['type'] == 'remove':
                        # Functional Utility: Remove a product from the cart.
                        self.marketplace.remove_from_cart(cart_id, operation['product'])

            # Functional Utility: Place the order for the fully populated cart.
            order = self.marketplace.place_order(cart_id)
            # Block Logic: Print the details of the products purchased by this consumer.
            # Invariant: Each product successfully ordered is logged to console.
            for product in order:
                print("%s bought %s" % (self.name, product))



from threading import Lock

class Marketplace:
    """
    @brief Acts as a central hub for producers and consumers to interact in a simulated e-commerce environment.
    @details This class manages product availability from various producers, handles shopping cart operations
    (creation, adding/removing products), and processes final orders. It is designed to be thread-safe,
    using multiple locks to protect critical sections and ensure data consistency during concurrent access
    from multiple producer and consumer threads.
    @architectural_intent Provides a thread-safe, synchronized environment for managing product flow
    from multiple producers to multiple consumers, abstracting the complexities of concurrent access
    and maintaining the integrity of inventory and cart data.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes a new Marketplace instance.
        @param queue_size_per_producer (int): The maximum number of products each producer can have
                                             in the marketplace at any given time.
        """
        self.queue_size_per_producer = queue_size_per_producer # Max products per producer.
        self.last_producer_id = 0 # Counter for assigning unique producer IDs.
        self.last_cart_id = 0     # Counter for assigning unique cart IDs.
        
        self.products_per_producer = {} # Dictionary: producer_id -> list of products published by that producer.
        
        self.carts = {}             # Dictionary: cart_id -> list of (producer_id, product) in that cart.
        self.cart_lock = Lock()     # Lock to protect `last_cart_id` and `carts` dictionary.
        self.producer_id_lock = Lock() # Lock to protect `last_producer_id` and `products_per_producer` dictionary.
        self.add_to_cart_lock = Lock() # Lock to protect `add_to_cart` method's critical section.

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace.
        @details Assigns a unique producer ID and initializes an empty product list for it.
        This operation is thread-safe using `producer_id_lock`.
        @return int: The newly assigned unique producer ID.
        @block_logic Assigns a unique ID to a new producer and prepares its product inventory.
        @pre_condition `self.producer_id_lock` is an initialized Lock.
        @invariant `self.last_producer_id` is incremented, and a new entry for the producer
                   is created in `self.products_per_producer`.
        """
        with self.producer_id_lock: # Acquire lock to ensure thread-safe ID assignment and dictionary modification.
            self.last_producer_id += 1 # Increment to get a new unique producer ID.
            self.products_per_producer[self.last_producer_id] = [] # Initialize an empty list for this producer's products.
            return self.last_producer_id

    def publish(self, producer_id, product):
        """
        @brief Publishes a product from a producer to the marketplace.
        @details A product can only be published if the producer's queue is not full.
        @param producer_id (int): The ID of the producer publishing the product.
        @param product (object): The product to be published.
        @return bool: True if the product was successfully published, False if the producer's queue is full.
        @block_logic Adds a product to a producer's inventory if capacity allows.
        @pre_condition `producer_id` is a valid, registered producer.
        @invariant If the product is published, it is added to `self.products_per_producer[producer_id]`.
        """
        # Block Logic: Check if the producer's queue has reached its maximum capacity.
        # Invariant: Product is added only if the queue is not full.
        if len(self.products_per_producer[producer_id]) == self.queue_size_per_producer:
            return False # Producer's queue is full, cannot publish.

        self.products_per_producer[producer_id].append(product) # Add the product to the producer's list.
        return True # Product successfully published.

    def new_cart(self):
        """
        @brief Creates a new empty shopping cart in the marketplace.
        @details Assigns a unique cart ID and initializes an empty list for its contents.
        This operation is thread-safe using `cart_lock`.
        @return int: The newly assigned unique cart ID.
        @block_logic Assigns a unique ID to a new cart and initializes its content.
        @pre_condition `self.cart_lock` is an initialized Lock.
        @invariant `self.last_cart_id` is incremented, and a new entry for the cart
                   is created in `self.carts`.
        """
        with self.cart_lock: # Acquire lock to ensure thread-safe ID assignment and dictionary modification.
            self.last_cart_id += 1 # Increment to get a new unique cart ID.
            self.carts[self.last_cart_id] = [] # Initialize an empty list for this cart's products.
            return self.last_cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a product to a specific shopping cart.
        @details This method searches for the product across all producers. If found,
        it moves the product from the producer's inventory to the specified cart.
        This operation is thread-safe using `add_to_cart_lock` to ensure atomicity
        of product transfer.
        @param cart_id (int): The ID of the cart to add the product to.
        @param product (object): The product to add.
        @return bool: True if the product was found and added to the cart, False otherwise.
        @block_logic Transfers a product from a producer's inventory to a consumer's cart.
        @pre_condition `cart_id` is a valid, existing cart ID. `self.products_per_producer`
                       and `self.carts` are properly initialized.
        @invariant If successful, `product` is removed from a producer's list and added to the cart.
        """
        # Block Logic: Iterate through all producers to find the requested product.
        # Invariant: Product is searched across all producer inventories.
        for producer_id, products in self.products_per_producer.items():
            # Block Logic: Acquire lock to ensure thread-safe modification of product inventories and carts.
            # Invariant: Product removal and addition are atomic for this operation.
            with self.add_to_cart_lock:
                if product in products: # Check if the product is available from this producer.
                    products.remove(product) # Remove product from producer's inventory.
                    self.carts[cart_id].append((producer_id, product)) # Add product to the cart.
                    return True # Product successfully added.

        return False # Product not found in any producer's inventory.

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a specific shopping cart and returns it to the producer.
        @details This method finds the product within the cart, identifies its original producer,
        removes it from the cart, and returns it to the producer's inventory.
        @param cart_id (int): The ID of the cart from which to remove the product.
        @param product (object): The product to remove.
        @block_logic Transfers a product from a consumer's cart back to its original producer's inventory.
        @pre_condition `cart_id` is a valid, existing cart ID. `product` is present in the specified cart.
                       `self.products_per_producer` and `self.carts` are properly initialized.
        @invariant `product` is removed from the cart and re-added to the correct producer's list.
        """
        producer_id = 0 # Initialize producer_id to a default value.
        # Block Logic: Find the producer ID associated with the product in the cart.
        # Invariant: `producer_id` will hold the correct ID if the product is found.
        for cart_producer_id, cart_product in self.carts[cart_id]:
            if cart_product == product:
                producer_id = cart_producer_id
                break # Found the producer, exit loop.

        # Functional Utility: Remove the product from the cart.
        self.carts[cart_id].remove((producer_id, product))
        # Functional Utility: Return the product to its original producer's inventory.
        self.products_per_producer[producer_id].append(product)

    def place_order(self, cart_id):
        """
        @brief Places an order for all products in a given cart.
        @details This method essentially "finalizes" the cart by returning a list of all
        products that were successfully added to it. The products are already moved
        from producers' inventories, so this just represents the final collection.
        @param cart_id (int): The ID of the cart for which to place the order.
        @return list: A list of products (objects) that are in the specified cart.
        @block_logic Retrieves the final list of products from a shopping cart.
        @pre_condition `cart_id` is a valid, existing cart ID.
        @invariant The contents of the cart are returned as a simple list of products.
        """
        return [product for _, product in self.carts[cart_id]]



from threading import Thread
from time import sleep

class Producer(Thread):
    """
    @brief Simulates a supplier in the marketplace, extending `threading.Thread`.
    @details Each Producer instance registers itself with the `Marketplace`,
    then continuously attempts to publish its predefined list of products.
    It handles republishing products if the marketplace queue is full, waiting
    for a specified time before retrying.
    @architectural_intent Represents an active entity supplying products to the marketplace,
    managing its inventory and interacting with the `Marketplace`'s publishing system.
    """
    
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a new Producer thread.
        @param products (list): A list of tuples, where each tuple contains
                                (product_name, quantity_to_publish, wait_time_after_publish).
        @param marketplace (Marketplace): A reference to the shared Marketplace instance.
        @param republish_wait_time (float): The time (in seconds) to wait before retrying to publish
                                            a product if the marketplace queue is full.
        @param kwargs: Arbitrary keyword arguments to be passed to the `Thread` constructor (e.g., `name`).
        """
        Thread.__init__(self, **kwargs) # Initialize the base Thread class.
        self.products = products # List of products this producer will supply.
        self.marketplace = marketplace # Reference to the shared Marketplace.
        self.republish_wait_time = republish_wait_time # Time to wait before retrying a publish operation.
        self.producer_id = -1 # Unique ID assigned by the marketplace after registration.

    def run(self):
        """
        @brief The main execution logic for the Producer thread.
        @details This method first registers the producer with the marketplace to obtain a unique ID.
        Then, it enters a continuous loop where it iterates through its defined products. For each product,
        it attempts to publish it to the marketplace a specified number of times, retrying if the
        marketplace's capacity is full. After each successful publish, it waits for a short period.
        @block_logic Manages the continuous supply of products to the marketplace.
        @pre_condition `self.marketplace` is an active `Marketplace` instance. `self.products` contains valid product definitions.
        @invariant The producer continuously attempts to publish its products, respecting marketplace capacity.
        """
        # Functional Utility: Register with the marketplace to get a unique producer ID.
        self.producer_id = self.marketplace.register_producer()

        # Block Logic: Main loop for continuous product publishing.
        # Invariant: The producer attempts to publish products indefinitely.
        while True:
            # Block Logic: Iterate through each product defined for this producer.
            # Invariant: Each product is considered for publishing.
            for product in self.products:
                # Block Logic: Publish the product a specified number of times (quantity).
                # Invariant: The product is published `product[1]` times.
                for _ in range(product[1]):
                    # Block Logic: Attempt to publish the product, retrying if the marketplace queue is full.
                    # Invariant: The product is eventually published, or the loop continues indefinitely if always full.
                    while not self.marketplace.publish(self.producer_id, product[0]):
                        sleep(self.republish_wait_time) # Wait before retrying publish.

                    sleep(product[2]) # Wait for a specified time after publishing a single instance of the product.

