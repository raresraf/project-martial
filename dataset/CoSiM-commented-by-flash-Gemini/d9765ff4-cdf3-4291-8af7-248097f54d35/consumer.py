
"""
@brief This module implements a simulation of a producer-consumer system using a `Marketplace` to mediate transactions.
@details It defines `Consumer` threads that purchase products, `Producer` threads that supply products,
and a `Marketplace` class that manages product queues, carts, and order placements with thread-safe operations,
utilizing class-level attributes for global state management. This simulation demonstrates concurrent access
to shared resources and synchronization challenges in a multi-threaded environment.
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
        @param carts (list): A list of shopping carts, where each cart is a list of operation dictionaries.
                             Each operation dictionary contains 'quantity', 'type' (add/remove), and 'product'.
        @param marketplace (Marketplace): A reference to the shared Marketplace instance.
        @param retry_wait_time (float): The time (in seconds) to wait before retrying an 'add' operation
                                        if a product is not available.
        @param kwargs: Arbitrary keyword arguments to be passed to the `Thread` constructor (e.g., `name`).
        """
        Thread.__init__(self, **kwargs) # Initialize the base Thread class.
        self.carts = carts # List of shopping carts this consumer will process.
        self.marketplace = marketplace # Reference to the shared Marketplace.
        self.wait_time = retry_wait_time # Time to wait before retrying an operation.
        self.name = kwargs['name'] # The name of this consumer thread.

    def run(self):
        """
        @brief The main execution logic for the Consumer thread.
        @details This method iterates through each shopping cart assigned to the consumer.
        For each cart, it performs a series of 'add' or 'remove' operations for products,
        retrying 'add' operations if the product is not immediately available.
        Once all operations for a cart are complete, an order is placed, and the purchased
        products are logged.
        @block_logic Processes a sequence of shopping carts, performing product transactions.
        @pre_condition `self.carts` contains valid cart definitions, `self.marketplace` is
                       an active `Marketplace` instance.
        @invariant Each cart is processed sequentially, and operations are performed with
                   potential retries until successful.
        """
        # Block Logic: Iterate through each shopping list (cart definition) assigned to this consumer.
        # Invariant: Each shopping list will be processed from cart creation to order placement.
        for shopping in self.carts:
            # Functional Utility: Create a new cart in the marketplace and obtain its ID.
            num_cart = self.marketplace.new_cart()
            # Block Logic: Process each product operation (add or remove) within the current shopping list.
            # Invariant: All operations for products in a shopping list are attempted.
            for product in shopping:
                number_action = int(product['quantity']) # Quantity of the current product to operate on.
                command = product['type'] # Type of operation: "add" or "remove".
                name_product = product['product'] # Name of the product for the operation.

                # Block Logic: Perform the operation for the specified quantity of the product.
                # Invariant: The product operation is attempted `number_action` times.
                while number_action != 0:
                    if command == "add":
                        # Block Logic: Attempt to add a product to the cart, retrying if unavailable.
                        # Invariant: The product is eventually added to the cart, or the loop continues indefinitely if always unavailable.
                        if self.marketplace.add_to_cart(num_cart, name_product):
                            number_action = number_action - 1 # Decrement count if successfully added.
                        else:
                            sleep(self.wait_time) # Wait before retrying.
                    if command == "remove":
                        # Functional Utility: Remove a product from the cart.
                        self.marketplace.remove_from_cart(num_cart, name_product)
                        number_action = number_action - 1 # Decrement count after removal.

            # Functional Utility: Place the order for the fully populated cart and get the list of bought items.
            shopping = self.marketplace.place_order(num_cart)
            # Block Logic: Print the details of the products purchased by this consumer.
            # Invariant: Each product successfully ordered is logged to console.
            for _, product in shopping:
                print(self.name, "bought", product)


from threading import Lock


class Marketplace:
    """
    @brief Acts as a central hub for producers and consumers to interact in a simulated e-commerce environment.
    @details This class manages product availability from various producers, handles shopping cart operations
    (creation, adding/removing products), and processes final orders. It uses class-level attributes (`producers`,
    `consumers`, `id_prod`, `id_cons`) to manage global state across all instances. Multiple locks ensure
    thread-safe access to these shared resources.
    @architectural_intent Provides a thread-safe, synchronized environment for managing product flow
    from multiple producers to multiple consumers, abstracting the complexities of concurrent access
    and maintaining the integrity of inventory and cart data using global state and explicit locks.
    """
    
    # Class-level attributes to manage global state of the marketplace.
    producers = {} # Dictionary: producer_id -> list of products published by that producer.
    consumers = {} # Dictionary: cart_id -> list of (producer_id, product) in that cart.
    id_prod = 1    # Counter for assigning unique producer IDs.
    id_cons = 1    # Counter for assigning unique cart IDs.
    lock_producer = Lock() # Lock to protect `id_prod` and `producers` dictionary.
    lock_cart = Lock()     # Lock to protect `id_cons` and `consumers` (carts) dictionary.

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes a new Marketplace instance.
        @param queue_size_per_producer (int): The maximum number of products each producer can have
                                             in the marketplace at any given time.
        """
        self.size = queue_size_per_producer # Max products per producer.


    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace.
        @details Assigns a unique producer ID and initializes an empty product list for it.
        This operation is thread-safe using `lock_producer`.
        @return int: The newly assigned unique producer ID.
        @block_logic Assigns a unique ID to a new producer and prepares its product inventory.
        @pre_condition `self.lock_producer` is an initialized Lock.
        @invariant `self.id_prod` is incremented, and a new entry for the producer
                   is created in `self.producers`.
        """
        self.lock_producer.acquire() # Acquire lock to ensure thread-safe ID assignment and dictionary modification.
        products = [] # Initialize an empty list for this producer's products.
        self.producers[self.id_prod] = products # Add the new producer and its product list.
        self.id_prod = self.id_prod+1 # Increment to get a new unique producer ID for the next registration.
        self.lock_producer.release() # Release lock.
        return self.id_prod-1 # Return the assigned producer ID.


    def publish(self, producer_id, product):
        """
        @brief Publishes a product from a producer to the marketplace.
        @details A product can only be published if the producer's queue is not full.
        This operation is not explicitly protected by a lock, assuming `add_to_cart`
        handles mutual exclusion for product movement.
        @param producer_id (int): The ID of the producer publishing the product.
        @param product (object): The product to be published.
        @return bool: True if the product was successfully published, False if the producer's queue is full.
        @block_logic Adds a product to a producer's inventory if capacity allows.
        @pre_condition `producer_id` is a valid, registered producer.
        @invariant If the product is published, it is added to `self.producers[producer_id]`.
        """
        # Block Logic: Check if the producer's queue has reached its maximum capacity.
        # Invariant: Product is added only if the queue is not full.
        if len(self.producers[producer_id]) == self.size:
            return False # Producer's queue is full, cannot publish.
        self.producers[producer_id].append(product) # Add the product to the producer's list.
        return True # Product successfully published.


    def new_cart(self):
        """
        @brief Creates a new empty shopping cart in the marketplace.
        @details Assigns a unique cart ID and initializes an empty list for its contents.
        This operation is thread-safe using `lock_cart`.
        @return int: The newly assigned unique cart ID.
        @block_logic Assigns a unique ID to a new cart and initializes its content.
        @pre_condition `self.lock_cart` is an initialized Lock.
        @invariant `self.id_cons` is incremented, and a new entry for the cart
                   is created in `self.consumers`.
        """
        self.lock_cart.acquire() # Acquire lock to ensure thread-safe ID assignment and dictionary modification.
        cart = [] # Initialize an empty list for this cart's products.
        self.consumers[self.id_cons] = cart # Add the new cart.
        self.id_cons = self.id_cons + 1 # Increment to get a new unique cart ID for the next creation.
        self.lock_cart.release() # Release lock.
        return self.id_cons - 1 # Return the assigned cart ID.

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a product to a specific shopping cart.
        @details This method searches for the product across all producers. If found,
        it moves the product from the producer's inventory to the specified cart.
        This operation assumes atomicity due to the simple list operations and
        the absence of explicit lock in this method for `producers` access,
        which might lead to race conditions if `publish` is also concurrent.
        @param cart_id (int): The ID of the cart to add the product to.
        @param product (object): The product to add.
        @return bool: True if the product was found and added to the cart, False otherwise.
        @block_logic Transfers a product from a producer's inventory to a consumer's cart.
        @pre_condition `cart_id` is a valid, existing cart ID. `self.producers`
                       and `self.consumers` (carts) are properly initialized.
        @invariant If successful, `product` is removed from a producer's list and added to the cart.
        """
        # Block Logic: Iterate through all producers to find the requested product.
        # Invariant: Product is searched across all producer inventories.
        for producer in self.producers: # Iterate through producer IDs.
            for prod in self.producers[producer]: # Iterate through products of the current producer.
                if product == prod: # If the product is found.
                    self.consumers[cart_id].insert(0, [producer, product]) # Add product to the cart with its producer ID.
                    self.producers[producer].remove(product) # Remove product from producer's inventory.
                    return True # Product successfully added.
        return False # Product not found in any producer's inventory.

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a specific shopping cart and returns it to the producer.
        @details This method finds the product within the cart, identifies its original producer,
        removes it from the cart, and returns it to the producer's inventory.
        This operation assumes atomicity and relies on correct state management.
        @param cart_id (int): The ID of the cart from which to remove the product.
        @param product (object): The product to remove.
        @return None: This function does not explicitly return a useful value, but modifies state.
        @block_logic Transfers a product from a consumer's cart back to its original producer's inventory.
        @pre_condition `cart_id` is a valid, existing cart ID. `product` is present in the specified cart.
                       `self.producers` and `self.consumers` are properly initialized.
        @invariant `product` is removed from the cart and re-added to the correct producer's list.
        """
        # Block Logic: Find the cart and the product within it.
        # Invariant: The correct producer ID and product are identified for removal.
        for cart in self.consumers: # Iterate through cart IDs.
            if cart == cart_id: # Find the matching cart.
                for index, prod in self.consumers[cart]: # Iterate through (producer_id, product) pairs in the cart.
                    if prod == product: # If the product is found.
                        self.consumers[cart_id].remove([index, product]) # Remove product from the cart.
                        self.producers[index].append(product) # Return product to its original producer's inventory.
                        return None # Operation complete.
        return None # Product not found or cart_id invalid.

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
        return self.consumers[cart_id] # Return the list of (producer_id, product) tuples in the cart.



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
        self.wait_time = republish_wait_time # Time to wait before retrying a publish operation.


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
        id_producer = self.marketplace.register_producer() # Register with the marketplace to get a unique producer ID.
        # Block Logic: Main loop for continuous product publishing.
        # Invariant: The producer attempts to publish products indefinitely.
        while True:
            # Block Logic: Iterate through each product defined for this producer.
            # Invariant: Each product is considered for publishing.
            for product in self.products:
                name_product = product[0] # Name of the product.
                number_pieces = int(product[1]) # Quantity of this product to publish.
                time_product = product[2] # Time to wait after publishing one piece.

                # Block Logic: Publish the product a specified number of times (number_pieces).
                # Invariant: The product is published `number_pieces` times.
                while number_pieces != 0:
                    # Block Logic: Attempt to publish the product, retrying if the marketplace queue is full.
                    # Invariant: The product is eventually published, or the loop continues indefinitely if always full.
                    if self.marketplace.publish(id_producer, name_product):
                        sleep(time_product) # Wait after successful publish of one piece.
                    else:
                        sleep(self.wait_time) # Wait before retrying publish if queue is full.
                    number_pieces = number_pieces - 1 # Decrement count of pieces to publish.

