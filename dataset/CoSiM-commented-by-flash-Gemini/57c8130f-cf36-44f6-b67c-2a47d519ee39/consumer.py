


"""
This module simulates a multi-threaded marketplace system.

It defines three main components:
- `Consumer`: Represents a buyer thread that interacts with the marketplace to add,
  remove, and place orders for products.
- `Marketplace`: Acts as the central hub, managing product inventory, producer registrations,
  and consumer carts. It handles product publishing by producers and order fulfillment for consumers.
- `Producer`: Represents a seller thread that continuously publishes products to the marketplace.

The system uses Python's `threading` module for concurrency and `threading.Lock`
for basic synchronization to manage shared resources in the `Marketplace` class.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    Represents a consumer (buyer) in the marketplace simulation.
    Each Consumer instance runs as a separate thread, processing a list of predefined
    shopping carts, adding/removing products, and finally placing orders.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping carts, where each cart is a list of orders.
                          An order is a dictionary specifying product, quantity, and type ("add" or "remove").
            marketplace (Marketplace): The shared marketplace instance to interact with.
            retry_wait_time (float): The time in seconds to wait before retrying an action
                                     (e.g., adding a product to cart) if it fails.
            **kwargs: Arbitrary keyword arguments passed to the Thread.__init__ constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

        def run(self):

            """

            The main execution method for the Consumer thread.

            It iterates through each assigned cart, creates a new cart in the marketplace,

            processes each order within the cart (adding or removing products with retries),

            and finally places the completed order.

            """

            for cart in self.carts:

                cart_id = self.marketplace.new_cart() # Create a new cart in the marketplace for this consumer.

                for order in cart:

                    if order["type"] == "add":

                        # Block Logic: Attempt to add the specified quantity of a product to the cart.

                        # Includes a retry mechanism with `time.sleep` if the add operation fails.

                        for i in range(0, order["quantity"]):

                            while True: # Loop until product is successfully added.

                                out = self.marketplace.add_to_cart(cart_id, order["product"])

                                while not out: # If add_to_cart returns False (e.g., product unavailable), retry after a delay.

                                    time.sleep(self.retry_wait_time)

                                    out = self.marketplace.add_to_cart(cart_id, order["product"])

                                else: # Product successfully added.

                                    break

                    elif order["type"] == "remove":

                        # Block Logic: Remove the specified quantity of a product from the cart.

                        for i in range(0, order["quantity"]):

                            self.marketplace.remove_from_cart(cart_id, order["product"]) # Call marketplace method to remove.

                self.marketplace.place_order(cart_id) # Place the final order for the cart.

    

    from threading import currentThread, Lock

class Marketplace:
    """
    Manages products, producers, and consumer carts in the simulated marketplace.
    It acts as a central hub where producers publish products and consumers
    can add/remove products from their carts and place orders.
    Basic synchronization with `Lock` is used to protect shared data structures.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have in the marketplace at any time.
        """
        self.max_q_per_prod = queue_size_per_producer # Maximum queue size per producer.
        self.producer_id = 0 # Counter for assigning unique producer IDs.
        self.products = {} # Dictionary to store products published by each producer {producer_id: [product1, product2, ...]}
        self.cart_id = 0 # Counter for assigning unique cart IDs.
        self.carts = {} # Dictionary to store consumer carts {cart_id: [(product, producer_id), ...]}
        self.marketplace = [] # Global list of available products in the marketplace [(product, producer_id), ...]

    def register_producer(self):
        """
        Registers a new producer with the marketplace, assigning it a unique ID.
        Each producer gets an empty list in `self.products` to track its published items.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        lock = Lock() # Inline: A temporary lock to protect the producer ID counter and products dictionary during registration.
        lock.acquire() # Acquire the lock.
        self.producer_id += 1 # Increment to get a new unique producer ID.
        self.products[self.producer_id] = [] # Initialize an empty list for the new producer's products.
        lock.release() # Release the lock.
        return self.producer_id

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.
        The product is added only if the producer has not exceeded its
        maximum allowed queue size.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (str): The name of the product to publish.

        Returns:
            bool: True if the product was successfully published, False otherwise.
        """
        num_prod = self.products[producer_id] # Get the list of products currently published by this producer.
        if len(num_prod) >= self.max_q_per_prod: # Check if the producer's queue is full.
            return False # Cannot publish if queue is full.

        self.marketplace.append((product, producer_id)) # Add the product to the global marketplace list.
        num_prod.append(product) # Add the product to the producer's individual product list.
        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart and assigns it a unique cart ID.

        Returns:
            int: The unique ID of the newly created cart.
        """
        lock = Lock() # Inline: A temporary lock to protect the cart ID counter and carts dictionary during cart creation.
        lock.acquire() # Acquire the lock.
        self.cart_id += 1 # Increment to get a new unique cart ID.
        cart_id = self.cart_id # Store the new cart ID.
        self.carts[cart_id] = [] # Initialize an empty list for the new cart.
        lock.release() # Release the lock.
        return cart_id
    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specific cart if the product is available in the marketplace.
        If successful, the product is removed from the global marketplace and the producer's inventory.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (str): The name of the product to add.

        Returns:
            bool: True if the product was successfully added, False if not found or unavailable.
        """
        # Block Logic: Iterate through the global marketplace to find the desired product.
        for (product_type, producer_id) in self.marketplace:
            if product_type == product: # Check if the product type matches.
                # Inline: Verify that the product is actually in the producer's inventory.
                if product in self.products[producer_id]: 
                    self.carts[cart_id].append((product, producer_id)) # Add product to the consumer's cart.
                    self.marketplace.remove((product_type, producer_id)) # Remove product from the global marketplace.
                    self.products[producer_id].remove(product) # Remove product from the producer's inventory.
                    return True # Indicate successful addition.
        return False # Product not found or unavailable.

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a specific cart and returns it to the marketplace.
        The product is re-added to the global marketplace and the corresponding producer's inventory.

        Args:
            cart_id (int): The ID of the cart to remove the product from.
            product (str): The name of the product to remove.
        """
        # Block Logic: Iterate through the items in the specified cart to find the product to remove.
        for (product_type, _producer_id) in self.carts[cart_id]:
            if product_type == product: # Check if the product type matches.
                self.carts[cart_id].remove((product, _producer_id)) # Remove product from the consumer's cart.
                self.marketplace.append((product_type, _producer_id)) # Add product back to the global marketplace.
                self.products[_producer_id].append(product) # Add product back to the producer's inventory.
                break # Exit loop after removing the first instance of the product.

    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart.
        It prints a message for each item bought and then removes the cart from the marketplace.

        Args:
            cart_id (int): The ID of the cart to place the order for.

        Returns:
            list or None: The list of items in the cart that was placed, or None if the cart_id was not found.
        """
        for (product, _producer_id) in self.carts[cart_id]:
            # Inline: Prints the name of the current thread (consumer) and the product bought.
            print("{} bought {}".format(currentThread().getName(), product))

        return self.carts.pop(cart_id, None) # Remove the cart from the active carts and return its contents.


class Producer(Thread):
    """
    Represents a producer (seller) in the marketplace simulation.
    Each Producer instance runs as a separate thread, continuously publishing
    products to the marketplace, with retry mechanisms if the marketplace
    cannot accept new products immediately.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of tuples, where each tuple describes a product:
                             (product_name: str, quantity: int, wait_time: float).
            marketplace (Marketplace): The shared marketplace instance to interact with.
            republish_wait_time (float): The time in seconds to wait before retrying to
                                         publish a product if the marketplace queue is full.
            **kwargs: Arbitrary keyword arguments passed to the Thread.__init__ constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

        # Registers this producer with the marketplace and gets a unique producer ID.
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        The main execution method for the Producer thread.
        It continuously attempts to publish its predefined products to the marketplace.
        If a product cannot be published immediately (e.g., due to marketplace queue limits),
        it retries after a specified waiting period.
        """
        while True: # Infinite loop to keep the producer active.
            for (product, quantity, wait_time) in self.products: # Iterate through each product type this producer offers.
                # Block Logic: Publish the specified quantity of the current product.
                # Includes a retry mechanism if publishing fails.
                while quantity: # Loop until the desired quantity of the current product is published.
                    out = self.marketplace.publish(self.producer_id, product) # Attempt to publish the product.
                    if out == False: # If publishing fails (e.g., marketplace queue full).
                        time.sleep(self.republish_wait_time) # Wait before retrying.
                    else: # If publishing is successful.
                        quantity -= 1 # Decrement the remaining quantity to publish.
                        time.sleep(wait_time) # Wait for a specified time before publishing the next item of the same type.
