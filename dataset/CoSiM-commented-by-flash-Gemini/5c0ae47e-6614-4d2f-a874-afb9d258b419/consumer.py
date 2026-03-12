

"""
This module simulates a multi-threaded marketplace system.

It defines three main components:
- `Consumer`: Represents a buyer thread that interacts with the marketplace to add,
  remove, and place orders for products.
- `Marketplace`: Acts as the central hub, managing product inventory, producer registrations,
  and consumer carts. It handles product publishing by producers and order fulfillment for consumers.
- `Producer`: Represents a seller thread that continuously publishes products to the marketplace.

The system utilizes Python's `threading` module for concurrency, multiple `threading.Lock`
instances for fine-grained synchronization of shared resources in the `Marketplace`,
and `dataclasses` for defining product types.
"""

import time
from threading import Thread
from threading import Lock


class Consumer(Thread):
    """
    Represents a consumer (buyer) in the marketplace simulation.
    Each Consumer instance runs as a separate thread, processing a list of predefined
    shopping carts, adding/removing products, and finally placing orders.
    It includes retry logic for adding products if they are not immediately available.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping carts, where each cart is a list of order requests.
                          An order request is a dictionary specifying product, quantity, and type ("add" or "remove").
            marketplace (Marketplace): The shared marketplace instance to interact with.
            retry_wait_time (float): The time in seconds to wait before retrying an action
                                     (e.g., adding a product to cart) if it fails.
            **kwargs: Arbitrary keyword arguments passed to the Thread.__init__ constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.print_locked = Lock() # A lock to ensure synchronized printing of bought products.

    def run(self):
        """
        The main execution method for the Consumer thread.
        It processes each assigned cart. For each cart, it creates a new cart
        in the marketplace, processes add/remove tasks for products (with a retry
        mechanism if products are not immediately available), and finally
        places the order, printing the purchased items in a thread-safe manner.
        """
        # Block Logic: Iterate through each shopping cart this consumer needs to process.
        for task_cart in self.carts:
            current_cart = self.marketplace.new_cart() # Create a new cart in the marketplace.
            for task in task_cart: # Iterate through each order task within the current cart.
                looper = task['quantity'] # Get the quantity of the product for the current task.
                while looper > 0: # Loop until the desired quantity is processed.
                    # Inline: Check if the product is currently in the marketplace's stock.
                    if task['product'] in self.marketplace.market_stock:
                        self.execute_task(task['type'], current_cart, task['product']) # Execute add/remove.
                        looper -= 1 # Decrement quantity after successful execution.
                    else:
                        time.sleep(self.retry_wait_time) # If product not in stock, wait and retry.

            # Block Logic: Place the final order for the completed cart.
            order = self.marketplace.place_order(current_cart)
            with self.print_locked: # Use a lock to ensure thread-safe printing.
                for product in order:
                    print(self.getName(), "bought", product) # Print what was bought by this consumer.

    def execute_task(self, task_type, cart_id, product):
        """
        Executes a single add or remove operation for a product in a cart.

        Args:
            task_type (str): The type of task, either 'add' or 'remove'.
            cart_id (int): The ID of the cart to modify.
            product (Product): The product to add or remove.
        """
        if task_type == 'remove':
            self.marketplace.remove_from_cart(cart_id, product) # Call marketplace method to remove.
        elif task_type == 'add':
            self.marketplace.add_to_cart(cart_id, product) # Call marketplace method to add.

from threading import Lock


class Marketplace:
    """
    Manages products, producers, and consumer carts in the simulated marketplace.
    It acts as a central hub where producers publish products and consumers
    can manage carts and place orders. It employs multiple locks for fine-grained
    synchronization of its various internal data structures.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have in the marketplace at any time.
        """
        self.queue_size_per_producer = queue_size_per_producer  # Max products a producer can have.
        self.no_of_producers = -1  # Counter for assigning unique producer IDs.
        self.no_of_carts = -1  # Counter for assigning unique cart IDs.
        self.product_creator = {}  # Maps product to its creating producer_id {product: producer_id}.
        self.market_stock = []  # Global list of all products currently available in the market.
        self.product_counter = []  # Stores count of products published by each producer [prod0_count, prod1_count, ...].
        self.cart = [[]]  # List of lists representing carts; cart[0] is unused, cart[id] holds products.

        # Multiple locks for fine-grained synchronization of different operations/data structures.
        self.register_locked = Lock()  # Lock for registering new producers.
        self.cart_locked = Lock()  # Lock for creating new carts.
        self.add_locked = Lock()  # Lock for adding products to a cart.
        self.remove_locked = Lock()  # Lock for removing products from a cart. (Seems unused in this impl)
        self.publish_locked = Lock()  # Lock for updating product_counter and product_creator during publish.
        self.market_locked = Lock()  # Lock for modifying market_stock.

    def register_producer(self):
        """
        Registers a new producer with the marketplace, assigning it a unique ID.
        This operation is protected by `register_locked`. An entry is also
        added to `product_counter` to track the producer's published items.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        with self.register_locked: # Ensure atomic update of producer ID.
            self.no_of_producers += 1
            new_prod_id = self.no_of_producers

        self.product_counter.append(0) # Initialize product count for the new producer.
        return new_prod_id

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.
        The product is added to `market_stock` and the producer's counter is incremented,
        but only if the producer has not exceeded its maximum allowed queue size.
        Updates to producer counts and product-to-creator mapping are synchronized.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (Product): The product object to publish.

        Returns:
            bool: True if the product was successfully published, False otherwise (queue full).
        """
        # Inline: Check if the producer's queue has reached its maximum size.
        if self.product_counter[producer_id] >= self.queue_size_per_producer:
            return False # Cannot publish if queue is full.

        self.market_stock.append(product) # Add the product to the global market stock.

        with self.publish_locked: # Protect updates to producer counts and mappings.
            self.product_counter[producer_id] += 1 # Increment product count for this producer.
            self.product_creator[product] = producer_id # Map product to its producer.

        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart and assigns it a unique cart ID.
        This operation is protected by `cart_locked`.

        Returns:
            int: The unique ID of the newly created cart.
        """
        with self.cart_locked: # Ensure atomic update of cart ID.
            self.no_of_carts += 1
            new_cart_id = self.no_of_carts

        self.cart.append([]) # Initialize an empty list for the new cart.
        return new_cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specific cart. The product is added only if it is
        currently available in the global `market_stock`. If successful, the product
        is moved from `market_stock` to the specified cart, and the producer's
        product counter is decremented.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (Product): The product object to add.

        Returns:
            bool: True if the product was successfully added, False if not found or unavailable.
        """
        if product not in self.market_stock: # Check if the product is in the global stock.
            return False # Product not available.
        self.cart[cart_id].append(product) # Add product to the consumer's cart.
        with self.add_locked: # Protect updates to product counters.
            self.product_counter[self.product_creator[product]] -= 1 # Decrement producer's product count.
        with self.market_locked: # Protect modifications to global market stock.
            if product in self.market_stock: # Double-check as another thread might have removed it.
                self.market_stock.remove(product) # Remove product from the global market stock.
        return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a specific cart and returns it to the marketplace.
        If the product is found in the cart, it's removed from there, added back
        to `market_stock`, and the original producer's product counter is incremented.

        Args:
            cart_id (int): The ID of the cart to remove the product from.
            product (Product): The product object to remove.
        """
        if product in self.cart[cart_id]: # Check if the product is in the specified cart.
            with self.cart_locked: # Protect updates to product counters.
                self.product_counter[self.product_creator[product]] += 1 # Increment producer's product count.
            self.cart[cart_id].remove(product) # Remove product from the consumer's cart.
            self.market_stock.append(product) # Add product back to the global market stock.

    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart.
        It returns the contents of the cart and effectively places the order,
        though the cart itself remains in the internal data structure for consumers
        to print.

        Args:
            cart_id (int): The ID of the cart to place the order for.

        Returns:
            list: A list of products (Product objects) that were in the placed order.
        """
        return self.cart[cart_id]

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
            products (list): A list of task tuples, where each tuple describes a product
                             to be published: (product_object, quantity, wait_time).
            marketplace (Marketplace): The shared marketplace instance to interact with.
            republish_wait_time (float): The time in seconds to wait before retrying to
                                         publish a product if the marketplace's queue is full.
            **kwargs: Arbitrary keyword arguments passed to the Thread.__init__ constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.prod_id = self.marketplace.register_producer() # Register producer and get unique ID.

    def run(self):
        


            def run(self):
        


                """
        


                The main execution method for the Producer thread.
        


                It continuously publishes its predefined products to the marketplace.
        


                For each product task, it attempts to publish the specified quantity.
        


                If a product cannot be published immediately (e.g., due to marketplace queue limits),
        


                it retries after a specified `republish_wait_time`. After each successful
        


                publication, it waits for a `wait_time` before publishing the next unit.
        


                """
        


                while True: # Infinite loop to keep the producer active.
        


                    for (product, quantity, wait_time) in self.products: # Iterate through each product task this producer offers.
        


                        looper = quantity # Counter for the remaining quantity of the current product to publish.
        


                        while looper > 0: # Loop until the desired quantity of the current product is published.
        


                            response = self.marketplace.publish(self.prod_id, product) # Attempt to publish the product.
        


                            if response: # If publishing is successful.
        


                                time.sleep(wait_time) # Wait for `wait_time` after successful publication of a unit.
        


                                looper -= 1 # Decrement the remaining quantity to publish.
        


                            else: # If publishing fails (e.g., marketplace queue full).
        


                                time.sleep(self.republish_wait_time) # Wait before retrying.
        


        
        


        from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Base dataclass representing a generic product in the marketplace.
    Attributes are immutable as `frozen=True`.

    Attributes:
        name (str): The name of the product.
        price (int): The price of the product.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Dataclass representing a Tea product, inheriting from `Product`.
    Attributes are immutable as `frozen=True`.

    Attributes:
        type (str): The type or variety of tea (e.g., "Green", "Black").
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Dataclass representing a Coffee product, inheriting from `Product`.
    Attributes are immutable as `frozen=True`.

    Attributes:
        acidity (str): The acidity level of the coffee (e.g., "Low", "High").
        roast_level (str): The roast level of the coffee (e.g., "Light", "Dark").
    """
    acidity: str
    roast_level: str
