


"""
This module simulates a multi-threaded marketplace system.

It defines three main components:
- `Consumer`: Represents a buyer thread that interacts with the marketplace to add,
  remove, and place orders for products.
- `Marketplace`: Acts as the central hub, managing product inventory, producer registrations,
  and consumer carts. It handles product publishing by producers and order fulfillment for consumers.
- `Producer`: Represents a seller thread that continuously publishes products to the marketplace.

The system uses Python's `threading` module for concurrency, `threading.Lock`
for basic synchronization, and `dataclasses` for defining product types.
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
            carts (list): A list of shopping carts, where each cart is a list of order requests.
                          An order request is a dictionary specifying product, quantity, and type ("add" or "remove").
            marketplace (Marketplace): The shared marketplace instance to interact with.
            retry_wait_time (float): The time in seconds to wait before retrying an action
                                     (e.g., adding a product to cart) if it fails.
            **kwargs: Arbitrary keyword arguments passed to the Thread.__init__ constructor,
                      including 'name' for the thread's name.
        """
        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs['name'] # Store the thread name for printing.

    def add_product(self, cart_id, product, quantity):
        """
        Attempts to add a specified quantity of a product to the given cart.
        It includes a retry mechanism: if `add_to_cart` fails, it waits for
        `retry_wait_time` and tries again until successful.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (Product): The product object to add.
            quantity (int): The number of units of the product to add.
        """
        for _ in range(quantity): # Loop for each unit of the product.
            tmp = self.marketplace.add_to_cart(cart_id, product) # Attempt to add.

            # Block Logic: Retry mechanism for adding product.
            while tmp is False: # If add_to_cart returns False (e.g., product unavailable).
                time.sleep(self.retry_wait_time) # Wait for a short period.
                tmp = self.marketplace.add_to_cart(cart_id, product) # Retry adding.


    def remove_product(self, cart_id, product, quantity):
        """
        Removes a specified quantity of a product from the given cart.

        Args:
            cart_id (int): The ID of the cart to remove the product from.
            product (Product): The product object to remove.
            quantity (int): The number of units of the product to remove.
        """
        for _ in range(quantity): # Loop for each unit of the product.
            self.marketplace.remove_from_cart(cart_id, product) # Call marketplace method to remove.


    def run(self):
        """
        The main execution method for the Consumer thread.
        It iterates through each assigned cart, creates a new cart in the marketplace,
        processes each order request within the cart (adding or removing products),
        and finally places the completed order, printing the purchased items.
        """
        for cart in self.carts: # Iterate through each shopping cart this consumer needs to process.
            cart_id = self.marketplace.new_cart() # Create a new cart in the marketplace.

            for request in cart: # Iterate through each order request within the current cart.
                order = request["type"] # Type of operation ("add" or "remove").
                product = request["product"] # The product involved in the request.
                quantity = request["quantity"] # The quantity of the product.

                if order == "add":
                    self.add_product(cart_id, product, quantity) # Call helper to add product.
                elif order == "remove":
                    self.remove_product(cart_id, product, quantity) # Call helper to remove product.

            order = self.marketplace.place_order(cart_id) # Place the final order for the cart.

            # Block Logic: Print the items bought in the placed order.
            for product in order:
                print(self.name + " bought " + str(product))

from threading import Lock


class Marketplace:
    """
    Manages products, producers, and consumer carts in the simulated marketplace.
    It acts as a central hub where producers publish products and consumers
    can add/remove products from their carts and place orders.
    Synchronization is handled using a single `threading.Lock` for critical sections.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have in the marketplace at any time.
        """
        self.queue_size_per_producer = queue_size_per_producer # Maximum queue size per producer.
        self.prod_count = 0 # Counter for assigning unique producer IDs.
        self.cart_count = 0 # Counter for assigning unique cart IDs.

        self.products = {}      # Dictionary to store products published by each producer {producer_id: [product1, product2, ...]}
        self.carts = {}         # Dictionary to store consumer carts {cart_id: [(product, producer_id), ...]}
        self.lock = Lock() # Global lock to protect shared counters and data structures during modifications.

    def register_producer(self):
        """
        Registers a new producer with the marketplace, assigning it a unique ID.
        Each producer gets an empty list in `self.products` to track its published items.
        The producer ID counter is protected by a global lock.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        # Block Logic: Acquire lock to protect shared counters during producer registration.
        self.lock.acquire()
        self.prod_count = self.prod_count + 1 # Increment to get a new unique producer ID.
        self.lock.release() # Release the lock.

        self.products[self.prod_count] = [] # Initialize an empty list for the new producer's products.
        return self.prod_count


    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.
        The product is added only if the producer has not exceeded its
        maximum allowed queue size (`queue_size_per_producer`).

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (Product): The product object to publish.

        Returns:
            bool: True if the product was successfully published, False otherwise (queue full).
        """
        lenght = len(self.products[producer_id]) # Get the current number of products from this producer.
        if lenght > self.queue_size_per_producer: # Check if the producer's queue is full.
            return False # Cannot publish if queue is full.

        self.products[producer_id].append(product) # Add the product to the producer's individual product list.
        return True


    def new_cart(self):
        """
        Creates a new, empty shopping cart and assigns it a unique cart ID.
        The cart ID counter is protected by a global lock.

        Returns:
            int: The unique ID of the newly created cart.
        """
        # Block Logic: Acquire lock to protect shared counters during cart creation.
        self.lock.acquire()
        self.cart_count = self.cart_count + 1 # Increment to get a new unique cart ID.
        self.lock.release() # Release the lock.

        self.carts[self.cart_count] = [] # Initialize an empty list for the new cart.
        return self.cart_count


    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specific cart if the product is available in any producer's inventory.
        If successful, the product is moved from the producer's inventory to the consumer's cart.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (Product): The product object to add.

        Returns:
            bool: True if the product was successfully added, False if not found or unavailable.
        """
        # Block Logic: Iterate through all producers to find the product.
        for producer_id in self.products:
            if product in self.products[producer_id]: # If the product is found in this producer's inventory.
                # Inline: Store the product and its original producer ID as a tuple.
                tmp = (product, producer_id)

                # Inline: Add the product to the consumer's cart.
                self.carts[cart_id].append(tmp)
                # Inline: Remove the product from the producer's inventory.
                self.products[producer_id].remove(product)
                return True # Indicate successful addition.

        return False # Product not found or unavailable.


    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a specific cart and returns it to the original producer's inventory.

        Args:
            cart_id (int): The ID of the cart to remove the product from.
            product (Product): The product object to remove.
        """
        # Block Logic: Iterate through the items in the specified cart to find the product to remove.
        for tmp in self.carts[cart_id]:
            current_prod = tmp[0] # The product object.
            producer_id = tmp[1] # The ID of the original producer.

            if product == current_prod: # Check if the product matches.
                # Inline: Add the product back to the producer's inventory.
                self.products[producer_id].append(product)
                # Inline: Remove the product from the consumer's cart.
                self.carts[cart_id].remove(tmp)
                break # Exit loop after removing the first instance of the product.


    def place_order(self, cart_id):
        """
        Finalizes an order for a given cart.
        It extracts the product names from the cart items and removes the cart
        from the active carts in the marketplace.

        Args:
            cart_id (int): The ID of the cart to place the order for.

        Returns:
            list: A list of product objects that were in the placed order.
        """
        order = [] # List to store only the product objects from the cart.
        
        # Inline: Extract just the product object from each (product, producer_id) tuple in the cart.
        for product in self.carts[cart_id]:
            order.append(product[0])

        # Inline: Remove the cart from the active carts in the marketplace.
        self.carts.pop(cart_id)

        return order


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


                             to be published: (product_object, quantity, making_time).


            marketplace (Marketplace): The shared marketplace instance to interact with.


            republish_wait_time (float): The time in seconds to wait before retrying to


                                         publish a product if the marketplace's queue is full.


            **kwargs: Arbitrary keyword arguments passed to the Thread.__init__ constructor.


        """


        Thread.__init__(self, **kwargs)


        self.products = products


        self.marketplace = marketplace


        self.republish_wait_time = republish_wait_time


        self.producer_id = self.marketplace.register_producer() # Register producer and get unique ID.





        def run(self):





            """





            The main execution method for the Producer thread.





            It continuously publishes its predefined products to the marketplace.





            If a product cannot be published immediately (e.g., due to marketplace queue limits),





            it retries after a specified `republish_wait_time`. After successfully publishing





            each unit of a product, it waits for a specified `making_time`.





            """





            while True: # Infinite loop to keep the producer active.





                for task in self.products: # Iterate through each product task this producer offers.





                    product = task[0] # The product object to publish.





                    quantity = task[1] # The quantity of this product to publish.





                    making_time = task[2] # The time to wait after publishing each unit.





    





                    # Block Logic: Publish the specified quantity of the current product.





                    # Includes a retry mechanism if publishing fails.





                    for _ in range(quantity): # Loop for each unit of the product.





                        temp = self.marketplace.publish(self.producer_id, product) # Attempt to publish.





    





                        # Block Logic: Retry mechanism for publishing product.





                        while not temp: # If publishing fails (e.g., marketplace queue full).





                            time.sleep(self.republish_wait_time) # Wait for a short period.





                            temp = self.marketplace.publish(self.producer_id, product) # Retry publishing.





    





                        time.sleep(making_time) # Wait for `making_time` after successful publication of a unit.





    





    from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Base dataclass representing a generic product in the marketplace.

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

    Attributes:
        type (str): The type or variety of tea (e.g., "Green", "Black").
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Dataclass representing a Coffee product, inheriting from `Product`.

    Attributes:
        acidity (str): The acidity level of the coffee.
        roast_level (str): The roast level of the coffee.
    """
    acidity: str
    roast_level: str
