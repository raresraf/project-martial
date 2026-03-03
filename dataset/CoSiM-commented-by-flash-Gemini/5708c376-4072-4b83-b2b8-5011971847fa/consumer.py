"""
This module implements a multi-threaded producer-consumer marketplace simulation.

It defines:
- Consumer: A thread that simulates a buyer, adding and removing products from a cart, and placing orders.
- Marketplace: The central hub managing product inventories from Producers and handling consumer cart operations, ensuring thread safety.
- Producer: A thread that simulates a seller, continuously publishing products to the Marketplace.

The module also includes `TestMarketplace` for unit testing the `Marketplace` class.
"""

import time
from threading import Thread, Lock
import unittest # Imported here for the TestMarketplace class
# from product import Product, Tea, Coffee # These dataclasses are often defined in a separate file.


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
                          where each task is a dictionary like
                          `{'type': 'add'/'remove', 'product': product_obj, 'quantity': int}`.
            marketplace (Marketplace): The shared marketplace instance to interact with.
            retry_wait_time (float): The time (in seconds) to wait before retrying a failed operation.
            **kwargs: Arbitrary keyword arguments passed to the base `Thread` constructor,
                      e.g., `name` for thread identification.
        """
        Thread.__init__(self, **kwargs) # Initialize the base Thread class with kwargs.
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        # A Lock to ensure that printing to console is thread-safe and output is not interleaved.
        self.print_locked = Lock()

    def run(self):
        """
        The main execution method for the Consumer thread.

        This method simulates the shopping process for the consumer:
        1. Creates a new shopping cart in the marketplace.
        2. Iterates through a list of product tasks (add or remove products).
        3. For 'add' tasks, it attempts to add products to the cart, retrying if the product is unavailable.
        4. For 'remove' tasks, it removes products from the cart.
        5. Finally, it places the order and prints the purchased items to standard output,
           ensuring thread-safe printing.
        """
        my_cart = self.marketplace.new_cart() # Request a new cart ID from the marketplace.
        
        # Block Logic: Process each shopping cart definition for this consumer.
        for cart in self.carts:
            # Block Logic: Process each task (add/remove product) within the current cart.
            for to_do in cart:
                repeat = to_do['quantity'] # Number of times to perform the current task for this product.
                while repeat > 0: # Loop until the desired quantity for the task is processed.
                    # Conditional Logic: Check if the product is actually available in the market before attempting task.
                    if to_do['product'] in self.marketplace.market_stock:
                        self.execute_task(to_do['type'], my_cart, to_do['product']) # Execute the add/remove task.
                        repeat -= 1 # Decrement remaining quantity for the task.
                    else:
                        # If product is not in stock, wait and retry.
                        time.sleep(self.retry_wait_time)

            # Block Logic: Place the order and print the purchased products in a thread-safe manner.
            order = self.marketplace.place_order(my_cart)
            with self.print_locked: # Acquire lock to ensure exclusive access to print.
                for product in order:
                    print(self.getName(), "bought", product)

    def execute_task(self, task_type, cart_id, product):
        """
        Executes a single add or remove task for a product in a given cart.

        Args:
            task_type (str): The type of task to perform ('add' or 'remove').
            cart_id (int): The ID of the cart to modify.
            product (Product): The product to add or remove.
        """
        if task_type == 'add':
            self.marketplace.add_to_cart(cart_id, product)
        elif task_type == 'remove':
            self.marketplace.remove_from_cart(cart_id, product)


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
        self.queue_size_per_producer = queue_size_per_producer  
        
        # Producer management
        self.num_producers = -1  # Counter for assigning unique producer IDs (starts from 0).
        self.register_locked = Lock()  # Lock for thread-safe producer registration.
        # List to track how many products each producer has published that are still in circulation.
        self.product_counter = []
        # Dictionary mapping a product instance to its original producer_id.
        self.product_owner = {}
        self.market_stock = [] # Centralized list of all products currently available in the market.
        
        # Consumer (cart) management
        self.num_consumers = -1  # Counter for assigning unique cart IDs (starts from 0).
        self.cart = [[]]  # List of lists, where each inner list represents a consumer's shopping cart.
        self.cart_locked = Lock() # Lock for thread-safe cart creation.
        
        # Locks for various marketplace operations to ensure thread safety.
        self.add_locked = Lock()     # Protects `add_to_cart` operations.
        self.remove_locked = Lock()  # Protects `remove_from_cart` operations.
        self.publish_locked = Lock() # Protects `publish` operations.
        self.market_locked = Lock()  # Protects access to `market_stock`.

    def register_producer(self):
        """
        Registers a new producer with the marketplace and assigns it a unique ID.

        Returns:
            int: The unique ID assigned to the registered producer.
        """
        with self.register_locked: # Critical Section: Protects `num_producers` increment and `product_counter` append.
            self.num_producers += 1
            new_producer_id = self.num_producers
            self.product_counter.append(0) # Initialize product counter for this new producer.
        return new_producer_id

    def publish(self, producer_id, product):
        """
        Publishes a product from a producer to the marketplace's central stock.

        Checks if the producer's allocated queue size limit has been reached for products
        currently in circulation (published by this producer and not yet removed).

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (Product): The product object to be published.

        Returns:
            bool: True if the product was successfully published (producer's queue was not full), False otherwise.
        """
        # Checks if the producer has reached its limit of active products.
        if self.product_counter[producer_id] >= self.queue_size_per_producer:
            return False # Cannot publish, queue is full.
        
        # Critical Section: Protects modifications to `market_stock`, `product_counter`, and `product_owner`.
        with self.publish_locked:
            self.market_stock.append(product) # Add product to the central market stock.
            self.product_counter[producer_id] += 1 # Increment producer's active product counter.
            self.product_owner[product] = producer_id # Record the producer as the owner of this product instance.
        return True

    def new_cart(self):
        """
        Creates a new empty shopping cart and assigns it a unique ID.

        Returns:
            int: The unique ID assigned to the new cart.
        """
        with self.cart_locked: # Critical Section: Protects `num_consumers` increment and `cart` append.
            self.num_consumers += 1
            new_consumer_cart_id = self.num_consumers
            self.cart.append([]) # Add a new empty cart to the list of all carts.
        return new_consumer_cart_id

    def add_to_cart(self, cart_id, product):
        """
        Attempts to add a specified product to a consumer's cart.

        If the product is available in the `market_stock`, it is moved from `market_stock`
        to the consumer's cart, and the producer's active product count is adjusted.

        Args:
            cart_id (int): The ID of the cart to which the product should be added.
            product (Product): The product object to add.

        Returns:
            bool: True if the product was found and added to the cart, False otherwise.
        """
        # Check if the product is currently available in the market stock.
        if product not in self.market_stock:
            return False # Product not available.
        
        # Critical Section: Protects modifications to `cart`, `product_counter`, and `market_stock`.
        with self.add_locked:
            self.cart[cart_id].append(product) # Add product to the consumer's cart.
            self.product_counter[self.product_owner[product]] -= 1 # Decrement producer's active product count.
            
            # Critical Section: Protects modifications to `market_stock` list.
            with self.market_locked:
                if product in self.market_stock: # Double-check for existence before removal.
                    element_index = self.market_stock.index(product)
                    del self.market_stock[element_index] # Remove product from central market stock.
        return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a specified product from a consumer's cart and returns it to its original producer's stock.

        Args:
            cart_id (int): The ID of the cart from which the product should be removed.
            product (Product): The product object to remove.
        """
        # Check if the product is actually in the cart before attempting to remove.
        if product in self.cart[cart_id]:
            with self.remove_locked: # Critical Section: Protects modifications to `product_counter`, `cart`, and `market_stock`.
                self.product_counter[self.product_owner[product]] += 1 # Increment producer's active product count.
                self.cart[cart_id].remove(product) # Remove product from the consumer's cart.
                
                with self.market_locked: # Critical Section: Protects modification of `market_stock`.
                    self.market_stock.append(product) # Return product to the central market stock.

    def place_order(self, cart_id):
        """
        Retrieves the contents of a specific shopping cart, simulating placing an order.

        Args:
            cart_id (int): The ID of the cart to retrieve.

        Returns:
            list: A list of products that were in the specified cart.
        """
        return self.cart[cart_id]


class TestMarketplace(unittest.TestCase):
    """
    Unit tests for the Marketplace class functionalities.
    """

    def setUp(self):
        """
        Set up method called before each test function.
        Initializes a fresh Marketplace instance and sample products.
        """
        # Dynamically import Product, Tea, Coffee here if they were not imported globally.
        # This setup assumes Product, Tea, Coffee are defined directly or imported.
        from dataclasses import dataclass
        @dataclass(init=True, repr=True, order=False, frozen=True)
        class Product:
            name: str
            price: int
        @dataclass(init=True, repr=True, order=False, frozen=True)
        class Tea(Product):
            type: str
        @dataclass(init=True, repr=True, order=False, frozen=True)
        class Coffee(Product):
            acidity: str
            roast_level: str

        self.marketplace = Marketplace(15) # Initialize marketplace with a queue size of 15.
        self.tea = Tea("Lipton", 9, "Green") # Sample Tea product.
        self.coffee = Coffee("Doncafe", 10, "5.05", "MEDIUM") # Sample Coffee product.

    def test_register_producer(self):
        """
        Tests the `register_producer` method for correct producer ID assignment.
        """
        id_prod = self.marketplace.register_producer() # Register first producer.
        i = 1
        while i < 10: # Register 9 more producers.
            id_prod = self.marketplace.register_producer()
            i = i + 1
        self.assertEqual(id_prod, 10) # Assert that the last assigned producer ID is 10.

    def test_publish(self):
        """
        Tests the `publish` method for successful product publishing.
        """
        id_prod = self.marketplace.register_producer() # Register a producer.
        
        is_published = self.marketplace.publish(id_prod, self.tea) # Publish a product.
        self.assertEqual(is_published, True) # Assert publishing was successful.

    def test_new_cart(self):
        """
        Tests the `new_cart` method for correct cart ID assignment.
        """
        id_consumer = self.marketplace.new_cart() # Create first cart.
        i = 1
        while i < 10: # Create 9 more carts.
            id_consumer = self.marketplace.new_cart()
            i = i + 1
        self.assertEqual(id_consumer, 10) # Assert that the last assigned cart ID is 10.

    def test_add_to_cart(self):
        """
        Tests the `add_to_cart` method for successful product addition and handling of unavailable products.
        """
        id_prod = self.marketplace.register_producer() # Register a producer.
        self.marketplace.publish(id_prod, self.coffee) # Publish coffee.
        id_consumer = self.marketplace.new_cart() # Create a new cart.
        
        is_added_to_cart_coffee = self.marketplace.add_to_cart(id_consumer, self.coffee) # Add published coffee.
        self.assertEqual(is_added_to_cart_coffee, True) # Assert successful addition.
        
        is_added_to_cart_tea = self.marketplace.add_to_cart(id_consumer, self.tea) # Try to add unavailable tea.
        self.assertEqual(is_added_to_cart_tea, False) # Assert unsuccessful addition.

    def test_remove_from_cart(self):
        """
        Tests the `remove_from_cart` method for correctly removing a product and returning it to the producer.
        """
        id_consumer = self.marketplace.new_cart() # Create a new cart.
        id_prod1 = self.marketplace.register_producer() # Register producer 1.
        self.marketplace.publish(id_prod1, self.coffee) # Producer 1 publishes coffee.
        id_prod2 = self.marketplace.register_producer() # Register producer 2.
        self.marketplace.publish(id_prod2, self.tea) # Producer 2 publishes tea.
        
        self.marketplace.add_to_cart(id_consumer, self.coffee) # Consumer adds coffee.
        self.marketplace.add_to_cart(id_consumer, self.tea) # Consumer adds tea.
        
        self.marketplace.remove_from_cart(id_consumer, self.coffee) # Consumer removes coffee.
        
        # Verify that only tea remains in the cart after coffee is removed.
        for key, values in self.marketplace.dictionar_cos.items():
            for value in values:
                produs = value[0] # Get the product object from the cart entry.
                self.assertEqual(produs, self.tea) # Assert it's the tea.
                break # Only checking the first item, assuming there's only one left.

    def test_place_order(self):
        """
        Tests the `place_order` method for returning the correct list of ordered products.
        """
        id_consumer = self.marketplace.new_cart() # Create a new cart.
        id_prod1 = self.marketplace.register_producer() # Register producer 1.
        self.marketplace.publish(id_prod1, self.coffee) # Producer 1 publishes coffee.
        id_prod2 = self.marketplace.register_producer() # Register producer 2.
        self.marketplace.publish(id_prod2, self.tea) # Producer 2 publishes tea.
        
        self.marketplace.add_to_cart(id_consumer, self.coffee) # Consumer adds coffee.
        self.marketplace.add_to_cart(id_consumer, self.tea) # Consumer adds tea.
        
        lista_creata = self.marketplace.place_order(id_consumer) # Place the order.
        lista = [self.coffee, self.tea] # Expected list of products.
        self.assertEqual(lista_creata, lista) # Assert the ordered list matches the expected list.


class Producer(Thread):
    """
    Represents a producer (seller) in the marketplace simulation.

    Each producer operates as a separate thread, continuously publishing products
    to the marketplace's queues. It handles delays for republishing if the queue
    is full.
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
            **kwargs: Arbitrary keyword arguments passed to the base `Thread` constructor,
                      e.g., `daemon` status and `name`.
        """
        Thread.__init__(self, **kwargs) # Initialize the base Thread class with kwargs.
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.kwargs = kwargs # Store kwargs for potential future use or debugging.
        # Register the producer with the marketplace immediately upon initialization.
        self.my_id = self.marketplace.register_producer()

    def run(self):
        """
        The main execution method for the Producer thread.

        It continuously attempts to publish its defined products to the marketplace.
        For each product type, it tries to publish the specified quantity.
        It introduces delays between successful publishes and implements a retry
        mechanism with `republish_wait_time` if the marketplace queue is full.
        """
        while True: # Continuous loop for publishing products.
            # Block Logic: Iterate through each product type this producer offers.
            for value in self.products: # `value` is a tuple: (product_obj, quantity, publish_wait_time).
                # Block Logic: Publish the specified quantity of the current product type.
                for _ in range(value[1]): # Loop for `quantity` times.
                    if self.marketplace.publish(self.my_id, value[0]): # Attempt to publish one unit of the product.
                        time.sleep(value[2]) # Wait for specified time after successful publish.
                    else:
                        # If publishing failed (e.g., producer's queue is full), wait and retry.
                        time.sleep(self.republish_wait_time)
        pass # Placeholder.


# --- Product Dataclasses ---
# These dataclasses define the structure for various product types.
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
