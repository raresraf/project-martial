"""
@file consumer.py
@brief Implements a simulated e-commerce marketplace system.
This module defines classes for Producers (threads that supply products),
Consumers (threads that buy products), and a central Marketplace (managing inventory,
carts, and orders). It utilizes multi-threading to simulate concurrent buying and selling,
with explicit synchronization for marketplace operations.
"""

from threading import Thread
from time import sleep

class Consumer(Thread):
    """
    @brief Represents a consumer thread in the marketplace simulation.
    Consumers create carts, add/remove products, and place orders.
    They retry adding products if the marketplace is temporarily unable to fulfill the request.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a new Consumer thread.

        @param carts (list): A list of shopping carts, each containing a sequence of operations (add/remove product).
        @param marketplace (Marketplace): A reference to the shared Marketplace instance.
        @param retry_wait_time (float): The time (in seconds) to wait before retrying an add operation.
        @param kwargs: Additional keyword arguments passed to the Thread constructor.
        """
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        super(Consumer, self).__init__(**kwargs) # Calls the constructor of the base Thread class.

    def run(self):
        """
        @brief The main execution loop for the Consumer thread.
        Iterates through its assigned carts, performs add/remove operations, and places orders.
        """
        
        # Block Logic: Iterates through each shopping cart assigned to this consumer.
        for cart in self.carts:

            
            cart_id = self.marketplace.new_cart() # Creates a new shopping cart in the marketplace.

            # Block Logic: Processes each command (add or remove product) within the current cart.
            for command in cart:
                # Conditional Logic: Handles 'add' operations.
                if command['type'] == 'add':
                    # Block Logic: Attempts to add the product `quantity` times.
                    for i in range(command['quantity']):

                        
                        # Loop until the product is successfully added to the cart.
                        # `sleep` simulates a delay before retrying if the add operation fails.
                        while not self.marketplace.add_to_cart(cart_id, command['product']):


                            sleep(self.retry_wait_time) # Waits before retrying to add the product.

                # Conditional Logic: Handles 'remove' operations.
                elif command['type'] == 'remove':
                    # Block Logic: Attempts to remove the product `quantity` times.
                    for i in range(command['quantity']):
                        self.marketplace.remove_from_cart(cart_id, command['product'])

            
            # Places the order for the completed cart and gets the list of products in the order.
            order_list = self.marketplace.place_order(cart_id)

            
            # Block Logic: Prints the details of the products bought in the order.
            for ol in order_list:
                print(self.name, end=" ") # Prints the consumer's name.
                print("bought ", end="") # Prints "bought ".
                print(ol) # Prints the product.

>>>> file: marketplace.py

from threading import Lock, currentThread # Imports Lock for synchronization and currentThread for consumer name.

class Marketplace:
    """
    @brief Manages the central logic of the e-commerce marketplace.
    It handles producer registration, product publishing, cart management,
    and order placement.
    Uses explicit locking for critical sections to ensure thread safety.
    """

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace.

        @param queue_size_per_producer (int): The maximum number of items a producer can have
                                             waiting in the marketplace at any given time.
        """
        self.producer_count = 0  # Counter for generating unique producer IDs.
        self.producers_list = [] # List storing producer information: [producer_id, published_item_count].
        self.carts_count = 0     # Counter for generating unique cart IDs.
        self.carts_list = []     # List storing cart information: [cart_id, list_of_products_in_cart].
        self.products_q = []     # List storing available products: [unique_product_id, product_details, producer_id, status (0=available, 1=in_cart)].
        self.products_count = 0  # Counter for generating unique product IDs within the marketplace.
        self.queue_size_per_producer = queue_size_per_producer # Max items per producer.
        self.lock = Lock()       # Global lock to protect shared marketplace data.

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace.

        @return int: A unique producer ID.
        """
        self.lock.acquire() # Acquires lock to protect shared data.
        self.producer_count = self.producer_count + 1
        self.producers_list.append([self.producer_count, 0]) # Adds new producer with 0 published items.
        self.lock.release() # Releases the lock.
        return self.producer_count

    def publish(self, producer_id, product):
        """
        @brief Publishes a product to the marketplace by a producer.

        @param producer_id (int): The ID of the producer publishing the product.
        @param product (Product): The product to publish.
        @return bool: True if the product was successfully published, False otherwise (e.g., queue full).
        """
        self.lock.acquire() # Acquires lock to protect shared data.
        
        # Block Logic: Finds the specific producer in the list.
        for p in self.producers_list:
            if p[0] == producer_id:

                # Conditional Logic: Checks if the producer has exceeded its queue size limit.
                if p[1] == self.queue_size_per_producer:
                    self.lock.release() # Releases the lock before returning.
                    return False # Publication failed due to queue size limit.

                self.products_count = self.products_count + 1 # Increments global product count.

                # Appends the product to the marketplace's available products list.
                # Format: [unique_marketplace_product_id, product_details, producer_id, status]
                self.products_q.append(
                    [self.products_count, product, producer_id, 0]) # Status 0 means available.
                p[1] = p[1] + 1 # Increments the count of items published by this producer.
                self.lock.release() # Releases the lock.
                return True
        self.lock.release() # Releases the lock if producer not found (shouldn't happen with proper registration).
        return False # Should not be reached if producer is registered.

    def new_cart(self):
        """
        @brief Creates a new empty shopping cart in the marketplace.

        @return int: A unique cart ID.
        """
        self.lock.acquire() # Acquires lock to protect shared data.
        self.carts_count = self.carts_count + 1
        cart_id = self.carts_count
        self.carts_list.append([cart_id]) # Initializes an empty list for the new cart.
        self.lock.release() # Releases the lock.

        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a product to a specific shopping cart.

        @param cart_id (int): The ID of the cart to add to.
        @param product (Product): The product to add.
        @return bool: True if the product was successfully added, False if the product is not available.
        """
        self.lock.acquire() # Acquires lock to protect shared data.
        
        # Block Logic: Finds the specific cart.
        for c in self.carts_list:
            if c[0] == cart_id:

                # Block Logic: Searches for the product in the available products queue.
                for i in self.products_q:

                    # Conditional Logic: If the product matches and is available (status 0).
                    if i[1] == product and i[3] == 0:

                        c.append(i) # Adds the product (full entry) to the cart.

                        i[3] = 1 # Marks the product as 'in cart' (status 1).
                        self.lock.release() # Releases the lock.
                        return True # Product successfully added.
                self.lock.release() # Releases the lock if product not found or not available.
                return False # Product not available in products_q or already in a cart.
        self.lock.release() # Releases the lock if cart not found (shouldn't happen).
        return False # Cart not found.

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a specific shopping cart and makes it available again in the marketplace.

        @param cart_id (int): The ID of the cart to remove from.
        @param product (Product): The product to remove.
        """
        self.lock.acquire() # Acquires lock to protect shared data.
        
        # Block Logic: Finds the specific cart.
        for c in self.carts_list:
            if c[0] == cart_id:

                # Block Logic: Searches for the product within the marketplace's product queue.
                for i in self.products_q:
                    if i[1] == product: # Matches the product.

                        # Block Logic: Finds the product within the cart's list of products.
                        for x, y in enumerate(c[1:]): # Iterates through products in the cart (skipping cart_id).
                            if y[1] == product: # If the product matches.

                                c.pop(x+1) # Removes the product from the cart.

                                i[3] = 0 # Marks the product as 'available' again (status 0).
                                break # Exits inner loop.
                        break # Exits middle loop.
                break # Exits outer loop.
        self.lock.release() # Releases the lock.

    def place_order(self, cart_id):
        """
        @brief Places an order for a given cart, removing products from the marketplace inventory
        and updating producer counts.

        @param cart_id (int): The ID of the cart to place an order for.
        @return list: The list of product details in the placed order.
        """
        order_list = []

        self.lock.acquire() # Acquires lock to protect shared data during order placement.
        
        # Block Logic: Finds the specific cart.
        for cart in self.carts_list:
            if cart[0] == cart_id:

                products = cart[1:] # Extracts the list of products from the cart.

                # Block Logic: Processes each product in the order.
                for pr in products:

                    # Block Logic: Finds the specific product entry in the marketplace's product queue.
                    for x, y in enumerate(self.products_q):
                        if y[0] == pr[0]: # Matches by unique marketplace product ID.

                            self.products_q.pop(x) # Removes the product from the marketplace's available products.

                            # Block Logic: Updates the item count for the producer of this product.
                            for producer in self.producers_list:
                                if y[2] == producer[0]: # Matches by producer ID.

                                    producer[1] = producer[1] - 1 # Decrements producer's item count.
                                    break # Exits inner loop.
                            break # Exits middle loop.

                    order_list.append(pr[1]) # Adds the product details to the order list.
                self.lock.release() # Releases the lock.
                return order_list # Returns the list of products in the order.
        self.lock.release() # Releases the lock if cart not found (shouldn't happen).
        return order_list # Returns empty list if cart not found.

>>>> file: producer.py


from threading import Thread
from time import sleep

class Producer(Thread):
    """
    @brief Represents a producer thread in the marketplace simulation.
    Producers continuously publish products to the marketplace.
    They retry publishing if the marketplace is temporarily unable to accept the product.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a new Producer thread.

        @param products (list): A list of (product, quantity, wait_time) tuples
                                representing the products this producer offers.
        @param marketplace (Marketplace): A reference to the shared Marketplace instance.
        @param republish_wait_time (float): The time (in seconds) to wait before retrying a publish operation.
        @param kwargs: Additional keyword arguments passed to the Thread constructor.
        """
        super(Producer, self).__init__(**kwargs) # Calls the constructor of the base Thread class.
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

    def run(self):
        """
        @brief The main execution loop for the Producer thread.
        Continuously attempts to publish its products to the marketplace.
        """
        
        self.id = self.marketplace.register_producer() # Registers itself with the marketplace and gets a unique ID.

        while True: # Infinite loop for continuous production.
            # Block Logic: Iterates through each product type this producer offers.
            for p in self.products:
                q = p[1] # Quantity of the current product to publish.
                # Block Logic: Attempts to publish each product `q` times.
                for i in range(q):

                    
                    # Attempts to publish the product to the marketplace.
                    published = self.marketplace.publish(self.id, p[0])

                    
                    # Conditional Logic: If publishing failed (e.g., marketplace queue full).
                    if not published:
                        i = i - 1 # Decrements 'i' to retry publishing this item in the next iteration.
                        sleep(self.republish_wait_time) # Waits before retrying to publish.
                    sleep(p[2]) # Simulates a delay between publishing individual items.


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Base class for products in the marketplace.
    Uses `dataclasses.dataclass` for concise definition.
    `frozen=True` makes instances immutable.
    """
    name: str  # Name of the product.
    price: int # Price of the product.


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Represents a Tea product, inheriting from Product.
    Adds a specific attribute for tea.
    """
    type: str  # Type of tea (e.g., "Green", "Black").


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Represents a Coffee product, inheriting from Product.
    Adds specific attributes for coffee.
    """
    acidity: str      # Acidity level of the coffee.
    roast_level: str  # Roast level of the coffee.
