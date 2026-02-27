


"""
@file consumer.py
@brief This module implements a producer-consumer simulation involving a marketplace for products.

@details It defines `Consumer` and `Producer` threads that interact with a shared
         `Marketplace` to simulate buying and selling. The `Marketplace` manages
         product inventories and shopping carts, ensuring thread safety with locks.
         Additionally, data classes (`Product`, `Tea`, `Coffee`) are defined
         to represent various product types in the simulation.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    @brief Represents a consumer thread that interacts with the marketplace to buy products.

    @details Each `Consumer` thread simulates a buyer by iterating through a list of
             carts, creating a new cart in the marketplace for each, adding and removing
             products based on predefined actions, and finally placing an order.
             It includes retry logic for adding products to handle marketplace capacity constraints.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a new Consumer thread.

        @param carts A list of dictionaries, where each dictionary represents a shopping cart
                     with products to add or remove.
        @param marketplace The shared `Marketplace` instance with which the consumer interacts.
        @param retry_wait_time The time in seconds to wait before retrying an `add_to_cart` operation.
        @param kwargs Additional keyword arguments passed to the `Thread` constructor (e.g., `name`).
        """
        
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs # Store kwargs to print consumer's name.


    def run(self):
        """
        @brief The main execution loop for the Consumer thread.

        @details This method iterates through each predefined shopping cart. For each cart,
                 it creates a new cart in the marketplace, then processes each product
                 action (add or remove). Adding products includes a retry mechanism.
                 Finally, it places the order and prints the items successfully bought.
        """
        for cart in self.carts:
            cart_id = self.marketplace.new_cart() # Create a new shopping cart in the marketplace.
            for product in cart:
                for _ in range(product["quantity"]): # Process each product multiple times if quantity > 1.
                    if product["type"] == "add":
                        # Block Logic: Continuously attempt to add the product to the cart until successful.
                        # Invariant: The loop retries if add_to_cart returns False (e.g., product not available).
                        while not self.marketplace.add_to_cart(cart_id, product["product"]):
                            time.sleep(self.retry_wait_time) # Wait before retrying.

                    elif product["type"] == "remove":
                        # Block Logic: Remove a product from the cart.
                        self.marketplace.remove_from_cart(cart_id, product["product"])

            # Block Logic: Place the order for the current cart and print bought items.
            bought = self.marketplace.place_order(cart_id) # Finalize the purchase.
            for item in bought:
                # Print the consumer's name and the item bought.
                print(self.kwargs['name'], "bought", item)

from threading import Lock
import collections

class Marketplace:
    """
    @brief Simulates a marketplace where producers publish products and consumers manage carts and place orders.

    @details This class acts as a central hub for all product-related transactions.
             It maintains separate buffers for each producer, manages unique cart IDs,
             and handles the logic for adding, removing, and purchasing products.
             Thread safety is ensured through the use of `threading.Lock` for critical sections.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes a new Marketplace instance.

        @param queue_size_per_producer The maximum number of products each producer can have in its buffer.
        """
        
        self.queue_size_per_producer = queue_size_per_producer # Max capacity for each producer's buffer.
        self.id_producer = 0 # Counter for assigning unique producer IDs.
        self.cart_ids = 0    # Counter for assigning unique cart IDs.
        # Dictionary to store products published by each producer. Key: producer_id, Value: list of products.
        self.producers_buffers = collections.defaultdict(list) 
        # Dictionary to store items in each shopping cart. Key: cart_id, Value: list of (product, producer_id) tuples.
        self.carts = collections.defaultdict(list) 
        # Locks for protecting access to shared counters and data structures during concurrent operations.
        self.register_producer_lock = Lock() 
        self.new_cart_lock = Lock()
        self.add_to_cart_lock = Lock()
        self.remove_from_cart_lock = Lock()


    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace and assigns a unique ID.

        @details This method atomically increments a producer ID counter to provide
                 a unique identifier for each new producer.
        @return A string representing the unique producer ID.
        """
        
        # Block Logic: Atomically assign a new producer ID.
        with self.register_producer_lock:
            producer_id = self.id_producer
            self.id_producer += 1

        
        return str(producer_id)


    def publish(self, producer_id, product):
        """
        @brief Publishes a product from a producer to the marketplace.

        @details The product is added to the producer's buffer. If the producer's
                 buffer is already full (exceeds `queue_size_per_producer`),
                 the product is not published.
        @param producer_id The ID of the producer publishing the product.
        @param product The product object to be published.
        @return `True` if the product was successfully published, `False` otherwise (buffer full).
        """
        
        # Precondition: Check if the producer exists and if their buffer has capacity.
        if producer_id in self.producers_buffers:
            # Block Logic: Check if the producer's buffer is full.
            if len(self.producers_buffers[producer_id]) >= self.queue_size_per_producer:
                return False # Buffer is full, cannot publish.

        # Add the product to the producer's buffer.
        self.producers_buffers[producer_id].append(product)
        return True


    def new_cart(self):
        """
        @brief Creates a new empty shopping cart in the marketplace and assigns a unique ID.

        @details This method atomically increments a cart ID counter to provide
                 a unique identifier for each new cart.
        @return An integer representing the unique cart ID.
        """
        
        # Block Logic: Atomically assign a new cart ID.
        with self.new_cart_lock:
            cart_id = self.cart_ids
            self.cart_ids += 1

        
        return cart_id


    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a product to a specified shopping cart.

        @details This method searches through all producers' buffers for the requested
                 product. If found, it moves the product from the producer's buffer
                 to the consumer's cart.
        @param cart_id The ID of the cart to add the product to.
        @param product The product object to add.
        @return `True` if the product was found and added, `False` otherwise.
        """
        
        # Block Logic: Iterate through all producers' buffers to find the product.
        for producer_id, products in self.producers_buffers.items():
            for prod in products:
                if product == prod: # If the product is found.
                    # Block Logic: Add the product to the cart and remove it from the producer's buffer.
                    self.carts[cart_id].append((product, producer_id)) # Store product and its original producer.
                    products.remove(product) # Remove from producer's buffer.
                    return True # Product successfully added.

        # Product not found in any producer's buffer.
        return False


    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a specified shopping cart and returns it to the producer's buffer.

        @details This method searches for the product in the given `cart_id`. If found,
                 it removes the product from the cart and places it back into the
                 original producer's buffer.
        @param cart_id The ID of the cart to remove the product from.
        @param product The product object to remove.
        """
        
        # Block Logic: Iterate through items in the cart to find the product.
        for item in self.carts[cart_id]:
            # Precondition: If the product is found in the cart.
            if product == item[0]:
                self.carts[cart_id].remove(item) # Remove from the cart.
                self.producers_buffers[item[1]].append(product) # Return to producer's buffer.
                return # Product successfully removed and returned.


    def place_order(self, cart_id):
        """
        @brief Finalizes a shopping cart, effectively "buying" its contents.

        @details This method retrieves all items currently in the specified cart.
                 It signifies the completion of a purchase for this cart.
        @param cart_id The ID of the cart to place the order for.
        @return A list of product objects that were in the cart at the time of order placement.
        """
        
        # Return a list of just the product objects from the cart.
        return [i[0] for i in self.carts[cart_id]]


from threading import Thread
import time

class Producer(Thread):
    """
    @brief Represents a producer thread that continuously publishes products to the marketplace.

    @details Each `Producer` thread is responsible for generating and publishing a
             defined set of products to the `Marketplace`. It handles a unique
             producer ID assigned by the marketplace and includes retry logic
             if the marketplace buffer is full.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a new Producer thread.

        @param products A list of tuples, where each tuple contains (product_object, quantity, wait_time).
                        `quantity` is how many times to publish the product, and `wait_time` is
                        the delay after each successful publication.
        @param marketplace The shared `Marketplace` instance with which the producer interacts.
        @param republish_wait_time The time in seconds to wait before retrying to publish a product.
        @param kwargs Additional keyword arguments passed to the `Thread` constructor (e.g., `name`).
        """
        
        Thread.__init__(self, **kwargs)
        self.products = products # List of products to publish.
        self.marketplace = marketplace # Reference to the shared marketplace.
        self.republish_wait_time = republish_wait_time # Delay for republishing.
        self.kwargs = kwargs # Stored for potential use (e.g., printing producer's name).
        self.id_producer = self.marketplace.register_producer() # Register with marketplace and get a unique ID.


    def run(self):
        """
        @brief The main execution loop for the Producer thread.

        @details This method continuously attempts to publish its predefined products
                 to the marketplace. For each product, it publishes it a specified
                 number of times, waiting between publications. If the marketplace's
                 buffer for this producer is full, it retries after a short delay.
        """
        
        # Block Logic: The producer continuously tries to publish its products.
        while True:
            for product_tuple in self.products: # Iterate through each type of product this producer offers.
                product_obj = product_tuple[0]
                product_quantity = product_tuple[1]
                product_wait_time = product_tuple[2]

                for _ in range(product_quantity): # Publish each product the specified number of times.
                    # Block Logic: Continuously attempt to publish the product until successful.
                    # Invariant: The loop retries if marketplace.publish returns False (e.g., producer's buffer is full).
                    while not self.marketplace.publish(self.id_producer, product_obj):
                        time.sleep(self.republish_wait_time) # Wait before retrying.

                    # Wait for a specified time after successfully publishing a product.
                    time.sleep(product_wait_time)


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Base data class representing a generic product.

    @details This dataclass provides a common structure for all products in the
             marketplace simulation, defining essential attributes like name and price.
             It is frozen to ensure immutability.
    @param name The name of the product.
    @param price The price of the product.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Data class representing a specific type of product: Tea.

    @details Inherits from `Product` and adds a specific attribute for tea products.
             It is frozen to ensure immutability.
    @param type The type of tea (e.g., "Green", "Black", "Herbal").
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Data class representing a specific type of product: Coffee.

    @details Inherits from `Product` and adds specific attributes for coffee products.
             It is frozen to ensure immutability.
    @param acidity The acidity level of the coffee.
    @param roast_level The roast level of the coffee.
    """
    acidity: str
    roast_level: str
