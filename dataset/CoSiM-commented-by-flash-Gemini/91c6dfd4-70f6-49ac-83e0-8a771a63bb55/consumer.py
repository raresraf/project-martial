




import time
from threading import Thread


class Consumer(Thread):
    

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a new Consumer thread.

        @param carts: A list of cart operations for this consumer to perform.
                      Each cart is a list of dictionaries, where each dictionary
                      represents an operation (e.g., {"type": "add", "product": product_obj, "quantity": 1}).
        @param marketplace: The Marketplace instance to interact with.
        @param retry_wait_time: The time in seconds to wait before retrying a failed operation.
        @param kwargs: Arbitrary keyword arguments passed to the Thread.__init__ method.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs # Stores additional keyword arguments, though not directly used in run().

    def run(self):
        """
        @brief The main execution loop for the Consumer thread.

        This method initializes a new cart, iterates through the predefined
        cart operations (add/remove products), handling retries for failed
        "add" operations, and finally places the order.
        """
        id_cart = self.marketplace.new_cart() # Block Logic: Creates a new shopping cart in the marketplace for this consumer.

        for cart in self.carts: # Block Logic: Iterates through each defined shopping cart scenario.
            for command in cart: # Block Logic: Iterates through each command (add/remove) within the current cart.
                # Pre-condition: Checks if the command type is "add".
                if command["type"] == "add":
                    # Block Logic: Attempts to add the product multiple times based on the specified quantity.
                    for i in range(command["quantity"]):
                        available = self.marketplace.add_to_cart(id_cart, command["product"]) 
                        # Block Logic: If the product is not immediately available, wait and retry.
                        while not available:
                            time.sleep(self.retry_wait_time) # Inline: Pauses before retrying the "add" operation.
                            available = self.marketplace.add_to_cart(id_cart, command["product"])
                else: # Block Logic: Handles "remove" command.
                    # Block Logic: Attempts to remove the product multiple times based on the specified quantity.
                    for i in range(command["quantity"]):
                        self.marketplace.remove_from_cart(id_cart, command["product"]) # Functional Utility: Removes product from cart.

        # Block Logic: Once all operations for the current cart are complete, the order is placed.
        self.marketplace.place_order(id_cart)

from threading import Lock


class Marketplace:
    """
    @brief Simulates a central marketplace for producers and consumers, with granular locking.

    This class manages product listings, producer registrations, shopping carts,
    and order placement. It uses multiple Lock objects to ensure thread-safe
    access to different aspects of the marketplace, such as adding/removing
    products from inventory, creating carts, and modifying producer/cart data.
    """
    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes a new Marketplace instance.

        @param queue_size_per_producer: The maximum number of items a single producer
                                       can have listed in the marketplace at any time.
        """
        self.queue_size = queue_size_per_producer # Maximum products per producer.
        self.producer_id = 0                      # Counter for assigning unique producer IDs.
        self.cart_id = 0                          # Counter for assigning unique cart IDs.
        self.market = [[]]                        # List of lists representing producer inventories.
                                                  # Each inner list stores products published by a producer.
        self.cart = [[]]                          # List of lists representing shopping carts.
                                                  # Each inner list stores (product, producer_id) tuples.
        self.lock_add = Lock()                    # Lock for protecting add_to_cart operations.
        self.lock_remove = Lock()                 # Lock for protecting remove_from_cart operations.
        self.lock_cart = Lock()                   # Lock for protecting cart creation (new_cart).
        self.lock_producer = Lock()               # Lock for protecting producer registration.
        self.lock_print = Lock()                  # Lock for protecting print statements during order placement.

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace and assigns a unique ID.

        @return The unique integer ID assigned to the newly registered producer.
        """
        self.lock_producer.acquire() # Pre-condition: Acquire lock to ensure exclusive access for producer registration.
        self.producer_id += 1        # Increment the global producer ID counter.
        self.market.append([])       # Post-condition: Add a new empty list for the new producer's inventory.
        self.lock_producer.release() # Post-condition: Release lock.
        return self.producer_id

    def publish(self, producer_id, product):
        """
        @brief Publishes a product to the marketplace by a specific producer.

        The product is published only if the producer has not exceeded its
        queue size limit.

        @param producer_id: The ID of the producer publishing the product.
        @param product: The product object to be published.
        @return True if the product was successfully published, False otherwise.
        """
        # Pre-condition: Checks if the producer has reached their item limit.
        # producer_id - 1 is used because market list is 0-indexed.
        if len(self.market[producer_id - 1]) >= self.queue_size:
            return False
        # Post-condition: Add the product to the producer's inventory.
        self.market[producer_id - 1].append(product)
        return True

    def new_cart(self):
        """
        @brief Creates a new shopping cart and assigns a unique cart ID.

        @return The unique integer ID of the newly created cart.
        """
        self.lock_cart.acquire()     # Pre-condition: Acquire lock to ensure exclusive access for cart creation.
        self.cart_id += 1            # Increment the global cart ID counter.
        self.cart.append([])         # Post-condition: Add a new empty list for the new cart.
        self.lock_cart.release()     # Post-condition: Release lock.
        return self.cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a product to a specified shopping cart.

        If the product is available in any producer's inventory, it's moved from
        the marketplace's inventory to the cart.

        @param cart_id: The ID of the cart to add the product to.
        @param product: The product object to add.
        @return True if the product was successfully added, False if not found.
        """
        self.lock_add.acquire() # Pre-condition: Acquire lock to ensure exclusive access for adding to cart.
        # Block Logic: Iterate through all producer inventories to find the product.
        for i in range(len(self.market)):
            for j in range(len(self.market[i])):
                # Pre-condition: If the product is found in a producer's inventory.
                if self.market[i][j] == product: 
                    # Post-condition: Add the product and its producer ID to the cart.
                    self.cart[cart_id - 1].append((product, i))
                    self.market[i].remove(product) # Remove product from producer's inventory.
                    self.lock_add.release() # Release lock.
                    return True
        self.lock_add.release() # Release lock if product not found.
        return False 

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a specified shopping cart and returns it to the marketplace.

        @param cart_id: The ID of the cart to remove the product from.
        @param product: The product object to remove.
        """
        self.lock_remove.acquire() # Pre-condition: Acquire lock to ensure exclusive access for removing from cart.
        # Block Logic: Iterate through the specified cart to find and remove the product.
        for i in range(len(self.cart[cart_id - 1])):
            # Pre-condition: If the product is found in the cart.
            if self.cart[cart_id - 1][i][0] == product: 
                # Post-condition: Return the product to its original producer's inventory.
                self.market[self.cart[cart_id - 1][i][1]].append(product)
                prod_id = self.cart[cart_id - 1][i][1] # Get producer ID.
                self.cart[cart_id - 1].remove((product, prod_id)) # Remove product from cart.
                break # Exit loop after removing.

        self.lock_remove.release() # Post-condition: Release lock.

    def place_order(self, cart_id):
        """
        @brief Places an order for all items currently in the specified cart.

        Prints a message indicating the purchase for each item in the cart.

        @param cart_id: The ID of the cart for which to place the order.
        """
        self.lock_print.acquire() # Pre-condition: Acquire lock to ensure exclusive access for printing.
        # Block Logic: Iterate through items in the cart and print purchase messages.
        for i in range(len(self.cart[cart_id - 1])):
            print("cons"+ str(cart_id) + " bought " + str(self.cart[cart_id - 1][i][0]))
        self.lock_print.release() # Post-condition: Release lock.


from threading import Thread
import time

class Producer(Thread):
    """
    @brief Simulates a producer agent that publishes products to a marketplace.

    This thread registers itself with the marketplace, and then continuously
    attempts to publish a predefined list of products, waiting and retrying
    if the marketplace is temporarily unable to accept more products from it.
    """
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a new Producer thread.

        @param products: A list of products (product object, quantity, wait time)
                         for this producer to publish.
        @param marketplace: The Marketplace instance to interact with.
        @param republish_wait_time: The time in seconds to wait before retrying to publish a product.
        @param kwargs: Arbitrary keyword arguments passed to the Thread.__init__ method.
        """
        Thread.__init__(self, **kwargs)
        self.marketplace = marketplace
        self.product_list = products # List of products to be published.

        self.wait_time = republish_wait_time # Time to wait before retrying to publish.
        self.kwargs = kwargs # Stores additional keyword arguments, though not directly used in run().

    def run(self):
        """
        @brief The main execution loop for the Producer thread.

        This method registers the producer with the marketplace, then
        iterates through its list of products, attempting to publish them
        to the marketplace. It handles retries if publishing fails.
        """
        id_producer = self.marketplace.register_producer() # Pre-condition: Registers the producer with the marketplace to obtain a unique ID.
        while 1: # Infinite loop for continuous production.
            for prod in self.product_list: # Block Logic: Iterates through each product in the producer's list.
                for i in range(prod[1]): # Block Logic: Attempts to publish the product multiple times based on its quantity.
                    can_add = self.marketplace.publish(id_producer, prod[0]) # Functional Utility: Attempts to publish the product.
                    # Block Logic: If publishing is not immediately successful, wait and retry.
                    if not can_add:
                        while not can_add:
                            time.sleep(self.wait_time) # Inline: Pauses before retrying the publish operation.
                            can_add = self.marketplace.publish(id_producer, prod[0])
                    else:
                        time.sleep(prod[2]) # Inline: Pauses for a specified time after successfully publishing a product.
