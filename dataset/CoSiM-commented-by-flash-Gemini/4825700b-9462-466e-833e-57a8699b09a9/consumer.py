




"""
@4825700b-9462-466e-833e-57a8699b09a9/consumer.py
@brief This module defines `Consumer`, `Marketplace`, and `Producer` classes
for simulating a multi-threaded e-commerce system with concurrent
product publishing, cart management, and order placement.
"""

from threading import Thread
import time


class Consumer(Thread):
    """
    Represents a consumer in the e-commerce simulation.

    Consumers create shopping carts, add/remove products, and place orders
    within the marketplace. Each consumer operates as a separate thread.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping cart command lists for this consumer.
            marketplace (Marketplace): The shared marketplace instance.
            retry_wait_time (float): Time to wait before retrying an action if it fails.
            **kwargs: Keyword arguments passed to the Thread constructor,
                      e.g., `name` for the thread's name.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs["name"] # Assign the thread's name.

    def add_product(self, cart_id, product):
        """
        Attempts to add a product to a specified cart, retrying if necessary.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (str): The name of the product to add.
        """
        added = False
        # Block Logic: Continuously attempt to add the product until successful.
        while not added:
            added = self.marketplace.add_to_cart(cart_id, product)
            if not added:
                time.sleep(self.retry_wait_time) # Wait before retrying if addition failed.

    def run(self):
        """
        The main execution method for the consumer thread.

        It processes a list of cart commands, creating new carts,
        adding/removing products, and finally placing orders.
        """
        carts_id = [] # List to store the IDs of carts created by this consumer.
        
        # Block Logic: Process each predefined cart for this consumer.
        for cart in self.carts:
            cart_id = self.marketplace.new_cart() # Create a new cart in the marketplace.
            carts_id.append(cart_id) # Store the new cart's ID.
            
            # Block Logic: Execute commands (add or remove products) for the current cart.
            for command in cart:
                if command["type"] == "add":
                    # Add products based on the specified quantity.
                    for _ in range(command["quantity"]):
                        self.add_product(cart_id, command["product"])
                else:
                    # Remove products based on the specified quantity.
                    for _ in range(command["quantity"]):
                        self.marketplace.remove_from_cart(cart_id, command["product"])

        # Block Logic: Place orders for all carts created by this consumer.
        for cart_id in carts_id:
            products = self.marketplace.place_order(cart_id) # Place the order and get the list of purchased products.
            # Output the purchased products.
            for product in products:
                print(f'{self.name} bought {product}', flush=True)




from threading import Lock


class Marketplace:
    """
    Manages products from multiple producers, handles shopping carts for consumers,
    and processes orders in a thread-safe manner.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of items a producer can
                                           have in the marketplace's inventory at one time.
        """
        self.queue_size_per_producer = queue_size_per_producer

        # Lock for synchronizing producer registration and ID assignment.
        self.register_lock = Lock()
        self.producer_id = 0 # Counter for assigning unique producer IDs.

        # Lock for synchronizing cart creation and ID assignment.
        self.cart_lock = Lock()
        self.cart_id = 0 # Counter for assigning unique cart IDs.

        # List to store dictionaries of products for each producer.
        # Each inner dictionary maps product name to its quantity.
        self.products = []
        
        # List to store shopping carts. Each cart is a dictionary mapping (product, producer_id) to quantity.
        self.carts = []

        # List to store current sizes (total products) for each producer's inventory.
        self.sizes = []
        
        # List to store individual locks for each producer's inventory, ensuring thread-safe access.
        self.producers_lock = []

    def register_producer(self):
        """
        Registers a new producer with the marketplace and returns a unique producer ID.

        Initializes inventory and a dedicated lock for the new producer.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        # Block Logic: Acquire lock to ensure atomic assignment of producer ID.
        self.register_lock.acquire()
        id_copy = self.producer_id
        self.producer_id = self.producer_id + 1
        self.register_lock.release()

        # Initialize data structures for the new producer.
        self.products.append({}) # New inventory for the producer.
        self.sizes.append(0) # Initial size of the producer's inventory.
        self.producers_lock.append(Lock()) # New lock for the producer's inventory.

        return id_copy

    def publish(self, producer_id, product):
        """
        Attempts to publish a product from a producer to the marketplace.

        The product is added only if the producer's inventory size is below
        the `queue_size_per_producer` limit.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (str): The name of the product to publish.

        Returns:
            bool: True if the product was successfully published, False otherwise.
        """
        # Block Logic: Acquire the producer's specific lock to ensure thread-safe inventory modification.
        self.producers_lock[producer_id].acquire()
        # Pre-condition: Check if the producer's inventory is not full.
        if self.sizes[producer_id] < self.queue_size_per_producer:

            # Block Logic: Increment product quantity in the producer's inventory.
            if product in self.products[producer_id]:
                self.products[producer_id][product] += 1
            else:
                self.products[producer_id][product] = 0 # Initialize if product is new.

            # Inline: Increment the total size of the producer's inventory.
            self.sizes[producer_id] += 1
            self.producers_lock[producer_id].release() # Release the producer's lock.
            return True # Product successfully published.

        self.producers_lock[producer_id].release() # Release the producer's lock.
        return False # Publication failed due to full inventory.

    def new_cart(self):
        """
        Creates a new, empty shopping cart and returns its unique ID.

        Returns:
            int: The unique ID of the newly created cart.
        """
        # Block Logic: Acquire lock to ensure atomic assignment of cart ID.
        self.cart_lock.acquire()
        id_copy = self.cart_id
        self.cart_id = self.cart_id + 1
        self.cart_lock.release()

        # Initialize an empty dictionary for the new cart.
        self.carts.append({})

        return id_copy

    def add_to_cart(self, cart_id, product):
        """
        Attempts to add a product to a consumer's shopping cart.

        This involves finding an available product from any producer, decrementing
        the producer's inventory, and adding the product to the specified cart.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (str): The name of the product to add.

        Returns:
            bool: True if the product was successfully added, False if not found.
        """
        # Block Logic: Iterate through all producers to find an available product.
        for producer_id in range(len(self.products)):

            # Block Logic: Acquire the producer's lock to check and modify inventory.
            self.producers_lock[producer_id].acquire()
            # Pre-condition: Check if the product exists in the current producer's inventory.
            if product in self.products[producer_id]:

                # Action: Decrement the product quantity in the producer's inventory.
                self.products[producer_id][product] -= 1
                self.sizes[producer_id] -= 1 # Decrement the total size of the producer's inventory.

                # If product quantity becomes zero, remove it from the producer's inventory.
                if self.products[producer_id][product] == 0:
                    self.products[producer_id].pop(product)
                self.producers_lock[producer_id].release() # Release the producer's lock.

                # Block Logic: Add the product to the consumer's cart.
                if (product, producer_id) in self.carts[cart_id]:
                    new_quantity = self.carts[cart_id].get((product, producer_id)) + 1
                    self.carts[cart_id].update({(product, producer_id): new_quantity})
                else:
                    self.carts[cart_id].update({(product, producer_id): 1})

                return True # Product successfully added to cart.

            self.producers_lock[producer_id].release() # Release the producer's lock if product not found with this producer.
        return False # Product not found in any producer's inventory.

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a consumer's shopping cart and returns it to the
        original producer's inventory.

        Args:
            cart_id (int): The ID of the cart to remove the product from.
            product (str): The name of the product to remove.
        """
        producer_id = -1
        # Block Logic: Find the producer from which the product was originally taken.
        for product_tuple in self.carts[cart_id].keys():
            if product == product_tuple[0]:
                producer_id = product_tuple[1]
                break

        # Block Logic: Acquire the producer's lock and return the product to its inventory.
        self.producers_lock[producer_id].acquire()
        if product in self.products[producer_id]:
            self.products[producer_id][product] += 1
        else:
            self.products[producer_id][product] = 0 # Initialize if product was completely removed.

        # Inline: Increment the total size of the producer's inventory.
        self.sizes[producer_id] += 1
        self.producers_lock[producer_id].release() # Release the producer's lock.

        # Block Logic: Decrement the product quantity in the consumer's cart.
        new_quantity = self.carts[cart_id].get((product, producer_id)) - 1
        self.carts[cart_id].update({(product, producer_id): new_quantity})
        # If product quantity in cart becomes zero, remove it from the cart.
        if self.carts[cart_id].get((product, producer_id)) == 0:
            self.carts[cart_id] = {key: val for key, val in self.carts[cart_id].items()
                                   if key != (product, producer_id)}

    def place_order(self, cart_id):
        """
        Processes an order for a given cart, consolidating all products into a
        simple list for the consumer.

        Args:
            cart_id (int): The ID of the cart for which to place the order.

        Returns:
            list: A list of product names representing the final order.
        """
        simple_list = []

        # Block Logic: Iterate through the items in the cart and flatten them into a simple list.
        for product_tuple in self.carts[cart_id]:
            for _ in range(self.carts[cart_id][product_tuple]):
                simple_list.append(product_tuple[0]) # Add each product by its name.

        return simple_list




from threading import Thread
import time


class Producer(Thread):
    """
    Represents a producer in the e-commerce simulation.

    Producers generate products and attempt to publish them to the marketplace,
    retrying if the marketplace's queue for that producer is full.
    Each producer operates as a separate thread.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of products this producer will publish,
                             each product being a tuple (name, quantity, publish_delay).
            marketplace (Marketplace): The shared marketplace instance.
            republish_wait_time (float): Time to wait before retrying to publish a product.
            **kwargs: Keyword arguments passed to the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products # List of products to be published by this producer.
        self.marketplace = marketplace # The marketplace instance to publish to.
        self.republish_wait_time = republish_wait_time # Delay before retrying publication.
        self.producer_id = marketplace.register_producer() # Register with the marketplace and get a unique ID.

    def run(self):
        """
        The main execution method for the producer thread.

        It continuously iterates through its list of products, attempting to
        publish each one to the marketplace, with delays for both successful
        publications and retries.
        """
        while True:
            # Block Logic: Iterate through each product this producer is responsible for.
            for product in self.products:
                published = False
                # Block Logic: Continuously attempt to publish the product until successful.
                while not published:
                    published = self.marketplace.publish(self.producer_id, product[0])
                    
                    # Pre-condition: If publication failed, wait before retrying.
                    if not published:
                        time.sleep(self.republish_wait_time)
                    else:
                        # Post-condition: If successful, wait for the product's defined publish delay.
                        time.sleep(product[2])
