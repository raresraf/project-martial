


import time
from threading import Thread

from tema.marketplace import Marketplace

class Consumer(Thread):
    """
    Represents a consumer in a marketplace simulation.

    Each consumer operates as a separate thread, creating shopping carts,
    performing add and remove operations on products based on a list of
    predefined actions, and finally placing orders. It handles retries
    for add operations that might temporarily fail (e.g., due to product unavailability).
    """
    

    def __init__(self,
                 carts: list,
                 marketplace: Marketplace,
                 retry_wait_time: int,
                 **kwargs) \
    :
        """
        Initializes a new Consumer instance.

        Args:
            carts (list): A list of shopping carts, where each cart is a list of
                          product operations (dictionaries containing 'type', 'product', 'quantity').
            marketplace (Marketplace): The marketplace object with which
                                       this consumer will interact.
            retry_wait_time (int): The time in seconds to wait before
                                     retrying a failed 'add to cart' operation.
            **kwargs: Arbitrary keyword arguments passed to the Thread constructor.
        """
        super().__init__(**kwargs) # Call the base class (Thread) constructor.

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        The main execution method for the consumer thread.

        It iterates through each predefined shopping cart. For each cart, it
        creates a new cart in the marketplace, then processes each operation
        (add or remove products). Add operations are retried if they fail (e.g.,
        product not available). After all operations for a cart are processed,
        the order is placed, and details of purchased products are printed.
        """
        # Iterate through each shopping cart definition provided to this consumer.
        for cart_definition in self.carts:
            # Create a new cart in the marketplace and get its unique ID.
            cart_id = self.marketplace.new_cart()

            # Process each action (add or remove product) within the current cart definition.
            for action in cart_definition:
                type_ = action['type']      # Type of operation ('add' or 'remove').
                product = action['product'] # The product involved in the operation.
                qty = action['quantity']    # The quantity for the operation.

                # Loop to perform the operation 'qty' number of times.
                for _ in range(qty):
                    if type_ == 'add':
                        # Continuously try to add the product until successful.
                        # If add_to_cart returns False (e.g., product unavailable),
                        # wait and retry.
                        while not self.marketplace.add_to_cart(cart_id, product):
                            time.sleep(self.retry_wait_time)
                    elif type_ == 'remove':
                        # Remove the product from the cart. This operation is assumed
                        # to always succeed if the product is in the cart.
                        self.marketplace.remove_from_cart(cart_id, product)

            # After processing all actions for the current cart, place the order.
            order = self.marketplace.place_order(cart_id)

            # Print each product that was successfully bought by this consumer.
            for product in order:
                print(f'{self.name} bought {product}')


from threading import Lock

from tema.product import Product

class Marketplace:
    """
    Simulates a central marketplace for producers and consumers to interact.

    It manages producer product queues and consumer shopping carts, providing
    thread-safe operations for producer registration, product publishing,
    cart creation, adding/removing products from carts, and placing orders.
    """
    

    def __init__(self, queue_size_per_producer: int):
        """
        Initializes the Marketplace with configuration for producer queues.

        Args:
            queue_size_per_producer (int): The maximum number of products
                                           a single producer can have in its
                                           queue within the marketplace.
        """
        self.queue_size_per_producer = queue_size_per_producer

        # List of producer queues. Each element is a tuple: (list of products, Lock for that queue).
        self.producer_queues = []
        # List of consumer carts. Each element is a list of (product, producer_id) tuples.
        self.consumer_carts = []

        # Lock to protect the registration of new producers.
        self.register_producer_lock = Lock()
        # Lock to protect the creation of new carts.
        self.new_cart_lock = Lock()

    def register_producer(self) -> int:
        """
        Registers a new producer with the marketplace and assigns a unique ID.

        The producer's queue is also initialized.

        Returns:
            int: The unique producer ID.
        """
        with self.register_producer_lock:
            # Assign a new producer ID based on the current number of registered producers.
            producer_id = len(self.producer_queues)
            # Initialize an empty product list and a lock for this new producer's queue.
            self.producer_queues.append(([], Lock()))

        return producer_id

    def publish(self, producer_id: int, product: Product) -> bool:
        """
        Allows a producer to publish a product to the marketplace.

        The product is added only if the producer has not exceeded its
        configured queue size limit. Access to the producer's queue is
        protected by its individual lock.

        Args:
            producer_id (int): The ID of the producer.
            product (Product): The product to be published.

        Returns:
            bool: True if the product was published successfully, False otherwise.
        """
        queue, lock = self.producer_queues[producer_id]

        with lock:
            # Check if the producer's queue is full.
            if len(queue) >= self.queue_size_per_producer:
                return False

            # Add the product to the producer's queue.
            queue.append(product)

        return True

    def new_cart(self) -> int:
        """
        Creates a new shopping cart for a consumer and assigns a unique cart ID.

        Returns:
            int: The unique cart ID.
        """
        with self.new_cart_lock:
            # Assign a new cart ID based on the current number of consumer carts.
            cart_id = len(self.consumer_carts)
            # Initialize an empty list for the new cart.
            self.consumer_carts.append([])

        return cart_id

    def add_to_cart(self, cart_id: int, product: Product) -> bool:
        """
        Adds a product to a specified shopping cart.

        This operation involves searching through producer queues to find and
        remove the product, then adding it to the consumer's cart.
        Each producer's queue is protected by its own lock.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (Product): The product to add.

        Returns:
            bool: True if the product was successfully added, False otherwise.
        """
        cart = self.consumer_carts[cart_id]

        # Iterate through all producer queues to find the product.
        for producer_id, (queue, lock) in enumerate(self.producer_queues):
            with lock:
                try:
                    # Attempt to remove the product from the producer's queue.
                    queue.remove(product) 
                except ValueError:
                    # If the product is not in this producer's queue, continue to the next producer.
                    continue

            # If the product was successfully removed from a producer's queue, add it to the cart.
            cart.append((product, producer_id))

            return True # Product successfully added.

        return False # Product not found in any producer's queue.

    def remove_from_cart(self, cart_id: int, product: Product) -> bool:
        """
        Removes a product from a specified shopping cart and returns it to
        its original producer's queue.

        Args:
            cart_id (int): The ID of the cart to remove the product from.
            product (Product): The product to remove.

        Returns:
            bool: True if the product was successfully removed, False otherwise.
        """
        cart = self.consumer_carts[cart_id]

        # Iterate through items in the cart to find the product to remove.
        for i, (prod, producer_id) in enumerate(cart):
            if prod == product:
                del cart[i] # Remove the product from the cart.

                # Return the product to its original producer's queue.
                queue, lock = self.producer_queues[producer_id]

                with lock:
                    queue.append(prod) # Add the product back to the producer's queue.

                return True # Product successfully removed.

        return False # Product not found in the cart.


    def place_order(self, cart_id) -> list:
        """
        Retrieves the products from a specified shopping cart.

        In this implementation, the cart's contents are returned.
        The actual printing of what was bought is handled by the consumer thread.

        Args:
            cart_id (int): The ID of the cart to retrieve products from.

        Returns:
            list: A list of products that were in the cart.
        """
        

        cart = self.consumer_carts[cart_id]

        return [product for product, producer_id in cart]
        """
        Processes a shopping cart, effectively "buying" the items and printing
        what was bought. The cart is then cleared from the marketplace.

        Args:
            cart_id (int): The ID of the cart to place the order for.

        Returns:
            list: A list of products that were in the cart (a copy of the purchased items).
        """
        # Retrieve the cart and remove it from the marketplace's list of carts.
        cart = self.consumer_carts[cart_id] # This line might lead to IndexError if cart_id is out of bounds or cart is already placed
        self.consumer_carts[cart_id] = [] # Clear the cart for the given cart_id
        
        # NOTE: The original code uses .pop(cart_id, None) which would remove the cart from the dictionary.
        # However, consumer_carts is a list, so .pop would work by index, not key.
        # This implementation simply returns the products and clears the list for the given index.

        # The current implementation of place_order in the original code does not remove the cart,
        # but clears the list at that index. This should be consistent with how cart_id is generated.

        # For printing, it accesses currentThread().getName() which is used by Thread.
        for prod, producer_id in cart: # Iterate through (product, producer_id) tuples in the cart.
            # print(f'{currentThread().getName()} bought {prod}') # Assuming current thread is Consumer.
            # Using lock for print to prevent interleaved output from multiple consumers.
            with Lock(): # Using a local Lock for print, potentially different from self.lock_for_print
                print(f'{currentThread().getName()} bought {prod}') # Assuming currentThread().getName() is the consumer thread name.
        
        return [product for product, producer_id in cart] # Return only the products, not the (product, producer_id) tuples.


import time



from threading import Thread

from tema.marketplace import Marketplace

class Producer(Thread):
    """
    Represents a producer in a marketplace simulation.

    Each producer operates as a separate thread, continuously attempting to
    publish products to the marketplace. It manages a list of products to
    publish, retrying if the marketplace's limits are reached, and waits
    for specified durations between publishing attempts.
    """
    

    def __init__(self,
                 products: list,
                 marketplace: Marketplace,
                 republish_wait_time: int,
                 **kwargs) \
    :
        """
        Initializes a new Producer instance.

        Args:
            products (list): A list of products this producer will offer,
                             each item being a tuple (product_object, quantity, wait_time).
            marketplace (Marketplace): The marketplace object with which
                                       this producer will interact.
            republish_wait_time (int): The time in seconds to wait before
                                         retrying to publish a product if the
                                         marketplace queue is full.
            **kwargs: Arbitrary keyword arguments passed to the Thread constructor.
        """
        super().__init__(**kwargs) # Call the base class (Thread) constructor.

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time

        self.id_ = self.marketplace.register_producer() # Register with marketplace and get a unique ID.

    def run(self):
        """
        The main execution method for the producer thread.

        It continuously iterates through its list of products. For each product,
        it attempts to publish the specified quantity to the marketplace. If
        publishing fails (e.g., marketplace queue is full), it waits for
        `republish_wait_time` before retrying. If successful, it waits for
        `product_wait_time` before attempting to publish the next item.
        The outer `while True` loop ensures continuous operation until the
        program terminates externally.
        """
        while True:
            # Iterate through the list of products this producer is offering.
            for product, qty, wait_time in self.products:
                # Loop to publish the specified quantity of the current product.
                for _ in range(qty):
                    time.sleep(wait_time) # Wait for the specified time before attempting to publish.

                    # Continuously try to publish the product until successful.
                    # If publish returns False (marketplace queue full), wait and retry.
                    while not self.marketplace.publish(self.id_, product):
                        time.sleep(self.republish_wait_time)
