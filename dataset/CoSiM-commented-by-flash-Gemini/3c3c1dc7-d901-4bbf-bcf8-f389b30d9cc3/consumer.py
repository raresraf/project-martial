"""
This module implements a multi-threaded producer-consumer simulation for a marketplace.
It defines classes for Consumer, Marketplace, and Producer, simulating interactions
where producers publish items and consumers process carts, adding and removing products.
Synchronization mechanisms are employed to ensure thread-safe operations.
"""


from threading import Thread
import time


class Consumer(Thread):
    """
    The Consumer class represents a buyer in the marketplace.
    Each consumer runs as a separate thread, simulating the process of
    creating carts, adding and removing items based on a predefined list of operations,
    and finally attempting to place an order. It incorporates retry logic for operations
    that might fail due to marketplace state (e.g., product unavailability).
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        :param carts: A list of cart specifications. Each cart is a list of operation dictionaries,
                      where each operation specifies "type" (add/remove), "product", and "quantity".
        :param marketplace: The shared marketplace instance to interact with.
        :param retry_wait_time: The time in seconds to wait before retrying a failed operation
                                 (e.g., adding a product to a cart if unavailable).
        :param kwargs: Additional keyword arguments passed to the Thread constructor,
                       e.g., 'name' for the consumer's identifier.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def run(self):
        """
        Executes the consumer's main logic.
        This method is called when the thread starts. It simulates the consumer's
        journey through the marketplace: for each predefined cart, it creates a new cart
        in the marketplace, processes all add/remove operations, and then places the order.
        Operations that fail are retried after a delay.
        """

        # Block Logic: Iterates through each predefined cart (a list of operations) for this consumer.
        for cart in self.carts:

            # Functional Utility: Creates a new shopping cart in the marketplace and obtains its ID.
            cart_id = self.marketplace.new_cart()

            # Block Logic: Processes each individual operation (add or remove a product) within the current cart.
            for operation in cart:
                cart_operations = 0 # Counter for the number of operations successfully performed for the current product.
                quantity = operation["quantity"] # The total quantity of the product to operate on.

                # Invariant: Continues to loop until the desired quantity of the product has been processed.
                while cart_operations < quantity:

                    operation_name = operation["type"] # Determines whether to 'add' or 'remove'.
                    product = operation["product"] # The product involved in the operation.

                    # Conditional Logic: Executes either 'add_to_cart' or 'remove_from_cart' based on operation type.
                    if operation_name == "add":
                        ret = self.marketplace.add_to_cart(cart_id, product)
                    elif operation_name == "remove":
                        ret = self.marketplace.remove_from_cart(cart_id, product)

                    # Conditional Logic: If the operation was successful (ret is True or None for remove).
                    if ret is None or ret:
                        cart_operations += 1 # Increments the count of successful operations for this product.
                    else:
                        time.sleep(self.retry_wait_time) # Pauses before retrying if the operation failed.

            # Functional Utility: Places the final order for the completed cart.
            self.marketplace.place_order(cart_id)



from threading import Lock, currentThread


class Marketplace:
    """
    The Marketplace class simulates a central hub where producers publish products
    and consumers can add/remove products from their carts to place orders.
    It manages product inventory, producer and consumer registration, and cart operations.
    Synchronization mechanisms (locks) are used to ensure thread-safe access to shared resources.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace with a specified queue size per producer.

        :param queue_size_per_producer: The maximum number of products a single producer
                                        can have available in the marketplace at any given time.
        """
        self.queue_size_per_producer = queue_size_per_producer

        # Data Structure: A list to hold all products currently available in the marketplace.
        # Products are added by producers and removed by consumers.
        self.products = []
        
        # Data Structure: A dictionary to store consumer carts.
        # Key: cart_id, Value: list of products in that cart.
        self.carts = {}

        # Data Structure: A list to track the number of products each producer currently has in the marketplace.
        # The index corresponds to the producer_id.
        self.nr_prod_in_queue = []

        # Data Structure: A dictionary to map products to their original producer_id.
        # This is used when a product is returned to the marketplace (e.g., removed from cart).
        self.products_owners = {}

        self.nr_carts = 0 # Counter for assigning unique cart IDs.

        # Synchronization: Lock to protect access to the 'products' list and 'products_owners' dictionary.
        self.lock_products_queue = Lock()

        # Synchronization: Lock to protect access to the 'nr_carts' counter.
        self.lock_nr_carts = Lock()


    def register_producer(self):
        """
        Registers a new producer with the marketplace and assigns a unique ID.
        This also initializes the product count for the new producer to zero.

        :return: A unique integer ID for the newly registered producer.
        """
        producer_id = len(self.nr_prod_in_queue) # Assigns a new producer ID based on the current number of producers.
        self.nr_prod_in_queue.append(0) # Initializes product count for this producer to 0.

        return producer_id


    def publish(self, producer_id, product):
        """
        Publishes a product from a given producer to the marketplace.
        The product is added only if the producer has not exceeded its maximum
        allowed products in the marketplace. This method is not fully thread-safe
        with respect to `self.products` and `self.products_owners` as it doesn't acquire
        `self.lock_products_queue` before modifying them.

        :param producer_id: The ID of the producer publishing the product.
        :param product: The product object to be published.
        :return: True if the product was successfully published, False otherwise (e.g., queue full).
        """

        # Conditional Logic: Checks if the producer has not exceeded its allowed product limit in the marketplace.
        if self.nr_prod_in_queue[producer_id] < self.queue_size_per_producer:
            self.products.append(product) # Adds the product to the marketplace's general product list.
            self.nr_prod_in_queue[producer_id] += 1 # Increments the count of products from this producer.
            self.products_owners[product] = producer_id # Records the producer as the owner of this product.

            return True # Indicates successful publication.

        return False # Indicates that the producer's queue limit was reached.

    def new_cart(self):
        """
        Creates a new, empty shopping cart for a consumer and assigns a unique ID.
        This method is thread-safe for incrementing the cart counter.

        :return: A unique integer ID for the new cart.
        """
        # Synchronization: Acquires a lock to safely increment the cart counter.
        with self.lock_nr_carts:
            cart_id = self.nr_carts # Assigns a new cart ID.
            self.nr_carts += 1 # Increments the total number of carts.
            

        self.carts[cart_id] = [] # Initializes an empty list for the new cart in the 'carts' dictionary.
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a specified product to a consumer's cart from the marketplace's available products.
        If the product is found, it's moved from the marketplace to the cart,
        and the producer's product count is decremented. This method is thread-safe.

        :param cart_id: The ID of the consumer's cart.
        :param product: The product object to add to the cart.
        :return: True if the product was successfully added, False if not found or unavailable.
        """

        # Synchronization: Acquires a lock to ensure thread-safe modification of the shared products list.
        with self.lock_products_queue:
            
            # Conditional Logic: Checks if the desired product is currently available in the marketplace.
            if product in self.products:

                # Functional Utility: Removes the product from the marketplace's general product list.
                self.products.remove(product)

                # Functional Utility: Retrieves the original producer's ID for this product.
                producer_id = self.products_owners[product]
                self.nr_prod_in_queue[producer_id] -= 1 # Decrements the producer's count of products in queue.

                # Functional Utility: Adds the product to the specified consumer's cart.
                self.carts[cart_id].append(product)

                return True # Indicates successful addition to cart.

        return False # Indicates that the product was not found in the marketplace.

    def remove_from_cart(self, cart_id, product):
        """
        Removes a specified product from a consumer's cart and returns it to the marketplace.
        The product is re-added to the marketplace's general product list, and the original
        producer's product count is incremented. This method implicitly assumes the product
        is in the cart and does not handle cases where it might not be.
        It has a potential race condition for `self.products` and `self.nr_prod_in_queue` if not called
        within a lock from the caller.

        :param cart_id: The ID of the consumer's cart.
        :param product: The product object to remove from the cart.
        :return: True (always, as currently implemented) or None if not found, depending on `products.remove` behavior.
        """

        self.carts[cart_id].remove(product) # Functional Utility: Removes the product from the consumer's cart.
        
        self.products.append(product) # Functional Utility: Appends the product back to the marketplace's general product list.

        # Synchronization: Acquires a lock to ensure thread-safe modification of the shared products list and producer counts.
        with self.lock_products_queue:
            
            producer_id = self.products_owners[product] # Retrieves the original producer's ID.
            self.nr_prod_in_queue[producer_id] += 1 # Increments the producer's count of products in queue.
        
        return True # Explicitly returning True, but the original code was implicitly returning None.

    def place_order(self, cart_id):
        """
        Finalizes the order for a given cart.
        This involves printing the items bought by the current thread (consumer)
        and then removing the cart from the marketplace.

        :param cart_id: The ID of the cart to place the order for.
        :return: The list of products that were in the cart, or None if the cart was not found.
        """
        products_list = self.carts.pop(cart_id, None) # Retrieves and removes the cart's products list.
        # This line `order = self.carts.pop(cart_id, None)` is redundant as the cart has already been popped.
        # If products_list is not None, order will be None.
        order = self.carts.pop(cart_id, None) 
        # Block Logic: Iterates through the products in the placed order and prints a message.
        for product in products_list:
            print(currentThread().getName(), "bought", product)

        return order # Returns the (potentially empty or None) order.


from threading import Thread
import time

class Producer(Thread):
    """
    The Producer class represents a seller in the marketplace.
    Each producer runs as a separate thread, continuously publishing products
    to the marketplace based on its predefined inventory, quantity, and timing.
    Producers will retry publishing if the marketplace's capacity for them is full.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        :param products: A list of product operations. Each operation is a tuple
                         (product_object, quantity, wait_time_after_publish).
        :param marketplace: The shared marketplace instance to interact with.
        :param republish_wait_time: The time in seconds to wait before retrying to publish
                                    a product if the marketplace is full for this producer.
        :param kwargs: Additional keyword arguments passed to the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products # The inventory and publishing schedule of products for this producer.
        self.marketplace = marketplace # Reference to the shared marketplace.
        self.republish_wait_time = republish_wait_time # Time to wait before retrying to publish.
        self.producer_id = self.marketplace.register_producer() # Registers the producer and gets its unique ID.


    def run(self):
        """
        Executes the producer's main logic.
        This method is called when the thread starts. It continuously iterates
        through its product list, attempting to publish each product to the marketplace.
        It includes retry logic if the marketplace refuses publication (e.g., due to capacity limits).
        """
        # Invariant: The producer continuously attempts to publish products.
        while True:
            # Block Logic: Iterates through each defined product operation in the producer's schedule.
            for (product, quantity, wait_time) in self.products:
                added_products = 0 # Counter for products successfully added in the current batch.

                # Invariant: Continues to loop until the desired quantity of the product has been published.
                while added_products < quantity:
                    # Conditional Logic: Attempts to publish the product.
                    ret = self.marketplace.publish(self.producer_id, product)
                    if ret: # If publishing was successful.
                        time.sleep(wait_time) # Pauses for a specified time after publishing a product.
                        added_products += 1 # Increments the count of successfully added products.
                    else: # If publishing failed (e.g., producer's queue is full).
                        time.sleep(self.republish_wait_time) # Pauses before retrying to publish.

