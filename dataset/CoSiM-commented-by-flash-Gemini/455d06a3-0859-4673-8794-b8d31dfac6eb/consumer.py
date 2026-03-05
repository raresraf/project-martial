"""
This module simulates a multi-threaded producer-consumer marketplace.

It defines classes for:
- Consumers: threads that interact with the marketplace to add/remove items from carts and place orders.
- Marketplace: manages product inventory, shopping carts, and producer registration.
- Producers: threads that publish products to the marketplace.
"""


from threading import Thread
import time


class Consumer(Thread):
    """
    Represents a consumer thread that simulates shopping activity in the marketplace.
    Consumers create carts, add/remove products, and place orders.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (list): A list of shopping cart operations (e.g., add, remove).
            marketplace (Marketplace): The marketplace instance to interact with.
            retry_wait_time (int): The time to wait before retrying an operation.
            **kwargs: Keyword arguments passed to the Thread.__init__ method.
        """

        Thread.__init__(self, **kwargs)

        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        remove_from_cart = "remove"
        add_to_cart = "add"
        
        self.cart_actions = {remove_from_cart: self.marketplace.remove_from_cart,
                             add_to_cart: self.marketplace.add_to_cart}

    def run(self):
        """
        Executes the consumer's shopping behavior.

        For each predefined cart, the consumer creates a new cart in the marketplace,
        performs add/remove operations for products, retrying if the marketplace is
        temporarily unable to fulfill the request, and finally places the order.
        """
        # Iterate through each cart defined for this consumer.
        # Invariant: Each 'cart' represents a sequence of desired operations for a single shopping session.
        for cart in self.carts:
            # Pre-condition: A new, empty shopping cart is requested from the marketplace.
            id_of_cart = self.marketplace.new_cart()
            # Iterate through each action within the current cart.
            # Invariant: Each 'action' dictates adding or removing a specific product quantity.
            for action in cart:
                index = 0
                action_quantity = action["quantity"]
                # Pre-condition: The desired quantity for the current action has not been fully processed.
                # Invariant: 'index' tracks the number of successful operations for the current product.
                while index < action_quantity:
                    action_type = action["type"]
                    action_product = action["product"]
                    # Attempt to perform the cart action (add or remove product).
                    result = self.cart_actions[action_type](id_of_cart, action_product)

                    # If the operation fails, wait and retry.
                    if result is False:
                        # Block Logic: Delays further attempts to prevent busy-waiting
                        # and to allow other threads to operate or marketplace state to change.
                        time.sleep(self.retry_wait_time)
                    # If the operation succeeds, increment the index.
                    elif result is True or result is None:
                        index += 1

            # Pre-condition: All actions for the current cart are completed or retries exhausted.
            # The consumer places the final order.
            self.marketplace.place_order(id_of_cart)



from threading import Lock, currentThread


class Marketplace:
    """
    Manages product inventory and shopping carts in a thread-safe manner.

    This class provides mechanisms for producers to publish products and for
    consumers to create carts, add/remove products, and place orders.
    It uses multiple locks to ensure data consistency across multiple threads
    due to concurrent access to shared resources.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products
                                           a single producer can have published
                                           in the marketplace at any given time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.products = [] 
        self.carts = {} 
        self.map_products_to_producer = {} 
        self.register_lock = Lock() 
        self.new_cart_lock = Lock() 
        self.products_lock = Lock() 
        self.final_lock = Lock() 
        self.cart_id = 0 

    def register_producer(self):
        """
        Registers a new producer with the marketplace.

        This method assigns a unique ID to the producer and initializes
        its product list within the marketplace. A lock ensures thread-safe
        producer registration.

        Returns:
            int: A unique producer ID.
        """
        with self.register_lock:
            producer_id = len(self.products)
            self.products.append([])
        return producer_id

    def publish(self, producer_id, product):
        """
        Publishes a product to the marketplace by a specific producer.

        This method adds a product to the producer's inventory in the marketplace,
        provided the producer has not exceeded its capacity. It is thread-safe.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product: The product to be published.

        Returns:
            bool: True if the product was successfully published, False otherwise
                  (e.g., if the producer has reached its capacity limit).
        """
        with self.products_lock:
            # Pre-condition: Check if the producer has reached its maximum capacity.
            if len(self.products[(producer_id)]) >= self.queue_size_per_producer:
                return False
            self.products[producer_id].append(product)
        self.map_products_to_producer[product] = producer_id
        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart in the marketplace.

        This method generates a unique cart ID and associates it with an empty
        list to store products. It is thread-safe.

        Returns:
            int: The ID of the newly created cart.
        """
        with self.new_cart_lock:
            self.cart_id += 1
        self.carts[self.cart_id] = []
        return self.cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a specific cart.

        This method attempts to move a product from the marketplace's available
        products to the specified shopping cart. It is a blocking operation
        that acquires a lock on the products to ensure atomicity during the
        product transfer.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product: The product to be added.

        Returns:
            bool: True if the product was successfully added, False if the
                  product was not found or could not be added.
        """
        with self.products_lock:
            # Pre-condition: Check if the product exists within any producer's inventory in the marketplace.
            if product not in [j for i in self.products for j in i]:
                return False
            # Check if the product is still associated with a producer.
            if product in self.map_products_to_producer.keys():
                # If the product is found in the producer's specific list, remove it.
                if product in self.products[self.map_products_to_producer[product]]:
                    self.products[self.map_products_to_producer[product]].remove(product)
        # Add the product to the consumer's cart.
        self.carts[cart_id].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a specific cart and returns it to the marketplace.

        This method removes the specified product from the given cart and
        re-publishes it to the marketplace by adding it back to the original
        producer's inventory. This operation is thread-safe.

        Args:
            cart_id (int): The ID of the cart from which to remove the product.
            product: The product to be removed.
        """
        # Acquire a lock to ensure exclusive access to the products' state
        # for re-publishing the product.
        with self.products_lock:
            # Remove the product from the cart.
            self.carts[cart_id].remove(product)
            # Return the product to the marketplace's available products (producer's inventory).
            self.products[self.map_products_to_producer[product]].append(product)

    def place_order(self, cart_id):
        """
        Places an order for the items in the specified cart.

        This method processes the items in the given cart, simulates printing
        an order summary, and returns the list of products that were part of the order.

        Args:
            cart_id (int): The ID of the cart to place the order for.

        Returns:
            list: A list of products that were in the ordered cart.
        """

        for product_type in range(len(self.carts[cart_id])):
            with self.final_lock:
                print(currentThread().getName(), "bought", self.carts[cart_id][product_type])

        return self.carts[cart_id]


from threading import Thread
import time


class Producer(Thread):
    """
    Represents a producer thread that continuously publishes products to the marketplace.

    Producers are responsible for making products available for consumers to purchase.
    They handle their own inventory and re-publish products if the marketplace
    is temporarily full.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (list): A list of (product, initial_quantity, wait_time) tuples
                             representing the products this producer will supply.
            marketplace (Marketplace): The marketplace instance to interact with.
            republish_wait_time (int): The time to wait before attempting to
                                       re-publish a product if the marketplace is full.
            **kwargs: Keyword arguments passed to the Thread.__init__ method.
        """
        Thread.__init__(self, **kwargs)

        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        Executes the producer's product publishing behavior.

        The producer continuously attempts to publish its defined products to the
        marketplace. If the marketplace is at capacity for this producer, it waits
        before retrying.
        """
        # The producer's main loop, ensuring continuous operation.
        # Invariant: The producer attempts to keep its products available in the marketplace.
        while True:
            # Iterate through each product type this producer is responsible for.
            # Invariant: Each 'product' tuple represents a specific product to publish along with its quantity and wait time.
            for product in self.products:
                index = 0
                product_type = product[0]
                count_prod = product[1]
                wait_time = product[2]
                # Pre-condition: There are still units of this product to publish.
                # Invariant: 'index' tracks the number of successful publications for the current product.
                while index < count_prod:
                    # Attempt to publish the product to the marketplace.
                    result = self.marketplace.publish(self.producer_id, product_type)

                    # If the operation fails, wait and retry.
                    if result is False:
                        # Block Logic: If the marketplace cannot accept more products from this producer,
                        # wait for a defined period before retrying to avoid busy-waiting.
                        time.sleep(self.republish_wait_time)
                    # If the operation succeeds, wait for the specified time and increment the index.
                    else:
                        # Block Logic: Simulates the time taken to produce/prepare the next unit of product.
                        time.sleep(wait_time)
                        index += 1
