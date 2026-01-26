


"""
This module implements a simulation of a marketplace with producers and consumers.
Producers continuously publish products to the marketplace, and consumers
interact with the marketplace to create carts, add/remove products,
and place orders. Concurrency is managed using threading and locks.
"""

from threading import Thread
import time


class Consumer(Thread):
    """
    Represents a consumer in the marketplace. Consumers create shopping carts,
    add and remove products, and ultimately place orders. Operations are
    retried if the marketplace is temporarily unavailable.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        :param carts: A list of lists, where each inner list contains actions
                      (add/remove product) to be performed for a timepoint.
        :param marketplace: The Marketplace instance to interact with.
        :param retry_wait_time: The time in seconds to wait before retrying
                                an 'add to cart' operation.
        :param kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.market = marketplace
        self.wait_time = retry_wait_time
        self.carts = carts

    def run(self):
        """
        The main execution method for the consumer thread. It creates a new cart,
        processes a series of add and remove actions for products, and then
        places the final order, printing the purchased products.
        """
        # Create a new shopping cart for this consumer.
        cart = self.market.new_cart()

        # Iterate through the list of product actions (carts represents timepoints or action groups).
        for cart_list in self.carts:
            # Process each action within the current action group.
            for act in cart_list:
                # Handle 'add' actions.
                if act["type"] == "add":
                    # Keep trying to add the product until the desired quantity is met.
                    quantity_to_add = act["quantity"]
                    while quantity_to_add > 0:
                        # Attempt to add the product to the cart.
                        success = self.market.add_to_cart(cart, act["product"])
                        if success:
                            quantity_to_add -= 1
                        else:
                            # If adding fails, wait and retry.
                            time.sleep(self.wait_time)
                # Handle 'remove' actions.
                else:
                    # Remove the specified quantity of the product from the cart.
                    quantity_to_remove = act["quantity"]
                    for _ in range(quantity_to_remove):
                        self.market.remove_from_cart(cart, act["product"])
        
        # Place the order with the accumulated items in the cart.
        order_products = self.market.place_order(cart)

        # Print the list of products successfully bought by this consumer.
        for product in order_products:
            print(self.name + " bought " + str(product))


from threading import Lock


class Marketplace:
    """
    The central marketplace where producers publish products and consumers
    manage their shopping carts. It handles product inventory, cart management,
    and thread synchronization using locks.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        :param queue_size_per_producer: The maximum number of products a single
                                        producer can have in the marketplace's buffer.
        """
        self.consumer_lock = Lock()  # Lock for consumer-related operations.
        self.buffer = []  # List of product buffers, one for each producer.
        self.carts = []  # List of shopping carts, indexed by cart_id.
        self.producer_lock = Lock()  # Lock for producer-related operations.
        self.producer_id_counter = -1 # Counter for assigning unique producer IDs.


        self.cart_id_counter = -1   # Counter for assigning unique cart IDs.
        self.queue_size = queue_size_per_producer # Max items per producer in buffer.


    def register_producer(self):
        """
        Registers a new producer with the marketplace, assigns a unique ID,
        and creates a dedicated buffer for its products.

        :return: The unique ID assigned to the new producer.
        """
        # Acquire lock to ensure unique producer ID assignment and buffer creation.
        self.producer_lock.acquire()
        self.producer_id_counter += 1
        self.buffer.append([])  # Create a new buffer for this producer.
        self.producer_lock.release()
        return self.producer_id_counter

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.
        The product is added to the producer's specific buffer if there's space.

        :param producer_id: The ID of the producer publishing the product.
        :param product: The product to be published.
        :return: True if the product was successfully published, False otherwise.
        """
        # Acquire lock to protect producer buffer modification.
        self.producer_lock.acquire()
        # Check if the producer's buffer has reached its capacity.
        if len(self.buffer[producer_id]) >= self.queue_size:
            self.producer_lock.release()
            return False
        
        # Structure the product information for storage.
        to_add = {
            'product': product,
            'producer_id': producer_id
        }
        self.buffer[producer_id].append(to_add) # Add product to buffer.
        self.producer_lock.release()
        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart and assigns it a unique ID.

        :return: The unique ID of the newly created cart.
        """
        # Acquire lock to ensure unique cart ID assignment and cart creation.
        self.consumer_lock.acquire()
        self.cart_id_counter += 1
        self.carts.append([])  # Add a new empty cart.
        self.consumer_lock.release()
        return self.cart_id_counter


    def add_to_cart(self, cart_id, product):
        """
        Attempts to add a specified product from the marketplace buffer to a consumer's cart.
        The first available instance of the product is moved.

        :param cart_id: The ID of the cart to add the product to.
        :param product: The product to be added.
        :return: True if the product was successfully added, False if not found.
        """
        # Acquire lock to ensure atomic cart and buffer modification.
        self.consumer_lock.acquire()
        # Search through all producer buffers for the requested product.
        for i in range(len(self.buffer)):
            for product_aux in self.buffer[i]:
                if product == product_aux['product']:
                    # If found, remove it from the producer's buffer.
                    self.buffer[i].remove(product_aux)
                    # Add it to the consumer's cart.
                    self.carts[cart_id].append(product_aux)
                    self.consumer_lock.release()
                    return True
        self.consumer_lock.release()
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a specified product from a consumer's cart and returns it
        to the original producer's buffer in the marketplace.

        :param cart_id: The ID of the cart to remove the product from.
        :param product: The product to be removed.
        """
        # Acquire lock for atomic cart and buffer modification.
        self.consumer_lock.acquire()
        # Search through the specified cart for the product to remove.
        for product_aux in self.carts[cart_id]:
            if product_aux['product'] == product: 
                # Return the product to its original producer's buffer.
                self.buffer[product_aux['producer_id']].append(product_aux)
                # Remove it from the cart.
                self.carts[cart_id].remove(product_aux)
                break # Assume only one instance is removed per call.
        self.consumer_lock.release()


    def place_order(self, cart_id):
        """
        Finalizes the order for a given cart, extracting the product names.

        :param cart_id: The ID of the cart to place the order for.
        :return: A list of product names that were in the placed order.
        """
        order = []
        # Extract only the product names from the items in the cart.
        for prod in self.carts[cart_id]:
            order.append(prod['product'])
        return order


from threading import Thread
import time

# Note: The import 'from httplib2 import ProxiesUnavailableError' appears to be unused
# and might be a remnant from previous development or an oversight.

class Producer(Thread):
    """
    Represents a producer that continuously publishes products to the marketplace.
    Products are published with a specified quantity and at a certain frequency.
    Retries publishing if the marketplace buffer is full.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        :param products: A list of products this producer will offer. Each product
                         is represented as a tuple: (product_object, quantity, wait_time).
        :param marketplace: The Marketplace instance to interact with.
        :param republish_wait_time: The time in seconds to wait before retrying
                                    to publish a product if the buffer is full.
        :param kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.market = marketplace
        self.wait_time = republish_wait_time
        self.products = products
        
        # Register with the marketplace to get a unique producer ID.
        self.id = self.market.register_producer()

    def run(self):
        """
        The main execution method for the producer thread. It continuously
        publishes products according to its product list, quantities, and wait times,
        handling marketplace buffer limitations with retries.
        """
        product_index = 0 # Index to track the current product in the list.
        # Initial sleep before the first product is published, based on the first product's wait_time.
        # Note: This logic assumes 'products' is not empty and product[2] is a sleep duration.
        time.sleep(self.products[product_index][2]) 
        
        # Infinite loop for continuous production.
        while True:
            # Iterate through each defined product for this producer.
            for product_info in self.products:
                # Publish the specified quantity of the current product.
                for _ in range(product_info[1]): # product_info[1] is the quantity.
                    # Continuously attempt to publish until successful.
                    while not self.market.publish(self.id, product_info[0]): # product_info[0] is the product object.
                        # If publishing fails (buffer full), wait and retry.
                        time.sleep(self.wait_time)
                    
                    # Wait for the specified time before publishing the next instance of the product.
                    time.sleep(product_info[2]) # product_info[2] is the wait time after publishing.


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    Base class for all products in the marketplace.
    It's a frozen dataclass, meaning instances are immutable.
    """
    name: str  # The name of the product.
    price: int # The price of the product.


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    Represents a specific type of product: Tea.
    Inherits from Product and adds a 'type' attribute.
    """
    type: str  # The type of tea (e.g., 'Green', 'Black', 'Herbal').


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    Represents a specific type of product: Coffee.
    Inherits from Product and adds 'acidity' and 'roast_level' attributes.
    """
    acidity: str      # The acidity level of the coffee.
    roast_level: str  # The roast level of the coffee.
