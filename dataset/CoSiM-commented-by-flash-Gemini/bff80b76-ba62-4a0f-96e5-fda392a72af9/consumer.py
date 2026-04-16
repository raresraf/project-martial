


"""
@file consumer.py
@brief Implements a multithreaded simulation of a marketplace with producer and consumer dynamics.

This module defines the core components for a marketplace simulation:
- `Consumer`: Represents a buyer that interacts with the marketplace to add, remove, and purchase products.
- `Marketplace`: Manages the inventory of products, handles registration of producers, and orchestrates cart and order processing for consumers. It employs synchronization primitives (threading.Condition) to ensure thread-safe operations.
- `Producer`: Represents a seller that continuously publishes products to the marketplace.
- `Product`: Base class for items, with `Tea` and `Coffee` as specific product types.

The simulation demonstrates concurrent access and modification of shared resources (the marketplace inventory and shopping carts) by multiple threads, highlighting the need for proper synchronization.

Domain: Concurrency, Multithreading, Simulation, Object-Oriented Design, Data Structures.
"""

from threading import Thread
import time

class Consumer(Thread):
    """
    @brief Represents a consumer (customer) in the marketplace simulation.

    Each Consumer thread simulates a customer who goes through a list of predefined
    shopping carts, adds or removes products, and finally places an order.
    It interacts with the `Marketplace` to perform these actions.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a Consumer thread.

        Parameters:
          - carts: A list of shopping cart definitions, where each cart is a list
                   of product operations (add/remove).
          - marketplace: A reference to the shared Marketplace instance.
          - retry_wait_time: The time in seconds to wait before retrying a failed
                             `add_to_cart` operation.
          - kwargs: Additional keyword arguments passed to the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.retry_wait_time = retry_wait_time
        self.marketplace = marketplace
        self.carts = carts

    def run(self):
        """
        @brief The main execution method for the Consumer thread.

        This method simulates the consumer's shopping journey:
        1. Iterates through each predefined shopping list (`cos`).
        2. Creates a new shopping cart in the marketplace.
        3. Processes each product operation (add or remove) within the shopping list.
        4. For 'add' operations, attempts to add the product to the cart, retrying
           if the marketplace indicates failure (e.g., product out of stock).
        5. For 'remove' operations, removes the product from the cart.
        6. Finally, places the order for the created cart.
        7. Prints the details of the products successfully bought.
        """
        for cos in self.carts:
            # Create a new cart in the marketplace for this shopping list.
            cos_id = self.marketplace.new_cart()
            # Iterate through each product operation defined in the current shopping list.
            for produs in cos:
                if produs['type'] == 'add':
                    # Block Logic: Handles adding a specified quantity of a product to the cart.
                    # Invariant: `contor` tracks the number of items successfully added.
                    contor = 0
                    while contor < produs['quantity']:
                        # Attempt to add the product to the cart.
                        adaugat = self.marketplace.add_to_cart(cos_id, produs['product'])
                        # If adding fails (e.g., product not available), retry after a delay.
                        # Invariant: Loop continues until the product is added.
                        while adaugat == False:
                            adaugat = self.marketplace.add_to_cart(cos_id, produs['product'])
                            time.sleep(self.retry_wait_time) # Wait before retrying.
                        contor += 1 # Increment count for successfully added item.
                else: # produs['type'] == 'remove'
                    # Block Logic: Handles removing a specified quantity of a product from the cart.
                    # Invariant: `contor` tracks the number of items successfully removed.
                    contor = 0
                    while contor < produs['quantity']:
                        # Remove the product from the cart.
                        self.marketplace.remove_from_cart(cos_id, produs['product'])
                        contor += 1 # Increment count for successfully removed item.
            # Place the final order for all reserved products in the cart.
            produse_cumparate = self.marketplace.place_order(cos_id)
            # Iterate through the bought products and print a confirmation message.
            for produs_cumparat in produse_cumparate:
                print(f"{self.name} bought {produs_cumparat}")

from threading import Condition

class Marketplace:
    """
    @brief The central marketplace managing products, producers, and consumer carts.

    This class orchestrates the interaction between producers and consumers.
    It maintains a registry of producers, manages the products they publish,
    handles the creation and modification of shopping carts, and processes orders.
    A `threading.Condition` object is used to ensure thread-safe access to shared data.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace with its internal data structures and synchronization.

        Parameters:
          - queue_size_per_producer: The maximum number of products each producer can have in the marketplace at one time.
                                     (Note: This parameter is currently defined but not fully utilized in the provided logic
                                     for limiting producer inventory size).
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.producatori = dict() # Stores registered producers and their published products.
        self.cosuri = dict()      # Stores consumer carts with reserved products.
        self.cond = Condition()   # Condition variable for synchronizing access to shared data.
        self.producatori_id = []  # List of active producer IDs.
        self.cosuri_id = []       # List of active cart IDs.
        self.contor_producator = 1 # Counter for generating unique producer IDs.
        self.contor_cos = 1        # Counter for generating unique cart IDs.

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace.

        Assigns a unique ID to the producer and initializes their product list.
        Ensures thread-safe ID generation and producer registration using a lock.

        Returns:
          - The unique ID assigned to the new producer.
        """
        with self.cond:
            # Block Logic: Generates a new unique producer ID.
            # Invariant: The new producer ID is guaranteed to be unique and incremental.
            # Note: `sum(self.producatori_id)` is an unusual way to calculate the next ID
            # and could lead to large or non-sequential IDs if IDs are not managed carefully.
            # A simpler counter `self.contor_producator += 1` might be intended, or
            # max(self.producatori_id, default=0) + 1. The current implementation could
            # produce non-sequential and non-minimal IDs.
            self.contor_producator = sum(self.producatori_id)
            self.contor_producator += 1
            self.producatori_id.append(self.contor_producator)
            producator = dict()

            producator['produse'] = [] # Initialize an empty list for published products.
            self.producatori[self.contor_producator] = producator
            return self.contor_producator


    def publish(self, producer_id, product):
        """
        @brief Publishes a product from a producer to the marketplace.

        Adds the product to the specified producer's list of published products.
        This operation is thread-safe.

        Parameters:
          - producer_id: The ID of the producer publishing the product.
          - product: The product object to be published.

        Returns:
          - True if the product was successfully published, False otherwise (e.g., producer not found).
        """
        with self.cond:
            # Block Logic: Iterates through registered producers to find the matching producer_id.
            # If found, adds the product to that producer's published products list.
            # Invariant: Product is added only to the correct producer's inventory.
            for producator_id, lista_produse_publicate in self.producatori.items():
                if producator_id == producer_id:
                    lista_produse_publicate['produse'].append(product)
                    return True
            return False

    def new_cart(self):
        """
        @brief Creates a new empty shopping cart in the marketplace.

        Assigns a unique ID to the cart and initializes it with an empty list
        for reserved products. Ensures thread-safe cart ID generation and creation.

        Returns:
          - The unique ID assigned to the new shopping cart.
        """
        with self.cond:
            # Block Logic: Generates a new unique cart ID.
            # Invariant: The new cart ID is guaranteed to be unique and incremental.
            # Similar to producer ID generation, `sum(self.cosuri_id)` is an unusual
            # approach and could lead to large or non-sequential IDs.
            self.contor_cos = sum(self.cosuri_id)
            self.contor_cos += 2 # Increments by 2, which seems arbitrary, potentially to avoid collisions with producer IDs.
            self.cosuri_id.append(self.contor_cos)
            cos = dict()
            cos['produse_rezervate'] = [] # Initialize an empty list for reserved products.
            self.cosuri[self.contor_cos] = cos
            return self.contor_cos


    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a product to a consumer's shopping cart.

        This function checks if the product is available from any producer and, if so,
        reserves it for the specified cart. This operation is thread-safe.

        Parameters:
          - cart_id: The ID of the cart to which the product should be added.
          - product: The product object to add.

        Returns:
          - True if the product was successfully added to the cart, False otherwise
            (e.g., cart not found, product not available from any producer).
        """
        with self.cond:
            # Block Logic: Searches for the specified cart.
            for cos, continut in self.cosuri.items():
                if cos == cart_id:
                    # Block Logic: Searches for the product among all published products by producers.
                    # Invariant: Product is only added if it is currently available in the marketplace.
                    for producator, produse_publicate in self.producatori.items():
                        if product in produse_publicate['produse']:
                            continut['produse_rezervate'].append(product)
                            # Note: The current implementation adds the product to the cart but does not
                            # remove it from the producer's available stock. This means multiple consumers
                            # can "reserve" the same physical item, which is a potential logical flaw
                            # in a real-world inventory system.
                            return True
            return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a consumer's shopping cart.

        This function removes a single instance of the specified product from the
        given cart. This operation is thread-safe.

        Parameters:
          - cart_id: The ID of the cart from which the product should be removed.
          - product: The product object to remove.

        Pre-condition: The product must exist in the cart.
        """
        with self.cond:
            # Block Logic: Finds the specified cart and removes one instance of the product
            # from its reserved products list.
            # Invariant: Only one instance of the product is removed per call.
            for cos, continut in self.cosuri.items():
                if cos == cart_id:
                    continut['produse_rezervate'].remove(product)

    def place_order(self, cart_id):
        """
        @brief Finalizes an order for a given shopping cart.

        Retrieves all reserved products from the specified cart. This operation
        is thread-safe.

        Parameters:
          - cart_id: The ID of the cart for which the order is being placed.

        Returns:
          - A list of products that were in the cart if the cart ID is valid,
            otherwise None.
        """
        with self.cond:
            # Block Logic: Finds the specified cart and returns its list of
            # reserved products, effectively completing the order.
            # Invariant: The cart's content is returned and the cart might be
            # considered "processed" (though not explicitly cleared or removed here).
            for cos, continut in self.cosuri.items():
                if cos == cart_id:
                    # Note: The products are returned but not explicitly removed from
                    # the producer's inventory in this `place_order` step, which
                    # might lead to over-selling in a real system.
                    return continut['produse_rezervate']
            return None


from threading import Thread
import time

class Producer(Thread):
    """
    @brief Represents a producer (seller) in the marketplace simulation.

    Each Producer thread continuously publishes a predefined list of products
    to the `Marketplace` at specified intervals, making them available for consumers.
    """
    

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a Producer thread.

        Parameters:
          - products: A list of products to be published, where each item is
                      a tuple (product_object, quantity, publish_delay).
          - marketplace: A reference to the shared Marketplace instance.
          - republish_wait_time: The time in seconds to wait before retrying a failed
                                 `publish` operation.
          - kwargs: Additional keyword arguments passed to the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.kwargs = kwargs
        self.republish_wait_time = republish_wait_time
        self.marketplace = marketplace
        self.products = products
        self.daemon = True # Set as daemon so it terminates when main program exits.

    def run(self):
        """
        @brief The main execution method for the Producer thread.

        This method continuously registers with the marketplace and publishes products:
        1. Registers itself as a producer with the marketplace to get a unique ID.
        2. Enters an infinite loop to simulate continuous production.
        3. For each product in its defined list, it attempts to publish the specified
           quantity to the marketplace, with delays between each publication.
        4. If publishing fails, it retries after `republish_wait_time`.
        """
        # Register the producer with the marketplace to obtain a unique producer ID.
        producator_id = self.marketplace.register_producer()
        # Enters an infinite loop to continuously publish products.
        while True:
            # Iterate through each product definition to publish.
            for produs in self.products:
                # Block Logic: Publishes a specified quantity of a single product.
                # Invariant: `contor` tracks the number of items successfully published.
                contor = 0
                while contor < produs[1]: # produs[1] is the quantity to publish.
                    # Attempt to publish the product to the marketplace.
                    in_market = self.marketplace.publish(producator_id, produs[0]) # produs[0] is the product object.
                    time.sleep(produs[2]) # produs[2] is the delay after publishing.
                    # If publishing fails, retry after a delay.
                    # Invariant: Loop continues until the product is successfully published.
                    while in_market == False:
                        in_market = self.marketplace.publish(producator_id, produs[0])
                        time.sleep(self.republish_wait_time) # Wait before retrying.
                    contor += 1 # Increment count for successfully published item.


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Base dataclass representing a generic product in the marketplace.

    This class provides common attributes for all products, such as a name and price.
    It is frozen to ensure immutability once created.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Dataclass representing a specific type of product: Tea.

    Inherits from `Product` and adds a `type` attribute to specify the tea variety.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Dataclass representing a specific type of product: Coffee.

    Inherits from `Product` and adds specific attributes for `acidity` and `roast_level`.
    """
    acidity: str
    roast_level: str
