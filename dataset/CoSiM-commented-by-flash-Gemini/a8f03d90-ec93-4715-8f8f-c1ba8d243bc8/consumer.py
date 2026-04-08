"""
@a8f03d90-ec93-4715-8f8f-c1ba8d243bc8/consumer.py
@brief This module defines the `Consumer` thread, `Marketplace` core logic, `Producer` thread, and product dataclasses
       for a simulated e-commerce system. It orchestrates the creation of shopping carts,
       adding/removing products, placing orders, and managing product availability
       with concurrent producers and consumers.
Functional Utility: Simulates an e-commerce marketplace to demonstrate multi-threaded
                    interactions between producers, consumers, and a central marketplace.
Domain: Concurrency, Multi-threading, E-commerce Simulation, Data Structures.
"""


from threading import Thread
import time

QUANTITY = "quantity"
PRODUCT = "product"
TYPE = "type"
ADD = "add"
REMOVE = "remove"

class Consumer(Thread):
    """
    @class Consumer
    @brief Represents a consumer thread that interacts with the marketplace to create carts,
           add/remove products, and place orders.

    Functional Utility: Simulates user behavior in an e-commerce system by attempting to
                        purchase various products, handling product unavailability through retries.
    Domain: Concurrency, E-commerce Simulation.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a new Consumer instance.
        @param carts (list): A list of shopping cart configurations, where each cart is a list of items.
                             Each item is a dictionary specifying product, quantity, and action (add/remove).
        @param marketplace (Marketplace): A reference to the shared marketplace object.
        @param retry_wait_time (float): The time in seconds to wait before retrying an action
                                        if a product is unavailable.
        @param kwargs: Arbitrary keyword arguments passed to the `Thread` constructor.
        Functional Utility: Sets up the consumer with its shopping list, a link to the marketplace,
                            and its retry behavior.
        Attributes:
            carts (list): The list of predefined shopping cart actions for this consumer.
            marketplace (Marketplace): The central marketplace instance.
            retry_wait_time (float): The duration to pause before retrying an action.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time

    def do_action(self, id_element, item):
        """
        @brief Performs an add or remove action on the cart in the marketplace.
        @param id_element (int): The ID of the cart to perform the action on.
        @param item (dict): A dictionary containing the "type" (add/remove) and "product"
                            to act upon.
        Functional Utility: Abstract a single interaction with the marketplace for either
                            adding or removing a product from a specific cart.
        @returns (bool): True if the action was successful, False otherwise.
        """
        if item[TYPE] == ADD:
            val = self.marketplace.add_to_cart(id_element, item[PRODUCT])
        else: # item["type"] == REMOVE
            val = self.marketplace.remove_from_cart(id_element, item[PRODUCT])
        return val

    def run(self):
        """
        @brief The main execution loop for the Consumer thread.
        Functional Utility: Iterates through its assigned `carts` (shopping lists),
                            creates a new cart in the marketplace for each,
                            and attempts to add/remove products. It includes retry logic
                            for actions that initially fail (e.g., due to product unavailability).
                            Finally, it places the order for each cart.
        """
        # Block Logic: This variable is used to store the result of marketplace actions.
        val = False
        
        # Block Logic: Iterates through each predefined cart configuration for this consumer.
        for i in range(0, len(self.carts)):
            
            # Block Logic: Retrieves the current cart configuration (list of items).
            element = self.carts[i]
            
            # Functional Utility: Requests a new shopping cart from the marketplace for the current iteration.
            id_element = self.marketplace.new_cart()

            # Block Logic: Iterates through each item (product, quantity, action) within the current cart configuration.
            for j in range(0, len(element)):
                item = element[j]
                
                # Block Logic: Attempts to perform the specified action (add/remove) for the item's quantity.
                for k in range(0, item[QUANTITY]):
                    val = self.do_action(id_element, item)

                    # Block Logic: Implements retry logic if a marketplace action (add/remove) fails.
                    # Invariant: The consumer will repeatedly attempt the action until it succeeds,
                    #            pausing for `retry_wait_time` between attempts.
                    if not val:
                        # Block Logic: Enters a loop to continuously retry the action.
                        while True:
                            # Functional Utility: Pauses the consumer thread for a specified time before retrying.
                            time.sleep(self.retry_wait_time)
                            val = self.do_action(id_element, item)
                            # Block Logic: If the action succeeds on retry, break out of the retry loop.
                            if val:
                                break
            
            # Functional Utility: Once all items in a cart configuration are processed, the order is placed.
            self.marketplace.place_order(id_element)

import uuid
from threading import Lock, currentThread

class Marketplace:
    """
    @class Marketplace
    @brief Manages products from various producers, handles consumer shopping carts, and processes orders.

    Functional Utility: Serves as the central hub for all product and cart-related operations,
                        ensuring thread-safe interactions between producers and consumers. It
                        maintains inventory (implicitly through producers' queues) and manages
                        the lifecycle of shopping carts.
    Domain: Concurrency, E-commerce Simulation, Resource Management.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes a new Marketplace instance.
        @param queue_size_per_producer (int): The maximum number of products a single producer
                                              can publish to the marketplace at any given time.
        Functional Utility: Sets up the initial state of the marketplace, including data structures
                            for managing carts, tracking the number of active carts, and
                            initializing locks for thread-safe access to shared resources.
        Attributes:
            queue_size_per_producer (int): The capacity limit for products from each producer.
            items_in_cart (dict): A dictionary mapping `cart_id` to a list of products in that cart.
                                  Format: `{cart_id: [producer_id, [product1, product2,...]]}`
            number_of_carts (int): A counter for newly created carts, used to generate unique `cart_id`s.
            lock_carts (Lock): Protects `number_of_carts` and ensures unique `cart_id` generation.
            lock_remove (Lock): Protects shared data during product removal operations.
            lock_print (Lock): Protects the print statement during order placement to prevent interleaved output.
            lock_add (Lock): Protects shared data during product addition operations.
            producer (dict): A dictionary mapping `producer_id` to a list containing
                             the producer's product list and its current queue size.
                             Format: `{producer_id: [ [product1, product2,...], current_queue_size ]}`
        """
        self.queue_size_per_producer = queue_size_per_producer

        
        self.items_in_cart = {}
        
        self.number_of_carts = 0

        
        self.lock_carts = Lock()
        self.lock_remove = Lock()
        self.lock_print = Lock()
        self.lock_add = Lock()

        
        
        self.producer = {}

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace.
        Functional Utility: Assigns a unique ID to a new producer and initializes its product queue.
        @returns (uuid.UUID): A unique identifier for the registered producer.
        Post-conditions: A new entry for the producer is created in `self.producer`.
        """
        id_producer = uuid.uuid4()
        # Block Logic: Initializes the producer's entry with an empty list of products and a queue size of 0.
        element = [[], 0]
        self.producer[id_producer] = element
        return id_producer

    def publish(self, producer_id, product):
        """
        @brief Allows a producer to publish a product to the marketplace.
        @param producer_id (uuid.UUID): The unique identifier of the producer.
        @param product (Product): The product object to be published.
        Functional Utility: Adds a product to the producer's inventory within the marketplace,
                            respecting the `queue_size_per_producer` limit.
        @returns (bool): True if the product was successfully published, False if the producer's queue is full.
        Pre-conditions: `producer_id` must be a registered producer.
        Post-conditions: If successful, `product` is added to the producer's product list, and its queue size is incremented.
        """
        # Block Logic: Checks if the producer's queue has reached its maximum capacity.
        if self.producer[producer_id][1] >= self.queue_size_per_producer:
            return False

        # Block Logic: Increments the count of products in the producer's queue.
        self.producer[producer_id][1] += 1
        # Block Logic: Adds the product to the producer's list of available products.
        self.producer[producer_id][0].append(product)
        return True

    def new_cart(self):
        """
        @brief Creates a new shopping cart and assigns it a unique ID.
        Functional Utility: Provides a fresh cart for a consumer to begin adding products,
                            ensuring thread-safe assignment of cart IDs.
        @returns (int): A unique integer identifier for the new cart.
        Post-conditions: A new cart entry is created in `self.items_in_cart` and `self.number_of_carts` is incremented.
        """
        # Block Logic: Acquires a lock to ensure atomic increment of `number_of_carts` and unique `cart_id` assignment.
        with self.lock_carts:
            self.number_of_carts += 1
            cart_id = self.number_of_carts

        # Block Logic: Initializes the new cart with a placeholder for the producer ID (to be set when adding the first item) and an empty list for products.
        element = ["", []]
        self.items_in_cart[cart_id] = element
        return cart_id

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a specified product to an existing shopping cart.
        @param cart_id (int): The ID of the cart to which the product will be added.
        @param product (Product): The product object to add.
        Functional Utility: Attempts to transfer a product from a producer's inventory to a consumer's cart.
                            Handles product availability and updates internal counts.
        @returns (bool): True if the product was successfully added, False if the product was not found.
        Pre-conditions: `cart_id` must correspond to an existing cart.
        Post-conditions: If successful, the product is removed from the producer's inventory and added to the cart.
        """
        # Block Logic: Acquires a lock to ensure thread-safe modification of producer inventories.
        with self.lock_add:
            id_prod = ""
            # Block Logic: Searches for the producer that currently has the desired product in stock.
            # Invariant: Assumes a product is uniquely associated with one producer at a time or the first found is sufficient.
            id_p = [x for x in self.producer.keys() if product in self.producer[x][0]]
            if len(id_p) == 0:
                return False

            id_prod = id_p[0]
            # Block Logic: Decrements the count of products from the producer's queue since it's being moved to a cart.
            self.producer[id_prod][1] -= 1

        # Block Logic: Removes the product from the producer's actual list of products.
        self.producer[id_prod][0].remove(product)

        # Block Logic: Records the producer ID for the cart (if not already set) and adds the product to the cart.
        # This implicitly assumes all products in a cart come from the same producer.
        self.items_in_cart[cart_id][0] = id_prod
        self.items_in_cart[cart_id][1].append(product)
        return True

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a specified product from an existing shopping cart and returns it to the producer.
        @param cart_id (int): The ID of the cart from which the product will be removed.
        @param product (Product): The product object to remove.
        Functional Utility: Returns a product from a consumer's cart back to the originating producer's inventory,
                            updating all relevant counts and ensuring thread safety.
        @returns (bool): True if the product was successfully removed. (Always True in current implementation).
        Pre-conditions: `cart_id` must correspond to an existing cart, and `product` must be in that cart.
        Post-conditions: The product is removed from the cart and added back to the producer's inventory.
        """
        # Block Logic: Adds the product back to the producer's list of available products.
        # The producer ID is retrieved from the cart's stored producer information.
        self.producer[self.items_in_cart[cart_id][0]][0].append(product)

        # Block Logic: Acquires a lock to ensure thread-safe increment of the producer's queue size.
        with self.lock_remove:
            self.producer[self.items_in_cart[cart_id][0]][1] += 1

        # Block Logic: Removes the product from the cart's list of items.
        self.items_in_cart[cart_id][1].remove(product)
        return True
        
    def place_order(self, cart_id):
        """
        @brief Finalizes a shopping cart by "placing an order" and prints the purchased items.
        @param cart_id (int): The ID of the cart to place the order for.
        Functional Utility: Removes the cart from the marketplace's active carts, simulating a checkout process,
                            and logs the purchased items in a thread-safe manner.
        @returns (list): A list containing the producer ID and the list of products that were in the cart.
        Post-conditions: The cart identified by `cart_id` is removed from `self.items_in_cart`.
                         The purchased items are printed to the console.
        """
        # Block Logic: Atomically removes the cart from `items_in_cart`. If cart_id doesn't exist, returns None.
        my_prods = self.items_in_cart.pop(cart_id, None)

        # Block Logic: Iterates through the products in the placed order and prints them.
        for elem in my_prods[1]:
            # Functional Utility: Acquires a lock to prevent interleaved output when multiple threads print.
            with self.lock_print:
                print(currentThread().getName() + " bought " + str(elem))

        return my_prods

class Producer(Thread):
    """
    @class Producer
    @brief Represents a producer thread that continuously publishes products to the marketplace.

    Functional Utility: Simulates a supplier in an e-commerce system, making products available
                        for consumers to purchase. It includes logic for waiting before republishing
                        and retrying if the marketplace queue is full.
    Domain: Concurrency, E-commerce Simulation, Resource Management.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a new Producer instance.
        @param products (list): A list of products this producer will offer. Each product entry
                                is a list containing `[product_object, quantity, time_to_wait_after_publish]`.
        @param marketplace (Marketplace): A reference to the shared marketplace object.
        @param republish_wait_time (float): The time in seconds to wait before retrying to publish
                                            a product if the marketplace queue is full.
        @param kwargs: Arbitrary keyword arguments passed to the `Thread` constructor.
        Functional Utility: Sets up the producer with its product list, a link to the marketplace,
                            its republishing behavior, and registers itself with the marketplace.
        Attributes:
            products (list): The list of products to be published by this producer.
            marketplace (Marketplace): The central marketplace instance.
            republish_wait_time (float): The duration to pause before retrying to publish.
            id_ (uuid.UUID): The unique identifier assigned to this producer by the marketplace.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # Functional Utility: Registers this producer with the marketplace to get a unique ID.
        self.id_ = self.marketplace.register_producer()

    def run(self):
        """
        @brief The main execution loop for the Producer thread.
        Functional Utility: Continuously iterates through its list of products, attempting to
                            publish them to the marketplace. It incorporates a `time_wait`
                            between successful publications and a `republish_wait_time`
                            for retries if the marketplace queue is full.
        """
        # Block Logic: Retrieves the producer's predefined list of products.
        prods = self.products

        # Block Logic: Enters an infinite loop to continuously publish products.
        while True:
            # Block Logic: Iterates through each unique product type (including quantity and wait time)
            #              that this producer is configured to offer.
            for i in range(0, len(prods)):

                # Variables:
                # nr_products: The number of units of the current product type to publish.
                # product: The actual product object to be published.
                # time_wait: The time to wait after successfully publishing one unit of the product.
                nr_products = prods[i][1]
                product = prods[i][0]
                time_wait = prods[i][2]

                # Block Logic: Attempts to publish each unit of the current product type.
                for j in range(0, nr_products):
                    val = self.marketplace.publish(self.id_, product)

                    # Block Logic: If publishing is successful, waits for `time_wait` before attempting the next unit.
                    if val:
                        time.sleep(time_wait)
                    # Block Logic: If publishing fails (e.g., marketplace queue is full), enters a retry loop.
                    else:
                        # Invariant: The producer will repeatedly attempt to publish the product
                        #            until it succeeds, pausing for `republish_wait_time` between attempts.
                        while True:
                            # Functional Utility: Pauses the producer thread for a specified time before retrying publication.
                            time.sleep(self.republish_wait_time)
                            val = self.marketplace.publish(self.id_, product)
                            # Block Logic: If the publication succeeds on retry, breaks out of the retry loop.
                            if val:
                                break


from dataclasses import dataclass


@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @dataclass Product
    @brief Base dataclass representing a generic product in the marketplace.

    Functional Utility: Provides a immutable structure to define products with fundamental attributes
                        like name and price, serving as a base for more specific product types.
    Attributes:
        name (str): The name of the product.
        price (int): The price of the product.
    Domain: E-commerce Simulation, Data Modeling.
    """
    name: str
    price: int


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @dataclass Tea
    @brief Represents a specific type of product: Tea.

    Functional Utility: Extends the `Product` dataclass to include a specific attribute
                        for tea, allowing for detailed product categorization.
    Attributes:
        type (str): The type of tea (e.g., "green", "black", "herbal").
    Domain: E-commerce Simulation, Data Modeling.
    """
    type: str


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @dataclass Coffee
    @brief Represents a specific type of product: Coffee.

    Functional Utility: Extends the `Product` dataclass to include specific attributes
                        for coffee, allowing for detailed product categorization.
    Attributes:
        acidity (str): The acidity level of the coffee (e.g., "low", "medium", "high").
        roast_level (str): The roast level of the coffee (e.g., "light", "medium", "dark").
    Domain: E-commerce Simulation, Data Modeling.
    """
    acidity: str
    roast_level: str
