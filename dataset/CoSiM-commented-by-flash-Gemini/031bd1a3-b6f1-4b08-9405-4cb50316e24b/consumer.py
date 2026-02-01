

# @file 031bd1a3-b6f1-4b08-9405-4cb50316e24b/consumer.py
# @brief Implements a multithreaded marketplace simulation with producer and consumer entities.
#
# This module defines the core components for a concurrent marketplace:
# - Consumer: Represents a buyer thread that interacts with the marketplace to add, remove, and purchase items.
# - Marketplace: Manages product availability, producer registration, cart operations, and order placement, ensuring thread-safe access to shared resources.
# - Producer: Represents a seller thread that continuously publishes products to the marketplace.
# - Product data classes: Define various types of products with their attributes.

import time
from threading import Thread
from threading import Lock # Added import for Lock in Marketplace class

class Consumer(Thread):
    """
    @brief Represents a consumer thread in the marketplace simulation.
    
    A Consumer creates carts, adds and removes products from them, and
    ultimately places orders. It handles retries for marketplace operations
    that might initially fail due to product unavailability or other concurrency issues.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a Consumer thread.

        @param carts: A list of shopping cart specifications, where each cart is a list of item dictionaries.
                      Each item dictionary specifies 'type' (add/remove), 'product', and 'quantity'.
        @param marketplace: The shared Marketplace instance with which the consumer interacts.
        @param retry_wait_time: The time in seconds to wait before retrying a failed marketplace operation.
        @param kwargs: Additional keyword arguments passed to the Thread constructor, e.g., 'name'.
        @pre 'marketplace' is a valid instance of the Marketplace class.
        @post The consumer thread is initialized with its assigned carts, a reference to the marketplace, and retry wait time.
        """
        # Functional Utility: Initializes the base Thread class.
        Thread.__init__(self, **kwargs)
        # Functional Utility: Stores the unique name of this consumer thread.
        self.name = kwargs["name"]
        # Functional Utility: Stores the list of shopping carts this consumer will process.
        self.carts = carts
        # Functional Utility: Stores a reference to the shared marketplace instance.
        self.marketplace = marketplace
        # Functional Utility: Stores the duration to wait before retrying marketplace operations.
        self.wait_time = retry_wait_time
        

    def run(self):
        """
        @brief Executes the consumer's shopping behavior.

        For each specified cart, the consumer creates a new cart in the marketplace,
        then iteratively adds and removes products according to the cart's specification.
        Operations are retried if they fail, with a delay. Finally, the order is placed,
        and purchased items are printed.
        @pre The marketplace is active and accessible.
        @post All assigned carts are processed, and orders are placed.
        """
        # Block Logic: Iterates through each shopping cart specification provided to the consumer.
        # Invariant: Each iteration processes one complete shopping cart.
        for cart in self.carts:
            # Functional Utility: Requests the marketplace to create a new, empty shopping cart and retrieves its unique ID.
            cart_id = self.marketplace.new_cart()

            # Block Logic: Processes each item within the current shopping cart specification.
            # Invariant: Each iteration handles an 'add' or 'remove' operation for a specific product.
            for item in cart:
                # Block Logic: Handles 'add' operations for products.
                if item["type"] == "add":
                    # Block Logic: Attempts to add the specified quantity of a product to the cart.
                    # Invariant: Continues retrying until the product is successfully added.
                    for _ in range(item["quantity"]):
                        while not self.marketplace.add_to_cart(cart_id, item["product"]):
                            # Functional Utility: Pauses execution for a defined period before retrying.
                            time.sleep(self.wait_time)

                # Block Logic: Handles 'remove' operations for products.
                elif item["type"] == "remove":
                    # Block Logic: Attempts to remove the specified quantity of a product from the cart.
                    # Invariant: Continues retrying until the product is successfully removed.
                    for _ in range(item["quantity"]):
                        while not self.marketplace.remove_from_cart(cart_id, item["product"]):
                            # Functional Utility: Pauses execution for a defined period before retrying.
                            time.sleep(self.wait_time)

            # Functional Utility: Places the final order for the populated cart.
            final_cart = self.marketplace.place_order(cart_id)
            # Block Logic: Prints the details of each item successfully bought in the order.
            # Invariant: Each iteration prints one item from the final cart.
            for item in final_cart:
                if item is not None: # Inline: Ensures only valid (non-None) items are reported.
                    print(self.name + " bought " + str(item))
    

class Marketplace:
    """
    @brief Manages products and shopping carts in a multi-threaded environment.
    
    This class provides thread-safe operations for producers to publish products
    and for consumers to create carts, add/remove products, and place orders.
    It handles inventory management, producer registration, and ensures data
    consistency using various locks and data structures.
    """
    
    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace.

        @param queue_size_per_producer: The maximum number of items each producer can have in the marketplace's inventory.
        @post The marketplace is initialized with empty product inventories, producer/cart counters, and necessary locks.
        """
        # Functional Utility: Maximum number of items each producer can offer.
        self.queue_size = queue_size_per_producer
        # Functional Utility: Counter for registered producers.
        self.producer_count = 0
        # Functional Utility: Counter for created shopping carts.
        self.cart_count = 0
        # Functional Utility: Dictionary mapping cart IDs to lists of products in each cart.
        self.carts = {}
        # Functional Utility: Dictionary mapping cart IDs to lists of producer IDs that supplied items in the cart.
        self.cart_suppliers = {}
        # Functional Utility: Lock to ensure thread-safe access to global producer registration.
        self.producer_lock_univ = Lock()
        # Functional Utility: List of locks, one for each producer, to protect their individual queues.
        self.producer_lock = []
        # Functional Utility: Lock to ensure thread-safe access to cart creation and management.
        self.cart_lock = Lock()
        # Functional Utility: List to track available capacity for each producer's queue.
        self.producer_capacity = []
        # Functional Utility: Dictionary of locks, one for each product, to ensure thread-safe inventory updates.
        self.item_lock = {}
        # Functional Utility: Dictionary mapping product names to their current available quantities in the marketplace.
        self.product_availability = {}
        # Functional Utility: Dictionary mapping product names to a list of producer IDs that currently supply that product.
        self.product_suppliers = {}
        

    def register_producer(self):
        """
        @brief Registers a new producer with the marketplace.

        Assigns a unique ID to the producer, initializes its capacity, and creates a dedicated lock.
        @return: The unique integer ID assigned to the new producer.
        @post A new producer is registered, its capacity is initialized, and a lock is created for it.
        """
        # Block Logic: Ensures exclusive access during producer registration to prevent race conditions.
        with self.producer_lock_univ:
            # Functional Utility: Initializes the capacity for the new producer based on the marketplace's configured queue size.
            self.producer_capacity.append(self.queue_size)
            # Functional Utility: Assigns the current producer count as the new producer's ID.
            retval = self.producer_count
            # Functional Utility: Increments the global producer count.
            self.producer_count += 1
            # Functional Utility: Adds a new lock specifically for this producer to manage its publishing operations.
            self.producer_lock += [Lock()]
            return retval

    def publish(self, producer_id, product):
        """
        @brief Publishes a product to the marketplace by a specific producer.

        The product is added to the marketplace's inventory if the producer has available capacity.
        @param producer_id: The ID of the producer publishing the product.
        @param product: The product being published (assumed to be a tuple or list where product[0] is the name).
        @return: True if the product was successfully published, False otherwise (e.g., if producer capacity is full).
        @pre 'producer_id' is a valid registered producer ID.
        @post If successful, the product's availability increases, and the producer's capacity decreases.
        """
        # Block Logic: Acquires the specific lock for the given producer to ensure thread-safe publishing.
        with self.producer_lock[producer_id]:
            # Block Logic: Checks if the producer has remaining capacity to publish more products.
            if self.producer_capacity[producer_id] > 0:
                # Functional Utility: Retrieves or initializes the current availability of the product.
                amount = self.product_availability.setdefault(product[0], 0)
                # Functional Utility: Retrieves or initializes the list of producers supplying this product.
                producers = self.product_suppliers.setdefault(product[0], [])
                # Functional Utility: Adds the current producer to the list of suppliers for this product.
                self.product_suppliers.update({product[0]: producers + [producer_id]})

                # Functional Utility: Increments the overall availability count for the product.
                self.product_availability.update({product[0]: 1 + amount})
                # Functional Utility: Decrements the producer's available capacity.
                self.producer_capacity[producer_id] -= 1
                return True

        return False

    def new_cart(self):
        """
        @brief Creates a new, empty shopping cart in the marketplace.

        @return: The unique integer ID of the newly created cart.
        @post A new cart is created and registered, and its ID is returned.
        """
        # Block Logic: Acquires a global cart lock to ensure thread-safe creation of new carts.
        with self.cart_lock:
            # Functional Utility: Assigns the current cart count as the new cart's ID.
            retval = self.cart_count
            # Functional Utility: Initializes an empty list for products in the new cart.
            self.carts.setdefault(self.cart_count, [])
            # Functional Utility: Initializes an empty list for tracking suppliers of products in the new cart.
            self.cart_suppliers.setdefault(self.cart_count, [])
            # Functional Utility: Increments the global cart count.
            self.cart_count += 1
            return retval

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a specified product to a given shopping cart.

        Retrieves the product from the marketplace's general inventory and assigns it to the cart.
        Handles product locks to ensure atomic inventory updates.
        @param cart_id: The ID of the cart to which the product should be added.
        @param product: The product to add to the cart.
        @return: True if the product was successfully added, False if the product is not available.
        @pre 'cart_id' is a valid, existing cart ID.
        @post If successful, product availability decreases, producer capacity increases, and the product is added to the cart.
        """
        # Functional Utility: Retrieves or creates a product-specific lock to ensure atomic operations on its inventory.
        lock = self.item_lock.setdefault(product, Lock())

        # Block Logic: Acquires the product-specific lock for thread-safe inventory management.
        with lock:
            # Functional Utility: Retrieves or initializes the current availability of the product.
            amount = self.product_availability.setdefault(product, 0)

            # Block Logic: Checks if the product is currently available in the marketplace.
            if amount == 0:
                return False

            # Functional Utility: Retrieves the list of producers currently supplying this product.
            producers = self.product_suppliers.get(product)
            # Block Logic: If there are suppliers, update the producer's capacity and track the supplier for this cart item.
            if producers is not None:
                # Functional Utility: Increments the capacity of the producer who supplied this specific product instance.
                self.producer_capacity[producers[0]] += 1
                # Functional Utility: Records the producer ID that supplied this item to the cart.
                self.cart_suppliers[cart_id].append(producers[0])
                # Functional Utility: Removes the producer from the front of the queue, as this product instance is now in a cart.
                producers.pop(0)


            # Functional Utility: Decrements the overall availability count for the product.
            self.product_availability.update({product: amount - 1})
            # Functional Utility: Adds the product to the consumer's cart.
            self.carts[cart_id].append(product)
            return True

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a specified product from a given shopping cart.

        Returns the product to the marketplace's general inventory. Handles product locks
        to ensure atomic inventory updates.
        @param cart_id: The ID of the cart from which the product should be removed.
        @param product: The product to remove from the cart.
        @return: True if the product was successfully removed, False otherwise (e.g., product not in cart).
        @pre 'cart_id' is a valid, existing cart ID, and 'product' is present in that cart.
        @post If successful, product availability increases, producer capacity decreases, and the product is removed from the cart.
        """
        # Functional Utility: Retrieves or creates a product-specific lock to ensure atomic operations on its inventory.
        lock = self.item_lock.setdefault(product, Lock())

        # Block Logic: Acquires the product-specific lock for thread-safe inventory management.
        with lock:
            # Functional Utility: Retrieves or initializes the current availability of the product.
            amount = self.product_availability.setdefault(product, 0)
            # Functional Utility: Retrieves or initializes the list of producers supplying this product.
            producers = self.product_suppliers.setdefault(product, [])

            # Functional Utility: Finds the index of the product within the cart.
            product_idx = self.carts[cart_id].index(product)
            # Functional Utility: Identifies the producer who supplied this specific item to the cart.
            producer_id = self.cart_suppliers[cart_id][product_idx]
            # Block Logic: Acquires the lock for the specific producer to update their capacity.
            with self.producer_lock[producer_id]:
                # Functional Utility: Returns the producer to the list of suppliers for this product.
                self.product_suppliers.update({product: producers + [producer_id]})
                # Functional Utility: Decrements the capacity of the producer, as the product is no longer "in transit".
                self.producer_capacity[producer_id] -= 1
                # Functional Utility: Increments the overall availability count for the product.
                self.product_availability.update({product: amount + 1})
                # Functional Utility: Marks the supplier for this specific item as None in the cart's supplier list.
                self.cart_suppliers[cart_id][product_idx] = None
                # Functional Utility: Marks the product as None in the cart, effectively removing it.
                self.carts[cart_id][product_idx] = None
                return True

        return False # Inline: This return statement would only be reached if the product lock acquisition fails, which shouldn't happen with setdefault.


    def place_order(self, cart_id):
        """
        @brief Places an order for the specified shopping cart.

        This effectively finalizes the cart and returns its contents.
        @param cart_id: The ID of the cart to place an order for.
        @return: A list of products contained in the specified cart.
        @pre 'cart_id' is a valid, existing cart ID.
        @post The content of the cart is returned.
        """
        # Functional Utility: Returns the list of products associated with the given cart ID.
        return self.carts[cart_id]


class Producer(Thread):
    """
    @brief Represents a producer thread in the marketplace simulation.
    
    A Producer continuously registers itself with the marketplace and publishes
    a predefined set of products, with specified quantities and republish delays.
    It handles retries for publishing operations that might initially fail.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a Producer thread.

        @param products: A list of product specifications, where each entry is a tuple:
                         (product_object, quantity_to_publish, publish_interval_seconds).
        @param marketplace: The shared Marketplace instance with which the producer interacts.
        @param republish_wait_time: The time in seconds to wait before retrying a failed publish operation.
        @param kwargs: Additional keyword arguments passed to the Thread constructor, e.g., 'name'.
        @pre 'marketplace' is a valid instance of the Marketplace class.
        @post The producer thread is initialized with its assigned products, a reference to the marketplace, and republish wait time.
        """
        # Functional Utility: Initializes the base Thread class.
        Thread.__init__(self, **kwargs)
        # Functional Utility: Stores the list of products this producer will publish.
        self.products = products
        # Functional Utility: Stores a reference to the shared marketplace instance.
        self.marketplace = marketplace
        # Functional Utility: Stores the duration to wait before retrying publishing operations.
        self.wait_time = republish_wait_time
        

    def run(self):
        """
        @brief Executes the producer's product publishing behavior.

        The producer registers itself with the marketplace, then continuously
        attempts to publish its products. Publishing operations are retried if they fail,
        with a delay. A delay is also introduced between publishing different products.
        @pre The marketplace is active and accessible.
        @post The producer continuously attempts to publish its products to the marketplace.
        """
        # Functional Utility: Registers this producer with the marketplace and obtains its unique ID.
        this_id = self.marketplace.register_producer()

        # Block Logic: Main loop for the producer, continuously publishing products.
        # Invariant: Products are published repeatedly. This loop is designed to run indefinitely.
        while True:
            # Block Logic: Iterates through each type of product the producer is responsible for.
            # Invariant: Each iteration processes one distinct product type.
            for item in self.products:
                # Block Logic: Attempts to publish the specified quantity of the current product.
                # Invariant: Each iteration publishes one instance of the product.
                for _ in range(item[1]): # Inline: item[1] represents the quantity to publish.
                    # Block Logic: Continuously retries publishing the product until successful.
                    while not self.marketplace.publish(this_id, item):
                        # Functional Utility: Pauses execution for a defined period before retrying.
                        time.sleep(self.wait_time)

                    # Functional Utility: Pauses execution for a defined period after successful publication (republish delay).
                    time.sleep(item[2]) # Inline: item[2] represents the delay before publishing the next item.
        

from dataclasses import dataclass

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """
    @brief Base data class for all products in the marketplace.

    Provides common attributes like name and price, and is designed for immutability.
    """
    name: str  # Functional Utility: The name of the product.
    price: int # Functional Utility: The price of the product.


@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """
    @brief Data class representing a Tea product.

    Extends the base Product class with a specific 'type' attribute for tea.
    """
    type: str # Functional Utility: The type of tea (e.g., "Green", "Black").


@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """
    @brief Data class representing a Coffee product.

    Extends the base Product class with specific attributes for coffee,
    such as 'acidity' and 'roast_level'.
    """
    acidity: str      # Functional Utility: Describes the acidity level of the coffee.
    roast_level: str  # Functional Utility: Indicates the roast level of the coffee.

