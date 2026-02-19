"""
@file consumer.py
@brief A simulation of a producer-consumer model based on an online marketplace.
@details This module defines three classes: Marketplace, Producer, and Consumer, which interact
in a multi-threaded environment to simulate the process of producers publishing products
and consumers buying them.

Note: This file contains the definitions for Consumer, Marketplace, and Producer classes,
which in a larger application would typically be separated into their own modules.
"""

from threading import Thread, Lock, currentThread
from time import sleep


class Consumer(Thread):
    """
    @brief Represents a consumer thread that interacts with the marketplace.
    @details A consumer processes a predefined list of shopping carts, where each cart contains
    a series of 'add' and 'remove' commands. It executes these commands and then places an order.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        @brief Initializes a Consumer thread.
        @param carts A list of carts, where each cart is a list of product commands.
        @param marketplace The shared Marketplace object.
        @param retry_wait_time The time to wait before retrying to add a product if it's not available.
        @param kwargs Additional arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.cart_id = -1

    def run(self):
        """
        @brief The main execution loop for the consumer thread.
        @details Iterates through its assigned carts, executes the commands for each, and places the order.
        """
        # Process each shopping cart assigned to this consumer.
        for cart in self.carts:
            # Get a new, unique cart ID from the marketplace.
            self.cart_id = self.marketplace.new_cart()
            # Execute all commands (add/remove) in the current cart.
            for command in cart:
                # Block Logic: Handle 'add' commands.
                if command["type"] == "add":
                    i = 0
                    while i < command["quantity"]:
                        # Attempt to add the product to the cart.
                        result = self.marketplace.add_to_cart(self.cart_id, command["product"])
                        if result:
                            i = i + 1
                        # If adding fails (product not available), wait and retry.
                        else:
                            sleep(self.retry_wait_time)
                
                # Block Logic: Handle 'remove' commands.
                elif command["type"] == "remove":
                    i = 0
                    while i < command["quantity"]:
                        self.marketplace.remove_from_cart(self.cart_id, command["product"])
                        i = i + 1
            
            # After all commands are processed, finalize the purchase.
            self.marketplace.place_order(self.cart_id)


class Marketplace:
    """
    @brief The central marketplace that manages producers, consumers, and products.
    @details This class acts as the shared resource in the producer-consumer model. It handles
    product inventory, shopping carts, and ensures thread-safe operations through locks.
    """

    def __init__(self, queue_size_per_producer):
        """
        @brief Initializes the Marketplace.
        @param queue_size_per_producer The maximum number of products a single producer can have in the marketplace at one time.
        """
        self.queue_size_per_producer = queue_size_per_producer
        self.id_producer = 0
        # A list of lists, where each inner list is the inventory of a producer.
        self.queue_list = [[]]
        # A list of lists, where each inner list represents a consumer's shopping cart.
        self.cart_list = [[]]
        # Lock to ensure atomic printing when placing an order.
        self.place_order_lock = Lock()
        # Lock to protect the registration of new producers.
        self.register_producer_lock = Lock()
        # Lock to protect shared cart and product queue operations.
        self.cart_lock = Lock()

    def register_producer(self):
        """
        @brief Registers a new producer, providing them with a unique ID.
        @return The newly assigned producer ID.
        """
        self.register_producer_lock.acquire()
        self.id_producer += 1
        self.queue_list.append([])
        self.register_producer_lock.release()
        return self.id_producer

    def publish(self, producer_id, product):
        """
        @brief Allows a producer to publish a product to the marketplace.
        @return True if the product was published successfully, False if the producer's queue is full.
        """
        # Pre-condition: Check if the producer's inventory queue is full.
        if len(self.queue_list[int(producer_id)]) >= self.queue_size_per_producer:
            return False

        self.queue_list[int(producer_id)].append(product)
        return True

    def new_cart(self):
        """
        @brief Creates a new, empty shopping cart for a consumer.
        @return The ID of the newly created cart.
        """
        self.cart_list.append([])
        return len(self.cart_list) - 1

    def add_to_cart(self, cart_id, product):
        """
        @brief Adds a product to a consumer's shopping cart by taking it from a producer's inventory.
        @return True if the product was found and added, False otherwise.
        """
        # This lock protects both the product queues and the cart list from concurrent modification.
        self.cart_lock.acquire()
        i = 0
        # Invariant: Iterates through all producers' inventories to find the requested product.
        for producer_list in self.queue_list:
            for prod in producer_list:
                if prod == product:
                    # Store the producer's index and the product in the cart.
                    self.cart_list[cart_id].append([i, prod])
                    # Remove the product from the producer's inventory.
                    producer_list.remove(prod)
                    self.cart_lock.release()
                    return True
            i = i + 1
        self.cart_lock.release()
        return False

    def remove_from_cart(self, cart_id, product):
        """
        @brief Removes a product from a shopping cart and returns it to the original producer's inventory.
        """
        self.cart_lock.acquire()
        i = 0
        # Invariant: Iterates through the cart to find the product to be removed.
        for tup in self.cart_list[cart_id]:
            if tup[1] == product:
                # Return the product to the producer's inventory queue.
                self.queue_list[tup[0]].append(tup[1])
                self.cart_list[cart_id].pop(i)
                break
            i = i + 1
        self.cart_lock.release()

    def place_order(self, cart_id):
        """
        @brief Finalizes an order, printing the items bought by the consumer.
        @return A list of products that were in the cart.
        """
        products = []
        # Invariant: Iterates through the finalized cart to log each purchased item.
        for tup in self.cart_list[cart_id]:
            products.append(tup[1])
            # Use a lock to ensure the print statement is atomic and not interleaved with other threads.
            self.place_order_lock.acquire()
            print(currentThread().getName() + " bought " + str(tup[1]))
            self.place_order_lock.release()
        return products


class Producer(Thread):
    """
    @brief Represents a producer thread that publishes products to the marketplace.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        @brief Initializes a Producer thread.
        @param products A list of products the producer will create.
        @param marketplace The shared Marketplace object.
        @param republish_wait_time The time to wait after successfully publishing a product.
        @param kwargs Additional arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        # Register with the marketplace to get a unique producer ID.
        self.producer_id = self.marketplace.register_producer()

    def run(self):
        """
        @brief The main execution loop for the producer thread.
        @details Continuously tries to publish its list of products to the marketplace.
        """
        # Infinite loop to continuously produce items.
        while 1:
            for prod in self.products:
                i = 0
                # Publish the specified quantity of the current product.
                while i < prod[1]:
                    result = self.marketplace.publish(str(self.producer_id), prod[0])
                    if result:
                        # If successful, wait before publishing the next item.
                        sleep(self.republish_wait_time)
                        i = i + 1
                    else:
                        # If the marketplace queue is full, wait for a product-specific time and retry.
                        sleep(prod[2])
