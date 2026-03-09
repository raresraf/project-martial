import time
from threading import Thread


class Consumer(Thread):
    """
    Represents a consumer thread that interacts with the marketplace.

    This class simulates a consumer who adds and removes products from a shopping cart
    and eventually places an order. It operates concurrently with other consumers and producers.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer instance.

        Args:
            carts (list): A list of shopping carts, where each cart is a list of commands
                          (add/remove products).
            marketplace (Marketplace): The shared marketplace object.
            retry_wait_time (float): The time in seconds to wait before retrying to add a
                                     product if it's not available.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.carts = carts
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        The main execution logic of the consumer thread.

        It processes each command in its assigned carts, interacting with the
        marketplace to add or remove products, and finally places the order.
        """
        # Each consumer gets a unique cart ID from the marketplace.
        id_cart = self.marketplace.new_cart()

        # Iterate through the list of carts assigned to this consumer.
        for cart in self.carts:



            # Process each command (add/remove) in the current cart.
            for command in cart:
                
                # If the command is to add a product.
                if command["type"] == "add":
                    # Attempt to add the specified quantity of the product to the cart.
                    for i in range(command["quantity"]):
                        available = self.marketplace.add_to_cart(id_cart, command["product"]) 
                        # Pre-condition: The product may not be immediately available.
                        # Invariant: Keep retrying until the product is successfully added.
                        while not available:
                            time.sleep(self.retry_wait_time)
                            available = self.marketplace.add_to_cart(id_cart, command["product"])

                # If the command is to remove a product.
                else:
                    # Remove the specified quantity of the product from the cart.
                    for i in range(command["quantity"]):
                        
                        self.marketplace.remove_from_cart(id_cart, command["product"])

        
        # After processing all commands, place the final order.
        self.marketplace.place_order(id_cart)

from threading import Lock


class Marketplace:
    """
    A thread-safe marketplace for producers to publish and consumers to buy products.

    This class manages the inventory of products from multiple producers and the shopping
    carts of multiple consumers, using locks to ensure safe concurrent access.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products a single
                                           producer can have in the marketplace at one time.
        """
        self.queue_size = queue_size_per_producer
        self.producer_id = 0 
        self.cart_id = 0 
        self.market = [[]] 
        self.cart = [[]] 
        # Locks for various critical sections to ensure thread safety.
        self.lock_add = Lock() 
        self.lock_remove = Lock()
        self.lock_cart = Lock()
        self.lock_producer = Lock()
        self.lock_print = Lock()

    def register_producer(self):
        """
        Registers a new producer, providing them with a unique ID.

        Returns:
            int: The unique ID for the newly registered producer.
        """

        

        # Atomically increment producer ID and expand the market list.
        self.lock_producer.acquire()
        self.producer_id += 1
        self.market.append([]) 
        self.lock_producer.release()
        return self.producer_id

    def publish(self, producer_id, product):
        """
        Allows a producer to publish a product to the marketplace.

        Args:
            producer_id (int): The ID of the producer.
            product: The product to be published.

        Returns:
            bool: True if the product was published successfully, False if the
                  producer's queue is full.
        """

        

        # Pre-condition: Check if the producer's product queue is not full.
        if len(self.market[producer_id - 1]) >= self.queue_size:
            return False
        self.market[producer_id - 1].append(product)
        return True

    def new_cart(self):
        """
        Creates a new, empty shopping cart for a consumer.

        Returns:
            int: The unique ID for the new cart.
        """

        

        # Atomically increment cart ID and expand the cart list.
        self.lock_cart.acquire()
        self.cart_id += 1
        self.cart.append([])
        self.lock_cart.release()
        return self.cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a consumer's shopping cart.

        Searches the entire market for the product. If found, it's moved from the
        market to the specified cart.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product: The product to add.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        self.lock_add.acquire()
        # Invariant: Search all producer queues for the requested product.
        for i in range(len(self.market)):
            for j in range(len(self.market[i])):
                if self.market[i][j] == product: 
                    # If found, move the product from the market to the consumer's cart.
                    self.cart[cart_id - 1].append((product, i))
                    self.market[i].remove(product)
                    self.lock_add.release()
                    return True

        self.lock_add.release()
        return False 


    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a consumer's shopping cart.

        The product is returned to its original producer's queue in the market.

        Args:
            cart_id (int): The ID of the cart to remove from.
            product: The product to remove.
        """
        self.lock_remove.acquire()
        # Invariant: Find the specified product in the cart.
        for i in range(len(self.cart[cart_id - 1])):
            if self.cart[cart_id - 1][i][0] == product: 
                # Return the product to the original producer's market queue.
                self.market[self.cart[cart_id - 1][i][1]].append(product)
                prod_id = self.cart[cart_id - 1][i][1]
                self.cart[cart_id - 1].remove((product, prod_id))
                break

        self.lock_remove.release()

    def place_order(self, cart_id):
        """
        Finalizes the order by printing the items in the cart.
        This simulates the checkout process.

        Args:
            cart_id (int): The ID of the cart to be ordered.
        """
        self.lock_print.acquire()
        # Invariant: Print every item currently in the consumer's cart.
        for i in range(len(self.cart[cart_id - 1])):
            print("cons"+ str(cart_id) + " bought " + str(self.cart[cart_id - 1][i][0]))
        self.lock_print.release()


from threading import Thread
import time

class Producer(Thread):
    """
    Represents a producer thread that publishes products to the marketplace.

    This class simulates a producer who continuously creates products and adds them
    to the marketplace for consumers to purchase.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer instance.

        Args:
            products (list): A list of products that the producer can create. Each product
                             is a tuple containing the product itself, its quantity, and the
                             time to wait after producing it.
            marketplace (Marketplace): The shared marketplace object.
            republish_wait_time (float): The time in seconds to wait before retrying to
                                         publish a product if the queue is full.
            **kwargs: Additional keyword arguments for the Thread constructor.
        """
        Thread.__init__(self, **kwargs)
        self.marketplace = marketplace
        self.product_list = products


        self.wait_time = republish_wait_time
        self.kwargs = kwargs

    def run(self):
        """
        The main execution logic of the producer thread.

        It continuously produces and publishes its products to the marketplace.
        """
        # Register the producer with the marketplace to get a unique ID.
        id_producer = self.marketplace.register_producer()
        # Loop indefinitely to continuously produce.
        while 1:
            for prod in self.product_list:
                
                # Produce the specified quantity of the current product.
                for i in range(prod[1]):
                    
                    can_add = self.marketplace.publish(id_producer, prod[0])
                    # Pre-condition: The marketplace queue for this producer might be full.
                    # Invariant: If publishing fails, wait and retry until successful.
                    if not can_add:
                        while not can_add:
                            time.sleep(self.wait_time)
                            can_add = self.marketplace.publish(id_producer, prod[0])
                    # After successful publishing, wait for the specified production time.
                    else:
                        time.sleep(prod[2])