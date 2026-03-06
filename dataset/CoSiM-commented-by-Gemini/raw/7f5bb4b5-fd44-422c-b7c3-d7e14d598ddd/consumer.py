"""
This module simulates a marketplace with producers and consumers using a multi-threaded
approach. It models the classic producer-consumer problem where Producers add products
to a shared Marketplace, and Consumers take them.

The simulation consists of four main components:
- Product: Dataclasses representing items for sale.
- Marketplace: The central, thread-safe shared resource that manages inventory
  and shopping carts.
- Producer: A thread that publishes products to the marketplace.
- Consumer: A thread that simulates a customer adding/removing items to a cart
  and placing an order.
"""

from threading import Thread, Lock
from time import sleep
from dataclasses import dataclass

# --- Product Definitions ---

@dataclass(init=True, repr=True, order=False, frozen=True)
class Product:
    """A base dataclass for an immutable product with a name and price."""
    name: str
    price: int

@dataclass(init=True, repr=True, order=False, frozen=True)
class Tea(Product):
    """An immutable Tea product, inheriting from Product and adding a 'type' attribute."""
    type: str

@dataclass(init=True, repr=True, order=False, frozen=True)
class Coffee(Product):
    """An immutable Coffee product, adding acidity and roast level attributes."""
    acidity: str
    roast_level: str


# --- Marketplace Simulation ---

class Marketplace:
    """
    The central marketplace, acting as the shared, thread-safe buffer.

    It manages multiple producer queues, consumer shopping carts, and uses
    fine-grained locking to handle concurrent access from multiple producer
    and consumer threads.
    """
    def __init__(self, queue_size_per_producer):
        """
        Initializes the marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of products each
                                           producer can have in their queue at one time.
        """
        self.last_producer_id = -1
        self.last_cart_id = -1
        # A list of lists, where each inner list is a product queue for one producer.
        self.prod_queue = []
        # A list of lists, where each inner list represents a consumer's shopping cart.
        self.all_carts = []
        # Tracks which producer an item in a cart came from, for 'remove_from_cart'.
        self.producerAndProduct = []
        
        # Locks for various critical sections to allow concurrent operations.
        self.addToCart_lock = Lock()
        self.removeFromCart_lock = Lock()
        self.lastProdId_lock = Lock()
        self.publish_lock = Lock()
        self.new_cart_lock = Lock()

        # A counter for active consumers, used to signal termination to producers.
        self.nr_of_consumers = -1
        self.queue_size_per_producer = queue_size_per_producer

    def register_producer(self):
        """
        Registers a new producer, giving it a unique ID and an inventory queue.

        Returns:
            int: The unique ID assigned to the new producer.
        """
        with self.lastProdId_lock:
            self.last_producer_id += 1
            self.prod_queue.append([])
            return self.last_producer_id

    def publish(self, producer_id, product):
        """
        Allows a producer to add a product to their inventory queue.

        Args:
            producer_id (int): The ID of the producer.
            product (Product): The product to be published.

        Returns:
            bool: True if the product was added successfully, False if the
                  producer's queue was full.
        """
        with self.publish_lock:
            if len(self.prod_queue[producer_id]) < self.queue_size_per_producer:
                self.prod_queue[producer_id].append(product)
                return True
            else:
                return False

    def new_cart(self):
        """
        Creates a new, empty shopping cart for a consumer.

        Returns:
            int: The unique ID assigned to the new cart.
        """
        with self.new_cart_lock:
            self.last_cart_id += 1
            self.all_carts.append([])
            return self.last_cart_id

    def add_to_cart(self, cart_id, product):
        """
        Adds a product to a consumer's cart by searching all producer inventories.

        This method iterates through all producer queues to find the requested product.
        If found, it moves the product to the consumer's cart.

        Args:
            cart_id (int): The ID of the consumer's cart.
            product (Product): The product to add.

        Returns:
            bool: True if the product was found and added, False otherwise.
        """
        with self.addToCart_lock:
            # Iterates through every producer's queue to find the product.
            # This is a potential performance bottleneck as it's a global search
            # within a single lock.
            for i in range(len(self.prod_queue)):
                if product in self.prod_queue[i]:
                    self.all_carts[cart_id].append(product)
                    # Remember the original producer for potential returns.
                    self.producerAndProduct.append((i, product))
                    self.prod_queue[i].remove(product)
                    return True
        return False

    def remove_from_cart(self, cart_id, product):
        """
        Removes a product from a consumer's cart and returns it to the
        original producer's inventory.
        """
        with self.removeFromCart_lock:
            if product in self.all_carts[cart_id]:
                self.all_carts[cart_id].remove(product)
                # Find the original producer and return the product to their queue.
                for j in range(len(self.producerAndProduct)):
                    (index, searchProduct) = self.producerAndProduct[j]
                    if searchProduct == product:
                        self.prod_queue[index].append(product)
                        self.producerAndProduct.pop(j)
                        break
                # Assuming one removal at a time.
                return True
        return False

    def place_order(self, cart_id):
        """
        Finalizes an order by returning the contents of the cart.

        Returns:
            list: A list of products that were in the cart.
        """
        return self.all_carts[cart_id]

# --- Producer and Consumer Threads ---

class Producer(Thread):
    """A thread that simulates a producer publishing products to the marketplace."""
    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """Initializes the Producer thread."""
        Thread.__init__(self, daemon=kwargs.get("daemon", False))
        self.products = products
        self.marketplace = marketplace
        self.republish_wait_time = republish_wait_time
        self.name = kwargs.get('name')

    def run(self):
        """
        The main execution loop for the producer.

        The producer continuously attempts to publish its products. If the
        marketplace queue for a product is full, it waits and retries.
        The thread terminates when it observes that there are no more active consumers.
        """
        producer_id = self.marketplace.register_producer()

        # Invariant: Loop continues as long as there are active consumers.
        while True:
            for prod_info in self.products:
                product, quantity, production_time = prod_info
                items_published = 0
                while items_published < quantity:
                    # Attempt to publish one item.
                    if self.marketplace.publish(producer_id, product):
                        sleep(production_time)
                        items_published += 1
                    else:
                        # If the queue is full, wait before retrying.
                        sleep(self.republish_wait_time)
            
            # Check for termination condition.
            if self.marketplace.nr_of_consumers == 0:
                break

class Consumer(Thread):
    """
    A thread that simulates a consumer performing a series of shopping tasks.
    """
    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """Initializes the Consumer thread."""
        Thread.__init__(self)
        self.carts = carts # A list of shopping lists (list of tasks).
        self.marketplace = marketplace
        self.retry_wait_time = retry_wait_time
        self.name = kwargs.get('name')

    def run(self):
        """
        The main execution loop for the consumer.

        It processes each shopping list ('cart'), executing 'add' or 'remove'
        tasks. If a task fails (e.g., product not available), it waits and retries.
        """
        # Register self as an active consumer.
        if self.marketplace.nr_of_consumers == -1:
            self.marketplace.nr_of_consumers = 1
        else:
            self.marketplace.nr_of_consumers += 1

        # Process each shopping list.
        for cart_tasks in self.carts:
            new_cart_id = self.marketplace.new_cart()
            for task in cart_tasks:
                i = 0
                while i < task.get('quantity'):
                    # Perform the add or remove action.
                    if task.get('type') == "add":
                        success = self.marketplace.add_to_cart(new_cart_id, task.get('product'))
                    elif task.get('type') == "remove":
                        success = self.marketplace.remove_from_cart(new_cart_id, task.get('product'))
                    
                    if not success:
                        # If the action failed, wait before retrying.
                        sleep(self.retry_wait_time)
                    else:
                        i += 1

            # After all tasks for a cart are done, place the order.
            for prod in self.marketplace.place_order(new_cart_id):
                print(f"{self.name} bought {prod}")
        
        # De-register self as an active consumer.
        self.marketplace.nr_of_consumers -= 1
