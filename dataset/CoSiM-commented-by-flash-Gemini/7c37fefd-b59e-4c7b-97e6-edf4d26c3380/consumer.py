

import time


from threading import Thread
from typing import List

from tema.marketplace import Marketplace
from tema.product import Product


"""
Represents a single operation (add or remove) to be performed on a product within a shopping cart.
This class encapsulates the details of a requested action, including the type of operation,
the product involved, and the quantity for that operation.
"""
class Operation:
    def __init__(self, op_type: str, product: Product, quantity: int):
        """
        Initializes an Operation object.

        Args:
            op_type (str): The type of operation, either 'add' or 'remove'.
            product (Product): The Product object on which the operation is to be performed.
            quantity (int): The quantity of the product for this operation.
        """
        self.op_type: str = op_type
        self.product: Product = product
        self.quantity: int = quantity



class Consumer(Thread):
    """
    A Consumer represents a buyer in the marketplace. It operates as a separate thread,
    managing multiple shopping carts, adding and removing products, and ultimately
    placing orders. It handles retries for 'add' operations if products are
    temporarily unavailable.
    """

    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):
        """
        Initializes a Consumer thread.

        Args:
            carts (List[List[Dict]]): A list of shopping cart definitions. Each cart
                                      is a list of dictionaries, where each dictionary
                                      represents an operation ('type', 'product', 'quantity').
            marketplace (Marketplace): The shared marketplace instance to interact with.
            retry_wait_time (float): The time in seconds to wait before retrying
                                     an 'add' operation if a product is unavailable.
            **kwargs: Arbitrary keyword arguments passed to the Thread constructor.
        """
        super(Consumer, self).__init__(**kwargs)
        self.carts: List[List[Operation]] = []
        for cart_operations in carts:
            ops = []
            for operation in cart_operations:
                ops.append(Operation(
                    operation['type'],
                    operation['product'],
                    operation['quantity'])
                )
            self.carts.append(ops)

        self.marketplace: Marketplace = marketplace
        self.retry_wait_time: float = retry_wait_time

        # A list to store products bought by this consumer (though not explicitly used in run)
        self.products: List[Product] = []

    def run(self):
        """
        The main execution method for the Consumer thread.
        It iterates through each defined cart, performs the specified operations
        (adding/removing products with retries), and then places the order.
        """
        # Block Logic: Iterates through each pre-defined list of operations,
        # each representing a shopping cart's actions.
        for operations in self.carts:
            # Pre-condition: Consumer is ready to start a new cart.
            # Post-condition: A new cart_id is obtained from the marketplace.
            cart_id: int = self.marketplace.new_cart()

            # Block Logic: Processes all operations for the current cart.
            # Invariant: `operations` list shrinks as operations are completed.
            # Pre-condition: `cart_id` is a valid ID for the current cart.
            while operations:
                # Conditional Logic: Handles 'add' operations.
                if operations[0].op_type == 'add':
                    # Attempt to add the product to the cart.
                    if self.marketplace.add_to_cart(cart_id, operations[0].product):
                        operations[0].quantity -= 1 # Decrement quantity on successful add.
                    else:
                        # If adding fails (product unavailable), wait and retry.
                        time.sleep(self.retry_wait_time)
                        continue # Skip to the next iteration of the while loop to retry.

                    # If all requested quantity for the current operation is added,
                    # move to the next operation.
                    if operations[0].quantity == 0:
                        operations = operations[1:]
                # Conditional Logic: Handles 'remove' operations.
                elif operations[0].op_type == 'remove':
                    # Block Logic: Removes the specified quantity of the product from the cart.
                    # Invariant: `operations[0].quantity` decreases with each successful removal.
                    while operations[0].quantity > 0:
                        self.marketplace.remove_from_cart(cart_id, operations[0].product)
                        operations[0].quantity -= 1
                    # After all removals for this operation are done, move to the next operation.
                    operations = operations[1:]

            # Block Logic: Places the final order for the current cart.
            # Pre-condition: All add/remove operations for the cart have been attempted.
            # Post-condition: `final_products` contains the list of products successfully ordered.
            final_products: List[Product] = self.marketplace.place_order(cart_id)
            for product in final_products:
                print(self.name + " bought " + str(product))




from threading import Lock
from typing import List, Dict

from tema.product import Product


class MarketplaceProduct:
    """
    Represents a product managed within the Marketplace, associating it with a producer
    and providing a lock for concurrent access control.
    """
    def __init__(self, producer_id: int, product: Product):
        """
        Initializes a MarketplaceProduct.

        Args:
            producer_id (int): The ID of the producer that supplied this product.
            product (Product): The Product object being managed.
        """
        self.producer_id = producer_id
        self.product = product
        self.lock: Lock = Lock()


class Cart:
    """
    Represents a shopping cart for a consumer. It stores the products added to it
    and provides methods for managing these products.
    """
    def __init__(self, cart_id: int):
        """
        Initializes a Cart object.

        Args:
            cart_id (int): A unique identifier for this cart.
        """
        self.cart_id: int = cart_id
        self.products: List[MarketplaceProduct] = []

    def add_product(self, product: MarketplaceProduct):
        """
        Adds a MarketplaceProduct to this cart.

        Args:
            product (MarketplaceProduct): The product to add to the cart.
        """
        self.products.append(product)

    def remove_product(self, product: MarketplaceProduct):
        """
        Removes a specific MarketplaceProduct from this cart.

        Args:
            product (MarketplaceProduct): The product to remove.
        """
        if product in self.products:
            self.products.remove(product)

    def find_product_in_cart(self, product) -> [MarketplaceProduct]:
        """
        Searches for a specific Product within the items currently in the cart.

        Args:
            product (Product): The Product object to find.

        Returns:
            MarketplaceProduct: The MarketplaceProduct instance if found, None otherwise.
        """
        for market_product in self.products:
            if market_product.product == product:
                return market_product

        return None


class Marketplace:
    """
    The central marketplace where producers publish products and consumers create carts,
    add/remove products, and place orders. It manages product inventory across producers
    and handles cart operations.
    """

    def __init__(self, queue_size_per_producer):
        """
        Initializes the Marketplace.

        Args:
            queue_size_per_producer (int): The maximum number of items a single producer
                                           can have in the marketplace at any given time.
        """
        self.max_items: int = queue_size_per_producer

        # Lists to keep track of registered consumers and producers (by their IDs).
        # Note: self.consumers is declared but not explicitly used in the provided code snippet.
        self.consumers: List[int] = []
        self.producers: List[int] = []

        # Dictionary to store products, keyed by producer ID.
        self.products: Dict[int, List[MarketplaceProduct]] = {}
        # Dictionary to store carts, keyed by cart ID.
        self.carts: Dict[int, Cart] = {}

    def register_producer(self) -> int:
        """
        Registers a new producer with the marketplace.

        Returns:
            int: The unique ID assigned to the registered producer.
        """
        producer_id = len(self.producers)

        self.producers.append(producer_id)
        self.products[producer_id] = [] # Initialize product list for this new producer.

        return producer_id

    def publish(self, producer_id, product) -> bool:
        """
        Publishes a product from a producer to the marketplace.

        Args:
            producer_id (int): The ID of the producer publishing the product.
            product (Product): The Product object to publish.

        Returns:
            bool: True if the product was successfully published, False otherwise
                  (e.g., if the producer's queue is full).
        """
        # Conditional Logic: Checks if the producer's current inventory exceeds the maximum allowed.
        if len(self.products[producer_id]) >= self.max_items:
            return False
        else:
            self.products[producer_id].append(MarketplaceProduct(producer_id, product))
            return True

    def new_cart(self) -> int:
        """
        Creates a new shopping cart and registers it with the marketplace.

        Returns:
            int: The unique ID assigned to the new cart.
        """
        cart_id = len(self.carts) + 1 # Assign a new unique cart ID.

        self.carts[cart_id] = Cart(cart_id)
        return cart_id

    def add_to_cart(self, cart_id: int, product: Product) -> bool:
        """
        Attempts to add a specified product to a consumer's cart.
        It searches for the product across all producers and locks it if found
        to prevent other consumers from buying it concurrently.

        Args:
            cart_id (int): The ID of the cart to add the product to.
            product (Product): The Product object to add.

        Returns:
            bool: True if the product was successfully added, False if the product
                  is not available or already locked by another cart.
        """
        # Block Logic: Iterates through products from all producers to find the desired product.
        for producer_products in self.products.values():
            for market_product in producer_products:
                # Conditional Logic: Checks if the product matches and is not currently locked.
                if market_product.product == product and not market_product.lock.locked():
                    market_product.lock.acquire()  # Acquire lock to reserve the product.
                    self.carts[cart_id].add_product(market_product)  # Add the product to the cart.
                    return True # Product successfully added.

        # If the product was not found or was locked, return False.
        return False

    def remove_from_cart(self, cart_id: int, product: Product):
        """
        Removes a product from a consumer's cart and releases its lock.

        Args:
            cart_id (int): The ID of the cart from which to remove the product.
            product (Product): The Product object to remove.
        """
        # Finds the specific MarketplaceProduct instance in the cart.
        market_product: MarketplaceProduct = self.carts[cart_id].find_product_in_cart(product)
        # Removes the product from the cart's list.
        self.carts[cart_id].remove_product(market_product)
        # Releases the lock, making the product available again for other carts or re-addition.
        market_product.lock.release()

    def place_order(self, cart_id: int) -> List[Product]:
        """
        Places an order for all items currently in the specified cart.
        This operation finalizes the purchase, removes the bought products
        from the marketplace's inventory, and returns the list of purchased products.

        Args:
            cart_id (int): The ID of the cart for which to place the order.

        Returns:
            List[Product]: A list of Product objects that were successfully purchased.
        """
        # Collects the actual Product objects from the MarketplaceProduct wrappers in the cart.
        final_products = [marketProduct.product for marketProduct in self.carts[cart_id].products]

        # Block Logic: Removes the purchased products from the marketplace's inventory.
        # This simulates the physical removal of items after a sale.
        for market_product in self.carts[cart_id].products:
            for producer_product in self.products[market_product.producer_id]:
                if producer_product == market_product:
                    self.products[market_product.producer_id].remove(producer_product)
                    break
        # The cart is effectively emptied by returning its contents; it can be reused or discarded.
        # Its internal `products` list is now empty as items have been moved to `final_products`
        # and removed from global inventory.
        return final_products


import time


from threading import Thread
from typing import List

from tema.marketplace import Marketplace
from tema.product import Product



class Production:
    """
    Represents a specific production task for a Producer, defining what product
    to produce, in what quantity, and with what wait time between productions.
    """
    def __init__(self, product: Product, quantity: int, wait_time: float):
        """
        Initializes a Production task.

        Args:
            product (Product): The Product object to be produced.
            quantity (int): The total quantity of this product to produce.
            wait_time (float): The time in seconds to wait after producing
                               one unit of this product.
        """
        self.product: Product = product
        self.quantity: int = quantity
        self.wait_time: float = wait_time

        self.number_produced = 0


class Producer(Thread):
    """
    A Producer represents a seller in the marketplace. It operates as a separate thread,
    continuously producing products and publishing them to the marketplace according
    to a defined production schedule.
    """

    def __init__(self, products, marketplace, republish_wait_time, **kwargs):
        """
        Initializes a Producer thread.

        Args:
            products (List[Tuple[Product, int, float]]): A list of production definitions.
                                                          Each definition is a tuple containing
                                                          (Product, quantity, wait_time).
            marketplace (Marketplace): The shared marketplace instance to interact with.
            republish_wait_time (float): The time in seconds to wait before attempting
                                         to republish a product if the marketplace's
                                         queue for this producer is full.
            **kwargs: Arbitrary keyword arguments passed to the Thread constructor.
        """
        super(Producer, self).__init__(**kwargs)
        self.productions: List[Production] = []
        for prod in products:
            self.productions.append(Production(prod[0], prod[1], prod[2]))
        self.marketplace: Marketplace = marketplace
        self.republish_wait_time: float = republish_wait_time

        # Registers this producer with the marketplace and obtains a unique ID.
        self.producer_id: int = self.marketplace.register_producer()

    def run(self):
        """
        The main execution method for the Producer thread.
        It continuously produces and publishes products to the marketplace.
        If a product cannot be published (e.g., marketplace queue is full),
        it waits and retries. Once a product's full quantity is produced,
        it cycles to the next product in its production list.
        """
        # Block Logic: The producer's main loop, running indefinitely to simulate
        # continuous production.
        while True:
            # Conditional Logic: Checks if the current product's required quantity has been produced.
            if self.productions[0].number_produced < self.productions[0].quantity:
                # Attempt to publish the current product to the marketplace.
                if self.marketplace.publish(self.producer_id, self.productions[0].product):
                    self.productions[0].number_produced += 1 # Increment count on successful publish.
                # Invariant: Producer waits `wait_time` regardless of publish success to simulate production time.
                time.sleep(self.productions[0].wait_time)
            else:
                # Block Logic: If the current product's quota is met, reset its counter
                # and move it to the end of the production list to cycle.
                self.productions[0].number_produced = 0
                self.productions = self.productions[1:] + [self.productions[0]]
