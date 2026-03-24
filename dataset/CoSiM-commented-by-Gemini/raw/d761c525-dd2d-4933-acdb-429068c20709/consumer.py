


"""



This module implements a multi-threaded producer-consumer simulation of an e-commerce marketplace.







It consists of three main components:



- Producer: A thread that generates products and publishes them to the marketplace.



- Consumer: A thread that simulates a customer adding products to a cart and placing an order.



- Marketplace: A central class that manages the inventory of products, producers, and customer carts,



              coordinating the interactions between producers and consumers.







The simulation uses threading to model concurrent producers and consumers interacting with the shared marketplace.



"""











from threading import Thread



from time import sleep







class Consumer(Thread):



    """



    Represents a consumer thread that interacts with the marketplace.







    Each consumer processes a list of carts, where each cart contains a sequence of



    'add' and 'remove' commands for products.



    """



    







    def __init__(self, carts, marketplace, retry_wait_time, **kwargs):



        """



        Initializes a Consumer thread.







        :param carts: A list of carts, where each cart is a list of commands.



        :param marketplace: The shared Marketplace object.



        :param retry_wait_time: Time to wait in seconds before retrying to add a product.



        :param kwargs: Additional keyword arguments for the Thread constructor.



        """



        



        self.carts = carts



        self.marketplace = marketplace



        self.retry_wait_time = retry_wait_time



        super(Consumer, self).__init__(**kwargs)







    def run(self):



        """



        The main execution logic for the consumer thread.







        Iterates through its assigned carts, processes the commands in each cart,



        places an order, and prints the items purchased.



        """







        



        # Process each cart assigned to this consumer.



        for cart in self.carts:







            



            # Get a new cart ID from the marketplace for the current shopping session.



            cart_id = self.marketplace.new_cart()







            











            # Execute commands within the cart (add/remove products).



            for command in cart:



                if command['type'] == 'add':



                    for i in range(command['quantity']):







                        



                        # Attempt to add the product to the cart, retrying if it's not available.



                        # This loop represents a blocking attempt to acquire a product.



                        while not self.marketplace.add_to_cart(cart_id, command['product']):











                            # If the product is not available, wait before retrying.



                            sleep(self.retry_wait_time)







                elif command['type'] == 'remove':



                    for i in range(command['quantity']):



                        self.marketplace.remove_from_cart(cart_id, command['product'])







            



            # Finalize the purchase by placing the order.



            order_list = self.marketplace.place_order(cart_id)







            



            # Print the successfully purchased items for this cart.



            for ol in order_list:



                print(self.name, end=" ")



                print("bought ", end="")



                print(ol)







# The following content seems to be from other files, concatenated into this one.



# Comments are added assuming they are part of the same logical system.



>>>> file: marketplace.py















class Marketplace:



    """



    Manages the entire marketplace, including producers, products, and carts.







    This class is the central hub for the simulation, providing an interface



    for producers and consumers. Note: This implementation may not be thread-safe



    as list operations are not protected by locks.



    """



    







    def __init__(self, queue_size_per_producer):



        """



        Initializes the Marketplace.







        :param queue_size_per_producer: The maximum number of products a single producer can have in the queue.



        """



        



        self.producer_count = 0



        self.producers_list = []



        self.carts_count = 0



        self.carts_list = []



        self.products_q = []



        self.products_count = 0



        self.queue_size_per_producer = queue_size_per_producer







    def register_producer(self):



        """



        Registers a new producer with the marketplace.







        :return: The ID assigned to the new producer.



        """



        



        self.producer_count = self.producer_count + 1



        self.producers_list.append([self.producer_count, 0])



        return self.producer_count







    def publish(self, producer_id, product):



        """



        Allows a producer to publish a new product to the marketplace.







        The product is added to the general product queue if the producer has not



        exceeded its individual queue size limit.







        :param producer_id: The ID of the producer publishing the product.



        :param product: The product to be published.



        :return: True if the product was published successfully, False otherwise.



        """



        







        



        # Find the producer to check their current published count.



        for p in self.producers_list:



            if p[0] == producer_id:







                



                # Block publishing if the producer's queue is full.



                if p[1] == self.queue_size_per_producer:



                    return False







                self.products_count = self.products_count + 1







                



                # Add product to the queue with metadata: [product_id, product_info, producer_id, in_cart_status]



                self.products_q.append(



                    [self.products_count, product, producer_id, 0])



                p[1] = p[1] + 1



                return True







    def new_cart(self):



        """



        Creates a new, empty cart for a consumer.







        :return: The ID of the newly created cart.



        """



        



        self.carts_count = self.carts_count + 1



        self.carts_list.append([self.carts_count])



        return self.carts_count







    def add_to_cart(self, cart_id, product):



        """



        Adds a product to a consumer's cart.







        Searches for an available product of the specified type in the product queue



        and adds it to the cart if found.







        :param cart_id: The ID of the cart to add the product to.



        :param product: The type of product to add.



        :return: True if the product was added successfully, False otherwise.



        """



        







        



        # Find the target cart.



        for c in self.carts_list:



            if c[0] == cart_id:







                



                # Search for an available product of the correct type.



                for i in self.products_q:







                    



                    # An available product matches the type and is not already in another cart (i[3] == 0).



                    if i[1] == product and i[3] == 0:







                        



                        # Add the product to the cart.



                        c.append(i)







                        



                        # Mark the product as reserved in a cart.



                        i[3] = 1



                        return True



                # No available product of this type was found.



                return False



        return False







    def remove_from_cart(self, cart_id, product):



        """



        Removes a product from a consumer's cart.







        :param cart_id: The ID of the cart to remove from.



        :param product: The type of product to remove.



        """



        



        



        # Find the target cart.



        for c in self.carts_list:



            if c[0] == cart_id:







                



                # Find the corresponding product in the main product queue to update its status.



                for i in self.products_q:



                    if i[1] == product:







                        



                        # Find the product within the cart's item list.



                        for x, y in enumerate(c[1:]):



                            if y[1] == product:







                                



                                # Remove product from cart.



                                c.pop(x+1)







                                



                                # Mark the product as available again in the marketplace.



                                i[3] = 0



                                break



                        break



                break







    def place_order(self, cart_id):



        """



        Finalizes an order for a given cart.







        This removes the products from the marketplace inventory permanently.







        :param cart_id: The ID of the cart for which to place the order.



        :return: A list of products that were in the order.



        """



        



        order_list = []







        



        # Find the target cart.



        for cart in self.carts_list:



            if cart[0] == cart_id:







                



                # Get the list of products in the cart.



                products = cart[1:]







                



                # Process each product in the order.



                for pr in products:







                    



                    # Find and remove the product from the main products queue.



                    for x, y in enumerate(self.products_q):



                        if y[0] == pr[0]:







                            



                            self.products_q.pop(x)







                            



                            # Decrement the producer's active product count.



                            for producer in self.producers_list:



                                if y[2] == producer[0]:







                                    



                                    producer[1] = producer[1] - 1



                                    break



                            break







                    # Add product name to the final order list.



                    order_list.append(pr[1])



                return order_list



>>>> file: producer.py











from threading import Thread



from time import sleep







class Producer(Thread):



    """



    Represents a producer thread that generates and publishes products.



    """



    







    def __init__(self, products, marketplace, republish_wait_time, **kwargs):



        """



        Initializes a Producer thread.







        :param products: A list of products that this producer can generate.



        :param marketplace: The shared Marketplace object.



        :param republish_wait_time: Time to wait before retrying to publish a product if the queue is full.



        :param kwargs: Additional keyword arguments for the Thread constructor.



        """



        



        super(Producer, self).__init__(**kwargs)



        self.products = products



        self.marketplace = marketplace



        self.republish_wait_time = republish_wait_time







    def run(self):



        """



        The main execution logic for the producer thread.







        Registers with the marketplace and then enters an infinite loop to produce



        and publish products.



        """







        



        # Register this producer with the marketplace to get a unique ID.



        self.id = self.marketplace.register_producer()







        # Infinite loop to continuously produce products.



        while True:



            for p in self.products:



                q = p[1]



                for i in range(q):







                    



                    # Attempt to publish a product.



                    published = self.marketplace.publish(self.id, p[0])







                    



                    # If publishing failed (e.g., producer queue is full), wait and retry.



                    if not published:



                        # This decrement is intended to retry the current iteration, but a 'while' loop



                        # would be a more conventional way to handle retries.



                        i = i - 1



                        sleep(self.republish_wait_time)



                    # Wait for a product-specific time before producing the next one.



                    sleep(p[2])











# This section defines the data models for the products in the marketplace.



from dataclasses import dataclass











@dataclass(init=True, repr=True, order=False, frozen=True)



class Product:



    """A base dataclass for a generic product with a name and price."""



    



    name: str



    price: int











@dataclass(init=True, repr=True, order=False, frozen=True)



class Tea(Product):



    """A dataclass for Tea, inheriting from Product and adding a 'type' attribute."""



    



    type: str











@dataclass(init=True, repr=True, order=False, frozen=True)



class Coffee(Product):



    """A dataclass for Coffee, inheriting from Product and adding acidity and roast level."""



    



    acidity: str



    roast_level: str




