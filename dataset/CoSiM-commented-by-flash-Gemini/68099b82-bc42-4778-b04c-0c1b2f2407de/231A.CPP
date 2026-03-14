/**
 * @68099b82-bc42-4778-b04c-0c1b2f2407de/231A.CPP
 * @brief This program solves Codeforces problem 231A, "Team".
 * It reads 'n' problems. For each problem, it takes three binary inputs (0 or 1)
 * representing whether each of three friends is sure about the solution.
 * A problem is considered solvable if at least two friends are sure about it.
 * The program counts and outputs the total number of solvable problems.
 *
 * Algorithm: Iterative processing of problems, checking the sum of binary inputs for a majority.
 * Time Complexity: O(N) where N is the number of problems.
 * Space Complexity: O(1) for storing counters and input.
 #include <iostream> // For standard input/output operations (cin, cout).
 using namespace std; // Uses the standard namespace.

 /**
  * @brief Main function to process problem certainty and count solvable problems.
  * Reads 'n' problems, three certainty values for each, and increments a counter
  * if at least two friends are sure.
  * @return 0 on successful execution.
  */
 int main() {
   int n; // Declares 'n', representing the total number of problems.
   int a, b, c; // Declares variables for each friend's certainty (0 or 1).
   int cont = 0; // Accumulator for the total number of solvable problems.

   cin >> n; // Reads the number of problems from standard input.

   /**
    * Block Logic: Iterates through each of the 'n' problems.
    * For each problem, it reads the certainty of three friends and determines
    * if the problem is solvable (sum of certainties > 1, meaning at least two sure friends).
    * Invariant: `cont` accurately reflects the number of solvable problems processed so far.
    */
   while (n > 0) { // Loops 'n' times, decrementing 'n' in each iteration.
     // Reads the certainty values for friends a, b, and c for the current problem.
     cin >> a >> b >> c; 

     // Inline: Checks if the sum of certainties is greater than 1 (meaning at least two friends are sure).
     if (a + b + c > 1) 
       cont++; // Increments the total count of solvable problems.
     n--; // Decrements the problem counter.
   }
   cout << cont; // Prints the final count of solvable problems.
   return 0; // Indicates successful program termination.
 }
