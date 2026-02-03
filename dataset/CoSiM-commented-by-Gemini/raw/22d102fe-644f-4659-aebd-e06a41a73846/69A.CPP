/**
 * @file 69A.CPP
 * @brief Solution to the "Young Physicist" problem.
 * @see https://codeforces.com/problemset/problem/69/A
 *
 * This program determines if a body is in equilibrium given a set of forces
 * acting on it. A body is in equilibrium if the vector sum of all forces is
 * zero.
 *
 * Algorithm:
 * 1. Read the number of forces (n).
 * 2. Initialize the sum of forces in each dimension (x, y, z) to zero.
 * 3. Loop n times:
 *    a. Read the x, y, and z components of a force vector.
 *    b. Add the components to their respective sums.
 * 4. If the sums of all three components are zero, the body is in
 *    equilibrium, so print "YES".
 * 5. Otherwise, print "NO".
 *
 * Time Complexity: O(N), where N is the number of forces.
 * Space Complexity: O(1).
 */
/* A guy named Vasya attends the final grade of a high school. One day Vasya
decided to watch a match of his favorite hockey team. And, as the boy loves
hockey very much, even more than physics, he forgot to do the homework.
Specifically, he forgot to complete his physics tasks. Next day the teacher got
very angry at Vasya and decided to teach him a lesson. He gave the lazy student
a seemingly easy task: You are given an idle body in space and the forces that
affect it. The body can be considered as a material point with coordinates (0;
0; 0). Vasya had only to answer whether it is in equilibrium. "Piece of cake"
thought Vasya, we need only to check if the sum of all vectors is equal to 0.
So, Vasya began to solve the problem. But later it turned out that there can be
lots and lots of these forces, and Vasya can not cope without your help. Help
him. Write a program that determines whether a body is idle or is moving by the
given vectors of forces.

Input
The first line contains a positive integer n (1   n   100), then follow n lines
containing three integers each: the xi coordinate, the yi coordinate and the zi
coordinate of the force vector, applied to the body ( - 100   xi, yi, zi   100).

Output
Print the word "YES" if the body is in equilibrium, or the word "NO" if it is
not. */

#include <iostream>

using namespace std;

int main() {
  int n;
  cin >> n;
  int x, y, z;
  int sumx = 0, sumy = 0, sumz = 0;
  // Invariant: This loop reads each force vector and accumulates the sum of
  // components in each dimension.
  for (int i = 0; i < n; i++) {
    cin >> x >> y >> z;
    sumx += x;
    sumy += y;
    sumz += z;
  }
  // Post-condition: If the net force in all three dimensions is zero, the body
  // is in equilibrium.
  if (!sumx && !sumy && !sumz)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
