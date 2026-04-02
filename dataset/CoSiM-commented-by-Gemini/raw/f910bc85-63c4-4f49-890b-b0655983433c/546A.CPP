/**
 * @file 546A.CPP
 * @brief Solution to the "Soldier and Bananas" problem (Codeforces 546A).
 *
 * Problem Statement:
 * A soldier wants to buy 'w' bananas. The first banana costs 'k', the second
 * costs '2k', and so on (the i-th banana costs i*k). The soldier has 'n'
 * dollars. The task is to calculate the amount of money the soldier must borrow
 * to buy all 'w' bananas. If the soldier has enough money, no borrowing is needed.
 *
 * Algorithm:
 * This program calculates the total cost using a direct mathematical formula,
 * which is highly efficient. The total cost is the sum of an arithmetic series:
 * k*1 + k*2 + ... + k*w = k * (1 + 2 + ... + w).
 * The sum of the first 'w' integers is given by the formula w * (w + 1) / 2.
 * The total cost is therefore `k * (w * (w + 1) / 2)`.
 * The program then subtracts the soldier's initial money 'n' from this total
 * cost to find the amount that needs to be borrowed. If the cost is less than
 * or equal to 'n', the amount to borrow is 0.
 *
 * Time Complexity: O(1), as the calculation is done with a direct formula.
 * Space Complexity: O(1).
 */
// RUMAN CSE (6th Intake) BRUR
#include <algorithm>
#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <strstream>
#include <vector>
#define MAX(a, b) a > b ? a : b
#define MIN(a, b) a < b ? a : b
#define PI 3.1416

using namespace std;
int main() {
  // freopen("input.txt","r",stdin);
  // freopen("output.txt","w",stdout);
  long long k, n, w, cost, borrow;
  cin >> k >> n >> w;
  
  // Calculate the total cost using the formula for the sum of an arithmetic series.
  // This is an O(1) approach, avoiding a loop.
  cost = k * ((w * (w + 1)) / 2);
  
  // Determine the amount of money to borrow.
  if (cost > n)
    borrow = cost - n;
  else
    borrow = 0;
  
  cout << borrow << endl;

  return 0;
}
