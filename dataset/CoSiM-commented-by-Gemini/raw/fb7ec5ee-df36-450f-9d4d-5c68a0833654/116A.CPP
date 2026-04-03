/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem from Codeforces (Problem 116A).
 * @see https://codeforces.com/problemset/problem/116/A
 *
 * @description
 * The program calculates the minimum possible capacity of a tram. It is given the
 * number of stops and for each stop, the number of passengers exiting and entering
 * the tram. The tram is empty before the first stop.
 *
 * @algorithm
 * The problem can be solved by simulating the passenger count at each stop.
 * 1. Initialize `current_passengers` and `max_capacity` to 0.
 * 2. Loop through each of the 'n' stops.
 * 3. For each stop, read the number of exiting (`a`) and entering (`b`) passengers.
 * 4. Update the passenger count: `current_passengers = current_passengers - a + b`.
 * 5. After the update, the new `current_passengers` count represents the number of
 *    people in the tram as it leaves the station. This is a moment when capacity
 *    is needed.
 * 6. Update the required capacity: `max_capacity = max(max_capacity, current_passengers)`.
 * 7. After iterating through all stops, `max_capacity` will hold the minimum
 *    capacity required for the entire trip.
 *
 * @time_complexity O(N), where N is the number of stops, because the solution
 *                  involves a single pass through the stops.
 * @space_complexity O(1), as only a few variables are needed to track the counts,
 *                   regardless of the number of stops.
 */
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <utility>
#include <vector>

//==================================================================================
// Section: Standard Using and Typedefs for Competitive Programming
//==================================================================================
using namespace std;

typedef vector<int> VI;
typedef vector<VI> VII;
typedef vector<string> VS;
typedef vector<long> VL;
typedef long long LL;
typedef vector<LL> VLL;

//==================================================================================
// Section: Macros for Competitive Programming
//----------------------------------------------------------------------------------
// These macros are shorthands for common operations to speed up coding.
//==================================================================================
#define PB push_back
#define PF push_front
#define GRT(x) greater<(x)>()
#define ASORT(x) sort((x).begin(), (x).end())
#define DSORT(x, y) sort((x).begin(), (x).end(), greater<(y)>())
#define FILL(x, y) fill((x).begin(), (x).end(), (y))
#define COPY(x, y)                                                             
  (y).clear();                                                                 
  copy((x).begin(), (x).end(), back_inserter(y))

#define FOR(i, a, b) for (int i = (a); i < (b); i++)
#define FORL(i, a, b) for (long i = (a); i < (b); i++)
#define REP(i, n) FOR(i, 0, n)
#define REPL(i, n) FORL(i, 0, n)

#define DUMP(a) cerr << #a << " = " << (a) << endl
#define DUMP2(a, b)                                                            
  cerr << #a << " = " << (a) << ", " << #b << " = " << (b) << endl
#define DUMP3(a, b, c)                                                         
  cerr << #a << " = " << (a) << ", " << #b << " = " << (b) << ", " << #c       
       << " = " << (c) << endl

int n;

/**
 * @brief Contains the core logic for solving the problem.
 */
void solve() {
  cin >> n;
  long a, b;
  // p: tracks the current number of passengers in the tram.
  long p = 0;
  // p_max: tracks the maximum number of passengers seen so far, which
  // determines the required tram capacity.
  long p_max = 0;

  /**
   * @brief Block Logic: Simulate passenger flow over 'n' stops.
   * At each stop, update the current passenger count and check if a new
   * maximum capacity is required.
   */
  REP(i, n) {
    cin >> a >> b;
    // Passengers exit first.
    p -= a;
    // Then new passengers enter.
    p += b;
    // Update the maximum capacity needed.
    p_max = max(p, p_max);
  }

  // Output the minimum required capacity.
  cout << p_max << endl;
}

/**
 * @brief Main function, entry point of the program.
 */
int main(int argc, char *argv[]) {
  solve();
  return 0;
}
