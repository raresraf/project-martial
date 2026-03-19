/**
 * @file 1221C.CPP
 * @brief Solution to Codeforces Problem 1221C: "Perfect Team".
 *
 * This problem asks to find the maximum number of "perfect teams" that can be formed
 * given a certain number of programmers (`c`), mathematicians (`m`), and generalists (`x`).
 * A perfect team requires one programmer, one mathematician, and one additional person
 * who can be either a programmer, a mathematician, or a generalist.
 *
 * Algorithm: The problem exhibits a monotonic property (if `k` teams can be formed, 
 * then any number of teams less than `k` can also be formed). This allows for
 * binary search on the answer (the number of teams). The `team_possible` function
 * checks if a given number of teams (`ts`) can be formed.
 *
 * Time Complexity: O(T * log(max_teams)), where T is the number of test cases, 
 * and `max_teams` is the maximum possible number of teams (limited by `1e9` in this solution).
 * Each call to `team_possible` is O(1).
 * Space Complexity: O(1) as only a few variables are used per test case.
 */
// Original author reference comment.
//  [amitdu6ey]
// g++ -std=c++11 -o2 -Wall filename.cpp -o filename
#include <bits/stdc++.h> // Includes various standard libraries for C++, like iostream, algorithm, etc.

/** @def hell
 * @brief A large prime number, often used as a modulo for hashing or in number theory problems.
 */
#define hell 1000000009
/** @def bug1(x)
 * @brief Debug macro to print a variable's value to standard output, formatted as "$ " << x << " $".
 */
#define bug1(x) cout << "$ " << x << " $" << endl
/** @def bug2(x)
 * @brief Debug macro to print a variable's value to standard output, formatted as "% " << x << " %" << endl.
 */
#define bug2(x) cout << "% " << x << " %" << endl
/** @def bug3(x)
 * @brief Debug macro to print a variable's value to standard output, formatted as "# " << x << " #" << endl.
 */
#define bug3(x) cout << "# " << x << " #" << endl
/** @def ll
 * @brief Type alias for `long long`, used for integers that might exceed the range of `int`.
 */
#define ll long long
/** @def pb
 * @brief Alias for `push_back`, commonly used with `std::vector`.
 */
#define pb push_back
/** @def mp
 * @brief Alias for `make_pair`, commonly used with `std::pair`.
 */
#define mp make_pair
/** @def tr(z)
 * @brief Loop macro to iterate through containers using iterators.
 */
#define tr(z) for (auto it = z.begin(); it != z.end(); it++)
/** @def rloop(i, a, b)
 * @brief Reverse loop macro: for `i` from `a` down to `b` (inclusive).
 */
#define rloop(i, a, b) for (long long i = a; i >= b; i--)
/** @def loop(i, a, b)
 * @brief Forward loop macro: for `i` from `a` up to `b` (exclusive).
 */
#define loop(i, a, b) for (long long i = a; i < b; i++)
/** @def vbool
 * @brief Type alias for `std::vector<bool>`.
 */
#define vbool vector<bool>
/** @def vvbool
 * @brief Type alias for `std::vector<std::vector<bool>>`.
 */
#define vvbool vector<vector<bool>>
/** @def vchar
 * @brief Type alias for `std::vector<char>`.
 */
#define vchar vector<char>
/** @def vi
 * @brief Type alias for `std::vector<int>`.
 */
#define vi vector<int>
/** @def vvi
 * @brief Type alias for `std::vector<std::vector<int>>`.
 */
#define vvi vector<vector<int>>
/** @def pll
 * @brief Type alias for `std::pair<long long, long long>`.
 */
#define pll pair<long long, long long>
/** @def vll
 * @brief Type alias for `std::vector<long long>`.
 */
#define vll vector<long long>
/** @def vvl
 * @brief Type alias for `std::vector<std::vector<long long>>`.
 */
#define vvl vector<vector<long long>>
/** @def ninf
 * @brief Represents the minimum possible value for `long long` (`numeric_limits<long long>::min()`).
 */
#define ninf numeric_limits<long long>::min()
/** @def pinf
 * @brief Represents the maximum possible value for `long long` (`numeric_limits<long long>::max()`).
 */
#define pinf numeric_limits<long long>::max()
/** @def comp(a, b)
 * @brief Macro to compare two double values for approximate equality, accounting for floating-point precision issues.
 */
#define comp(a, b) (abs(a - b) < 1e-9) // to compare doubles
using namespace std;

/**
 * @brief Checks if it's possible to form `ts` (team size) number of perfect teams.
 *
 * A perfect team requires:
 * 1. One programmer.
 * 2. One mathematician.
 * 3. One additional person (who can be a programmer, mathematician, or generalist).
 *
 * @param c Available programmers.
 * @param m Available mathematicians.
 * @param x Available generalists.
 * @param ts The number of teams to attempt to form.
 * @return `true` if `ts` teams can be formed, `false` otherwise.
 */
bool team_possible(ll c, ll m, ll x, ll ts) {
  /**
   * Block Logic: Initial check - ensure enough programmers and mathematicians for the first two roles.
   * Pre-condition: `c`, `m`, `x` are non-negative, `ts` is non-negative.
   * Invariant: If `ts` exceeds available `c` or `m`, `false` is returned immediately.
   */
  if (ts > min(c, m)) { // Each team needs 1 programmer and 1 mathematician.
    return false;
  }
  // Deduct the programmers and mathematicians used for the first two roles in `ts` teams.
  m -= ts;
  c -= ts;

  /**
   * Block Logic: Check if there are enough remaining people for the "one additional person" role.
   * The remaining `c` (programmers), `m` (mathematicians), and `x` (generalists) can all serve as "additional persons".
   * Pre-condition: `c`, `m`, `x` are the remaining counts after assigning primary roles.
   * Invariant: If the total remaining personnel `c + m + x` is at least `ts`, `true` is returned.
   */
  if (ts <= c + m + x) { // Each of the `ts` teams needs one more person.
    return true;
  }
  return false;
}

/**
 * @brief Solves a single test case for the "Perfect Team" problem.
 *
 * It reads the counts of programmers, mathematicians, and generalists, then
 * uses binary search to find the maximum number of perfect teams that can be formed.
 */
void solve() {
  ll c, m, x; // 'c': programmers, 'm': mathematicians, 'x': generalists.
  cin >> c >> m >> x; // Reads input for the current test case.

  // Binary search variables:
  // 'l': lower bound for the number of teams.
  // 'r': upper bound for the number of teams (a sufficiently large number like 10^9).
  // 'mid': current guess for the number of teams.
  // 'ans': stores the maximum number of teams found to be possible.
  ll l = 0, r = 1e9, mid = (l + r) / 2;
  ll ans = 0;

  /**
   * Block Logic: Performs binary search on the answer (maximum number of teams).
   * Pre-condition: `l` and `r` define the search space, `ans` stores current best.
   * Invariant: The optimal solution lies within `[l, r]`.
   */
  while (l <= r) {
    mid = (l + r) / 2; // Calculate the middle point.
    /**
     * Block Logic: If `mid` teams are possible, try for more. Otherwise, try for fewer.
     * Pre-condition: `mid` is the current team count being tested.
     * Invariant: Search space is narrowed in each iteration.
     */
    if (team_possible(c, m, x, mid)) {
      l = mid + 1; // It's possible to form `mid` teams, so try for more.
      ans = mid;   // Update `ans` with this possible value.
    } else {
      r = mid - 1; // It's not possible to form `mid` teams, so reduce the upper bound.
    }
  }
  cout << ans << "\n"; // Prints the maximum number of perfect teams.
}

/**
 * @brief Main function of the program.
 * Sets up fast I/O and handles multiple test cases.
 */
int main() {
  // Functional Utility: Optimizes C++ standard streams for faster input/output.
  ios_base::sync_with_stdio(0);
  cout.tie(0); // Unties cout from cin.
  cin.tie(0);  // Unties cin from cout.
  // Example of commented-out file I/O operations, common in competitive programming setup.
  // freopen("input.txt","r",stdin);
  // freopen("output.txt","w",stdout);
  
  ll test_cases = 1; // Default to 1 test case if not specified.
  cin >> test_cases; // Reads the number of test cases.
  
  /**
   * Block Logic: Processes each test case by calling the `solve()` function.
   * Pre-condition: `test_cases` holds the number of scenarios to run.
   * Invariant: After each iteration, one test case has been solved.
   */
  while (test_cases--) {
    solve(); // Calls the function to solve the current test case.
  }
  return 0; // Indicates successful execution.
}
