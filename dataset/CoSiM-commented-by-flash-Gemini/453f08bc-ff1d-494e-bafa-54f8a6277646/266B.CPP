/**
 * @file 266B.CPP
 * @brief Solution for a competitive programming problem involving character rearrangement in a queue.
 *
 * This program simulates the process of boys ('B') moving past girls ('G') in a queue
 * over a given number of time steps. In each step, if a boy is immediately followed by a girl,
 * they swap positions.
 *
 * Algorithm: Iterative single-pass swap simulation.
 * Time Complexity: $O(K \times N)$ where K is the number of simulation steps and N is the queue length.
 * Space Complexity: $O(N)$ for storing the queue string.
 */
// please refer from line no. 207
/////*31022618*/////
//_MONU KUMAR
#include <bits/stdc++.h>
using namespace std;
#define fast                                                                   \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(0);                                                                  \
  cout.tie(0); /**< @brief Optimizes C++ standard streams for faster input/output. */
#define tc                                                                     \
  ll t = 1;                                                                    \
  cin >> t;                                                                    \
  while (t--) /**< @brief Macro for competitive programming: defines a test case loop. */
#define ll long long int /**< @brief Type definition for long long integer. */
#define ld long double /**< @brief Type definition for long double. */
#define ull unsigned long long /**< @brief Type definition for unsigned long long integer. */
#define pll pair<ll, ll> /**< @brief Type definition for a pair of long long integers. */
#define vi std::vector<ll> /**< @brief Type definition for a vector of long long integers. */
#define vp vector<pll> /**< @brief Type definition for a vector of pairs of long long integers. */
#define mll std::map<ll, ll> /**< @brief Type definition for a map with long long keys and values. */
#define mpl                                                                    \
  std:                                                                         \
  map<pll, ll> /**< @brief Type definition for a map with pairs of long long as keys and long long values. */
#define p1(x) cout << x << "\n"; /**< @brief Prints a single value followed by a newline. */
#define p2(x, y) cout << x << " " << y << "\n"; /**< @brief Prints two values separated by a space, followed by a newline. */
#define p3(x, y, z) cout << x << " " << y << " " << z << "\n"; /**< @brief Prints three values separated by spaces, followed by a newline. */
#define p4(x, y, z, w) cout << x << " " << y << " " << z << " " << w << "\n"; /**< @brief Prints four values separated by spaces, followed by a newline. */
#define trace(x)                                                               \
  cout << #x << ": " << x << " "                                               \
       << "\n"; /**< @brief Debug macro: prints variable name and its value. */
#define trace2(x, y)                                                           \
  cout << #x << ": " << x << " | " << #y << ": " << y << "\n"; /**< @brief Debug macro: prints two variable names and their values. */
#define trace3(x, y, z)                                                        \
  cout << #x << ":" << x << " | " << #y << ": " << y << " | " << #z << ": "    \
       << z << "\n"; /**< @brief Debug macro: prints three variable names and their values. */
#define trace4(a, b, c, d)                                                     \
  cout << #a << ": " << a << " | " << #b << ": " << b << " | " << #c << ": "   \
       << c << " | " << #d << ": " << d << "\n"; /**< @brief Debug macro: prints four variable names and their values. */
#define nL cout << "\n"; /**< @brief Prints a newline character. */
#define lp(i, l, r) for (long long int i = l; i < r; i++) /**< @brief Loop macro: iterates from `l` (inclusive) to `r` (exclusive). */
#define lpr(i, r, l) for (long long int i = r - 1; i >= l; i--) /**< @brief Loop macro: iterates from `r-1` (inclusive) down to `l` (inclusive). */
#define mem(a, x) memset(a, x, sizeof(a)) // set elements of array to some value /**< @brief Macro to set all bytes of an array to a specific value. */
#define pi 3.1415926535897932384626 /**< @brief Definition of Pi. */
#define mod 1000000007 /**< @brief Modulo value commonly used in competitive programming. */
#define big 1e18 /**< @brief Represents a large positive number (used as infinity). */
#define small -big /**< @brief Represents a large negative number (used as negative infinity). */
#define mp(a, b) make_pair(a, b) /**< @brief Macro for creating a pair. */
#define pb(x) push_back(x) /**< @brief Macro for pushing an element to the back of a vector. */
#define ff first /**< @brief Accesses the first element of a pair. */
#define ss second /**< @brief Accesses the second element of a pair. */
#define siz 256 /**< @brief Defines a size constant, possibly for character set or array dimensions. */
#define leadzero(a) __builtin_clz(a) // count leading zeroes /**< @brief Counts leading zeros in an integer (GCC builtin). */
#define trailzero(a) __builtin_ctz(a) // count trailing zeroes /**< @brief Counts trailing zeros in an integer (GCC builtin). */
#define bitcount(a) __builtin_popcount(a) // count set bits /**< @brief Counts set bits (1s) in an integer (GCC builtin). */
#define all(v) v.begin(), v.end() /**< @brief Returns iterators to the beginning and end of a container for algorithms. */
#define lps(v, x) (lower_bound(all(v), x) - v.begin()) /**< @brief Finds the first element not less than x in a sorted range. */
#define ups(v, x) (upper_bound(all(v), x) - v.begin()) /**< @brief Finds the first element greater than x in a sorted range. */
#define ina(a, n)                                                              \
  ll a[n];                                                                     \
  lp(i, 0, n) cin >> a[i]; /**< @brief Macro to declare and read 'n' long long integers into an array. */
#define ona(a, n) lp(i, 0, n) cout << a[i] << " "; /**< @brief Macro to print 'n' long long integers from an array. */
//#########################################################################################
/**
 * @brief Checks if two strings are anagrams of each other.
 * @param s1 The first character array (string).
 * @param s2 The second character array (string).
 * @return True if s1 and s2 are anagrams, false otherwise.
 *
 * Algorithm: Character counting. It uses a frequency array to count character
 * occurrences in s1 and decrements for s2. If all counts are zero and lengths match,
 * they are anagrams.
 * Time Complexity: O(N) where N is the length of the strings.
 * Space Complexity: O(Siz) where Siz is the character set size (256 for ASCII).
 */
bool anagram(char *s1, char *s2) {
  ll count[siz];
  mem(count, 0);
  ll i;
  for (i = 0; s1[i] && s2[i]; i++) {
    count[s1[i]]++;
    count[s2[i]]--;
  }
  if (s1[i] || s2[i]) {
    return false;
  }
  lp(i, 0, siz) {
    if (count[i]) {
      return false;
    }
  }
  return true;
}
//###########################################################################################
/**
 * @brief Computes (a * b) % zz using binary multiplication to prevent overflow.
 * @param a The first operand.
 * @param b The second operand.
 * @param zz The modulus.
 * @return The result of (a * b) % zz.
 *
 * Algorithm: Binary exponentiation (also known as Russian peasant multiplication).
 * Time Complexity: O(log b)
 * Space Complexity: O(1)
 */
ll moduloMultiplication(ll a, ll b, ll zz) {
  ll res = 0;
  a %= mod;
  while (b) {
    if (b & 1)
      res = (res + a) % zz;

    a = (2 * a) % zz;

    b >>= 1;
  }
  return res;
}
//###########################################################################################

/**
 * @brief Converts a string of digits to a long long integer.
 * @param s The string to convert.
 * @return The converted long long integer.
 *
 * Algorithm: Iterative parsing of digits.
 * Time Complexity: O(L) where L is the length of the string.
 * Space Complexity: O(1)
 */
ll convert(string s) {
  bool o = true;
  ll ans = 0;
  for (ll i = 0; i < s.length(); i++) {
    ans = ans * 10 + (s[i] - 48);
  }
  return ans;
}
//###########################################################################################
/**
 * @brief Computes (a^b) % modulo using binary exponentiation.
 * @param a The base.
 * @param b The exponent.
 * @param modulo The modulus.
 * @return The result of (a^b) % modulo.
 *
 * Algorithm: Binary exponentiation (exponentiation by squaring).
 * Time Complexity: O(log b)
 * Space Complexity: O(log b) due to recursive calls (can be O(1) iteratively).
 */
ll powmod(ll a, ll b, ll modulo) {
  if (b == 0 || a == 1)
    return 1;
  ll half = powmod(a, (b / 2), modulo) % modulo;
  ll full = (half * half) % modulo;

  if (b % 2)
    return (full * a) % modulo;
  return full % modulo;
}
//###########################################################################################
/**
 * @brief Computes the modular multiplicative inverse of a under modulo.
 * @param a The number for which to find the inverse.
 * @param modulo The modulus (must be a prime number for Fermat's Little Theorem).
 * @return The modular multiplicative inverse of a, assuming gcd(a, modulo) = 1.
 *
 * Algorithm: Fermat's Little Theorem (a^(modulo-2) % modulo).
 * Time Complexity: O(log modulo)
 * Space Complexity: O(log modulo)
 */
ll invmod(ll a, ll modulo) {
  ll check = powmod(a, modulo - 2, modulo) % modulo;
  return check;
}
//###########################################################################################

/**
 * @brief Returns the maximum of two long long integers.
 * @param a The first long long integer.
 * @param b The second long long integer.
 * @return The greater of a and b.
 * Time Complexity: O(1)
 * Space Complexity: O(1)
 */
ll max(ll a, ll b) {
  if (a > b) {
    /* code */
    return a;
  }
  return b;
}
//###########################################################################################
/**
 * @brief Returns the minimum of two long long integers.
 * @param a The first long long integer.
 * @param b The second long long integer.
 * @return The smaller of a and b.
 * Time Complexity: O(1)
 * Space Complexity: O(1)
 */
ll min(ll a, ll b) {
  if (a < b) {
    /* code */
    return a;
  }
  return b;
}
//###########################################################################################
/**
 * @brief Returns the minimum of three long long integers.
 * @param a The first long long integer.
 * @param b The second long long integer.
 * @param c The third long long integer.
 * @return The smallest of a, b, and c.
 * Time Complexity: O(1)
 * Space Complexity: O(1)
 */
ll min(ll a, ll b, ll c) {
  if (a < b) {
    /* code */
    return min(a, c);
  } else {
    return min(b, c);
  }
}
//###########################################################################################
/**
 * @brief Returns the maximum of three long long integers.
 * @param a The first long long integer.
 * @param b The second long long integer.
 * @param c The third long long integer.
 * @return The largest of a, b, and c.
 * Time Complexity: O(1)
 * Space Complexity: O(1)
 */
ll max(ll a, ll b, ll c) {
  if (a < b) {
    /* code */
    return max(b, c);
  } else {
    return min(a, c); // This looks like a bug in original code, should be max(a,c)
  }
}
//###########################################################################################
/**
 * @brief Returns the minimum of a long long integer and an int.
 * @param a The long long integer.
 * @param b The int.
 * @return The smaller of a and b.
 * Time Complexity: O(1)
 * Space Complexity: O(1)
 */
ll min(ll a, int b) {
  if (a < b)
    return a;
  return b;
}
//###########################################################################################
/**
 * @brief Returns the minimum of an int and a long long integer.
 * @param a The int.
 * @param b The long long integer.
 * @return The smaller of a and b.
 * Time Complexity: O(1)
 * Space Complexity: O(1)
 */
ll min(int a, ll b) {
  if (a < b)
    return a;
  return b;
}
//###########################################################################################
/**
 * @brief Returns the maximum of a long long integer and an int.
 * @param a The long long integer.
 * @param b The int.
 * @return The greater of a and b.
 * Time Complexity: O(1)
 * Space Complexity: O(1)
 */
ll max(ll a, int b) {
  if (a > b)
    return a;
  return b;
}
//###########################################################################################
/**
 * @brief Returns the maximum of an int and a long long integer.
 * @param a The int.
 * @param b The long long integer.
 * @return The greater of a and b.
 * Time Complexity: O(1)
 * Space Complexity: O(1)
 */
ll max(int a, ll b) {
  if (a > b)
    return a;
  return b;
}
//###########################################################################################
/**
 * @brief Computes the greatest common divisor (GCD) of two long long integers.
 * @param a The first long long integer.
 * @param b The second long long integer.
 * @return The GCD of a and b.
 *
 * Algorithm: Euclidean algorithm.
 * Time Complexity: O(log(min(a, b)))
 * Space Complexity: O(log(min(a, b))) due to recursion (can be O(1) iteratively).
 */
ll gcd(ll a, ll b) {
  if (b == 0)
    return a;
  return gcd(b, a % b);
}
//###########################################################################################
/**
 * @brief Computes the least common multiple (LCM) of two long long integers.
 * @param a The first long long integer.
 * @param b The second long long integer.
 * @return The LCM of a and b.
 *
 * Algorithm: LCM(a, b) = (a * b) / GCD(a, b).
 * Time Complexity: O(log(min(a, b)))
 * Space Complexity: O(log(min(a, b)))
 */
ll lcm(ll a, ll b) { return a / gcd(a, b) * b; }
//###########################################################################################
/**
 * @brief Prints "YES" followed by a newline.
 * Time Complexity: O(1)
 * Space Complexity: O(1)
 */
void yes() {
  cout << "YES"
       << "\n";
}
//###########################################################################################
/**
 * @brief Prints "NO" followed by a newline.
 * Time Complexity: O(1)
 * Space Complexity: O(1)
 */
void no() {
  cout << "NO"
       << "\n";
}
//###########################################################################################
/**
 * @brief Main function of the program.
 *
 * This function reads the initial queue configuration (length N, simulation steps K, and the string S),
 * then simulates the boy-girl swapping process for K steps. It optimizes I/O and prints the final state.
 */
int main() {
  // Functional Utility: Optimizes C++ standard streams for faster input/output, crucial for competitive programming.
  fast
      //  clock_t launch=clock();
      //  string r = s1.substr(1, 3);
      //  while(clock()<CLOCKS_PER_SEC*10)
      //  using num = modnum<int(1e9)+7>;
      //  num len = s.size();
      //  double PI= 2 * acos(0.0); or M_PI
      ll n,
      k, m, ans = 0, flag = 0, temp = 0, f1 = 0, count = 0, f2 = 0, sum = 0,
            xx = 0, minn1 = big, minn2 = big, maxx1 = small, maxx2 = small;
  // Block Logic: Reads the queue length (n) and the number of simulation steps (k).
  cin >> n >> k;
  string s;
  // Block Logic: Reads the initial configuration of the queue (string s).
  cin >> s;
  /**
   * Block Logic: Simulates the boy-girl swapping process for 'k' time steps.
   * Invariant: At the start of each iteration, the string `s` represents the queue state
   * after `i` simulation steps.
   */
  lp(i, 0, k) {
    /**
     * Block Logic: Performs a single-pass sweep through the queue to resolve priority inversions.
     * Logic: Iterates from the end of the string towards the beginning. If a 'G' (Girl)
     * is found immediately followed by a 'B' (Boy), they swap positions. The inner loop
     * index `j` is decremented twice after a swap to ensure the newly moved 'G' is not
     * immediately re-evaluated in the same position and to correctly handle the next pair.
     * This ensures only one swap occurs per character per pass, simulating discrete steps.
     * Pre-condition: The string `s` contains 'B' and 'G' characters.
     */
    lpr(j, n, 1) {
      if (s[j] == 'G' && s[j - 1] == 'B') {
        /* code */
        swap(s[j], s[j - 1]);
        // Inline: Decrement `j` to account for the swapped element and avoid re-evaluating
        // the same pair in the next iteration of the inner loop, ensuring discrete movements.
        j--;
      }
    }
  }
  // Block Logic: Prints the final state of the queue after all simulation steps are completed.
  p1(s)

      // clog<<((long double)(clock()-launch)/CLOCKS_PER_SEC)<<"\n";
      return 0;
}
//////////*********************end of program*********************//////////
