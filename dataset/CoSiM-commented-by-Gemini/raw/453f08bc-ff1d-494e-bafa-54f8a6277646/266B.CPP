/**
 * @file 266B.CPP
 * @brief Solution for the "Queue at the School" problem (Codeforces 266B).
 *
 * This program simulates the state of a queue of boys ('B') and girls ('G')
 * over a given number of time steps. In each time step, any adjacent pair of
 * a boy followed by a girl ('BG') swaps places.
 *
 * @algorithm Iterative single-pass swap simulation.
 * @time_complexity O(T * N), where T is the number of time steps and N is the
 *                  length of the queue.
 * @space_complexity O(N) to store the queue as a string.
 */
// please refer from line no. 207
/////*31022618*/////
//_MONU KUMAR
#include <bits/stdc++.h>
using namespace std;
#define fast                                                                   
  ios_base::sync_with_stdio(false);                                            
  cin.tie(0);                                                                  
  cout.tie(0);
#define tc                                                                     
  ll t = 1;                                                                    
  cin >> t;                                                                    
  while (t--)
#define ll long long int
#define ld long double
#define ull unsigned long long
#define pll pair<ll, ll>
#define vi std::vector<ll>
#define vp vector<pll>
#define mll std::map<ll, ll>
#define mpl                                                                    
  std:                                                                         
  map<pll, ll>
#define p1(x) cout << x << "
";
#define p2(x, y) cout << x << " " << y << "
";
#define p3(x, y, z) cout << x << " " << y << " " << z << "
";
#define p4(x, y, z, w) cout << x << " " << y << " " << z << " " << w << "
";
#define trace(x)                                                               
  cout << #x << ": " << x << " "                                               
       << "
";
#define trace2(x, y)                                                           
  cout << #x << ": " << x << " | " << #y << ": " << y << "
";
#define trace3(x, y, z)                                                        
  cout << #x << ":" << x << " | " << #y << ": " << y << " | " << #z << ": "    
       << z << "
";
#define trace4(a, b, c, d)                                                     
  cout << #a << ": " << a << " | " << #b << ": " << b << " | " << #c << ": "   
       << c << " | " << #d << ": " << d << "
";
#define nL cout << "
";
#define lp(i, l, r) for (long long int i = l; i < r; i++)
#define lpr(i, r, l) for (long long int i = r - 1; i >= l; i--)
#define mem(a, x) memset(a, x, sizeof(a)) // set elements of array to some value
#define pi 3.1415926535897932384626
#define mod 1000000007
#define big 1e18
#define small -big
#define mp(a, b) make_pair(a, b)
#define pb(x) push_back(x)
#define ff first
#define ss second
#define siz 256
#define leadzero(a) __builtin_clz(a) // count leading zeroes
#define trailzero(a) __builtin_ctz(a) // count trailing zeroes
#define bitcount(a) __builtin_popcount(a) // count set bits
#define all(v) v.begin(), v.end()
#define lps(v, x) (lower_bound(all(v), x) - v.begin())
#define ups(v, x) (upper_bound(all(v), x) - v.begin())
#define ina(a, n)                                                              
  ll a[n];                                                                     
  lp(i, 0, n) cin >> a[i];
#define ona(a, n) lp(i, 0, n) cout << a[i] << " ";
//#########################################################################################
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
// modulo Multiplication
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

ll convert(string s) {
  bool o = true;
  ll ans = 0;
  for (ll i = 0; i < s.length(); i++) {
    ans = ans * 10 + (s[i] - 48);
  }
  return ans;
}
//###########################################################################################
// function to calculate a^b under mod...
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
// function to calculate inverse modulo under mod...assuming gcd is 1
ll invmod(ll a, ll modulo) {
  ll check = powmod(a, modulo - 2, modulo) % modulo;
  return check;
}
//###########################################################################################

ll max(ll a, ll b) {
  if (a > b) {
    /* code */
    return a;
  }
  return b;
}
//###########################################################################################
ll min(ll a, ll b) {
  if (a < b) {
    /* code */
    return a;
  }
  return b;
}
//###########################################################################################
ll min(ll a, ll b, ll c) {
  if (a < b) {
    /* code */
    return min(a, c);
  } else {
    return min(b, c);
  }
}
//###########################################################################################
ll max(ll a, ll b, ll c) {
  if (a < b) {
    /* code */
    return max(b, c);
  } else {
    return min(a, c);
  }
}
//###########################################################################################
ll min(ll a, int b) {
  if (a < b)
    return a;
  return b;
}
//###########################################################################################
ll min(int a, ll b) {
  if (a < b)
    return a;
  return b;
}
//###########################################################################################
ll max(ll a, int b) {
  if (a > b)
    return a;
  return b;
}
//###########################################################################################
ll max(int a, ll b) {
  if (a > b)
    return a;
  return b;
}
//###########################################################################################
ll gcd(ll a, ll b) {
  if (b == 0)
    return a;
  return gcd(b, a % b);
}
//###########################################################################################
ll lcm(ll a, ll b) { return a / gcd(a, b) * b; }
//###########################################################################################
void yes() {
  cout << "YES"
       << "
";
}
//###########################################################################################
void no() {
  cout << "NO"
       << "
";
}
//###########################################################################################
int main() {
  // A common competitive programming optimization to untie C++ streams from C stdio.
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
  cin >> n >> k;
  string s;
  cin >> s;

  /**
   * Block Logic: This is the main simulation loop, which runs for `k` time steps.
   * Invariant: At the beginning of each iteration, `s` holds the state of the
   * queue after `i` time steps have passed.
   */
  lp(i, 0, k) {
    /**
     * Block Logic: This loop performs a single pass over the queue for one time step.
     * It iterates from right to left to correctly handle the swaps in a single pass.
     * If iterated from left to right, a 'G' could move multiple spots in one pass,
     * which is incorrect.
     */
    lpr(j, n, 1) {
      // Pre-condition: Check for an adjacent Boy-Girl ('BG') pair.
      if (s[j] == 'G' && s[j - 1] == 'B') {
        swap(s[j], s[j - 1]);
        /**
         * Inline: After a swap, decrement `j`. This is crucial. It prevents the
         * now-swapped 'G' at index `j-1` from being considered for another
         * swap in the same time step. This ensures each person moves at most
         * one position per time step.
         */
        j--;
      }
    }
  }
  // Print the final state of the queue after all simulations.
  p1(s)

      // clog<<((long double)(clock()-launch)/CLOCKS_PER_SEC)<<"
";
      return 0;
}
//////////*********************end of program*********************//////////
