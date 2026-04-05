/**
 * @file 1366A.CPP
 * @brief Solution for a competitive programming problem.
 * @details The problem involves maximizing the number of items created from two
 * types of resources, 'A' and 'B', where items have different resource costs.
 *
 * Algorithm: Greedy, Constraint-based
 * The solution determines the maximum number of items possible by considering
 * the most limiting constraints. The maximum number of items, k, is given by
 * k = min(A, B, (A+B)/3). This formula is derived from the constraints of the
 * problem where you can produce 'x' items of type 1 (2A, 1B) and 'y' items of
 * type 2 (1A, 2B), such that k = x+y. The constraints are:
 * 1. 2x + y <= A
 * 2. x + 2y <= B
 * Summing these gives 3(x+y) <= A+B, so k <= (A+B)/3. Also, k=x+y <= 2x+y <= A
 * and k=x+y <= x+2y <= B. Thus, k is limited by the minimum of A, B, and (A+B)/3.
 *
 * Time Complexity: O(1) per test case.
 * Space Complexity: O(1) per test case.
 */
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 

using namespace std;

#define PF push_front
#define PB push_back
#define INS insert

#define FI first
#define SE second
#define all(obj) begin(obj), end(obj)
#define rall(obj) (obj).rbegin(), (obj).rend()

#define LB lower_bound
#define UB upper_bound

typedef long long ll;
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;
typedef vector<set<int>> vsi;

typedef unsigned long long ull;
typedef pair<int, bool> pib;
typedef pair<ll, bool> plb;
typedef vector<vector<int>> vvi;
typedef vector<set<pii>> vspi;
typedef vector<vector<pii>> vvpi;
typedef map<int, int> mii;
typedef map<ll, ll> mll;
typedef map<char, int> mci;
typedef map<string, int> msi;
typedef map<string, string> mss;

ll gcd(ll a, ll b) { return b ? gcd(b, a % b) : a; }

template <class T1, class T2> struct cmpf {
  bool rev;
  inline bool operator()(const pair<T1, T2> &a, const pair<T1, T2> &b) const {
    return (a.first < b.first) ^ rev;
  }
  cmpf(bool b = false) { rev = b; }
};
template <class T1, class T2> struct cmps {
  bool rev;
  inline bool operator()(const pair<T1, T2> &a, const pair<T1, T2> &b) const {
    return (a.second < b.second) ^ rev;
  }
  cmps(bool b = false) { rev = b; }
};

string to_string(const char &c) { return string(1, c); }
string to_string(const string &s) { return '"' + s + '"'; }
string to_string(const char *s) { return to_string((string)s); }
string to_string(const bool &b) { return (b ? "true" : "false"); }
string to_string(const vector<bool> &v, const string &sep = " ") {
  string s = "[";
  for (int i = 0; i < v.size(); i++) {
    if (i) {
      s += sep;
    }
    s += to_string(v[i]);
  }
  return s + "]";
}
template <size_t N> string to_string(const bitset<N> &v) {
  string s = "[";
  for (size_t i = 0; i < N; i++)
    s += v[i] ? '1' : '0';
  return s + "]";
}
template <class T1, class T2>
string to_string(const pair<T1, T2> &p, const string &sep = ",");
template <class T> string to_string(const T &v, const string &sep = " ") {
  bool first = true;
  string s = "{";
  for (const auto &x : v) {
    if (!first)
      s += sep;
    else
      first = false;
    s += to_string(x);
  }
  return s + "}";
}
template <class T>
string to_string(const T &v, const int &sz, const string &sep = " ") {
  string s = "[";
  for (int i = 0; i < sz; i++) {
    if (i) {
      s += sep;
    }
    s += to_string(v[i]);
  }
  return s + "]";
}
template <class T1, class T2>
string to_string(const pair<T1, T2> &p, const string &sep) {
  return "(" + to_string(p.first) + sep + to_string(p.second) + ")";
}

#define debug(obj) cout << #obj << ": " << to_string(obj) << endl;
#define debug1(obj, sz) cout << #obj << ": " << to_string(obj, sz) << endl;
#define debug2(obj, sz1, sz2)                                                  
  cout << "-----" << #obj << "-----
";                                        
  for (int i = 0; i < sz1; i++)                                                
    cout << to_string(obj[i], sz2) << " ";                                     
  cout << endl;
#define debug3(obj, sz1, sz2, sz3)                                             
  cout << "-----" << #obj << "-----
";                                        
  for (int i = 0; i < sz1; i++) {                                              
    for (int j = 0; j < sz2; j++)                                              
      cout << to_string(obj[i][j], sz3) << " ";                                
    cout << endl;                                                              
  }                                                                            
  cout << endl;

ll binpow(const ll &x, const ll &p, const ll &mod) {
  assert(mod > 0);
  if (p == 0)
    return 1;
  if (p == 1)
    return x % mod;
  if (p & 1)
    return (binpow((x * x) % mod, p / 2, mod) * x) % mod;
  return binpow((x * x) % mod, p / 2, mod) % mod;
}

template <class T> void chmn(T &a, const T &b) {
  if (a > b)
    a = b;
}
template <class T> void chmx(T &a, const T &b) {
  if (a < b)
    a = b;
}

void setIO() {
  ios_base::sync_with_stdio(false);
  cin.tie(nullptr);
  cout.tie(nullptr);
}

int chr[8] = {-1, 0, 1, 0, -1, -1, 1, 1};
int chc[8] = {0, 1, 0, -1, -1, 1, -1, 1};

const ll MOD = 1e9 + 7; // 998244353;
const int INIT = 1001001001;
const int MN = -1;

// int A[MN];
/**
 * @brief Solves a single test case for the resource allocation problem.
 */
void solve() {
  int A, B;
  cin >> A >> B;

  // The maximum number of items is the minimum of A, B, and (A+B)/3.
  // This code implements this logic in a clever way.
  // First, it computes the bound based on the total number of resources.
  int ans = (A + B) / 3;

  // Then, it refines the answer by considering the case where one resource
  // is much scarcer than the other. If A > 2*B, then B is the limiting
  // factor among the three constraints (A, B, (A+B)/3). A similar argument
  // holds if B > 2*A. In balanced cases, (A+B)/3 is the tightest bound.
  // This is a fast way to compute min({A, B, (A+B)/3}).
  if (A > 2 * B) {
    ans = B;
  }
  if (B > 2 * A) {
    ans = A;
  }

  cout << ans << endl;
}

int main() {
  // Functional Utility: Optimizes I/O operations for speed in competitive
  // programming by decoupling C++ streams from C standard streams.
  setIO();

  ll T = 1;
  cin >> T;

  // Block Logic: Process T independent test cases.
  while (T--)
    solve();

  return 0;
}
