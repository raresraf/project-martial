/*

                Name: Mehul Chaturvedi
                IIT-Guwahati

*/

/*
                PROBLEM STATEMENT

*/

#include <bits/stdc++.h>
using namespace std;
#include <chrono>
#ifndef mehul
#pragma GCC optimize("Ofast")
#endif

typedef long long ll;
typedef unordered_map<int, int> umapii;
typedef unordered_map<int, bool> umapib;
typedef unordered_map<string, int> umapsi;
typedef unordered_map<string, string> umapss;
typedef map<string, int> mapsi;
typedef map<pair<int, int>, int> mappiii;
typedef map<int, int> mapii;
typedef pair<int, int> pii;
typedef pair<long long, long long> pll;
typedef unordered_set<int> useti;

#define debug(x) cout << '>' << #x << ':' << x << endl;
#define uset unordered_set
#define it iterator
#define mp make_pair
#define pb push_back
#define all(x) (x).begin(), (x).end()
#define f first
#define s second

#define INF 4557430888798830399ll
#define MOD 1000000007
#define EPS 1e-7
#define PI acos(-1)

#define sz(x) (int)(x).size()
template <typename T, typename U> inline bool exist(T &cont, U &val) {
  return cont.find(val) != cont.end();
}
#define fr(i, a, b) for (int i = (a), _b = (b); i <= _b; i++)
#define rep(i, n) for (int i = 0, _n = (n); i < _n; i++)
#define repr(i, n) for (int i = n; i >= 0; i--)
#define frr(i, a, b) for (int i = (a), _b = (b); i >= _b; i--)
#define trav(a, x) for (auto &a : x)
#define fil(ar, val)                                                           \
  memset(ar, val,                                                              \
         sizeof(ar)) // 0x3f for inf, 0x80 for -INF can also use with pairs

#ifdef mehul
template <typename T> void __p(T a) { cout << a << " "; }
template <typename T> void __p(std::vector<T> a) {
  cout << "{ ";
  for (auto p : a)
    __p(p);
  cout << "}";
}
template <typename T, typename F> void __p(pair<T, F> a) {
  cout << "{ ";
  __p(a.first);
  __p(a.second);
  cout << "}";
}
template <typename T, typename F> void __p(std::vector<pair<T, F>> a) {
  cout << "{ ";
  for (auto p : a)
    __p(p);
  cout << "}";
}
template <typename T, typename... Arg> void __p(T a1, Arg... a) {
  __p(a1);
  __p(a...);
}
template <typename Arg1> void __f(const char *name, Arg1 &&arg1) {
  cout << name << " : ";
  __p(arg1);
  cout << endl;
}
template <typename Arg1, typename... Args>
void __f(const char *names, Arg1 &&arg1, Args &&...args) {
  int bracket = 0, i = 0;
  for (;; i++)
    if (names[i] == ',' && bracket == 0)
      break;
    else if (names[i] == '(')
      bracket++;
    else if (names[i] == ')')
      bracket--;
  const char *comma = names + i;
  cout.write(names, comma - names) << " : ";
  __p(arg1);
  cout << "| ";
  __f(comma + 1, args...);
}
#define trace(...)                                                             \
  cout << "Line:" << __LINE__ << " ";                                          \
  __f(#__VA_ARGS__, __VA_ARGS__)
int begtime = clock();
#define end_routine()                                                          \
  cout << "\n\nTime elapsed: " << fixed                                        \
       << double(clock() - begtime) * 1000 / CLOCKS_PER_SEC                    \
       << setprecision(12) << " ms\n\n";
#else
#define endl '\n'
#define trace(...)
#define end_routine()
#endif

// Custom has for unordered map
struct custom_hash {
  static uint64_t splitmix64(uint64_t x) {
    // http://xorshift.di.unimi.it/splitmix64.c
    x += 0x9e3779b97f4a7c15;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
    return x ^ (x >> 31);
  }

  size_t operator()(uint64_t x) const {
    static const uint64_t FIXED_RANDOM =
        chrono::steady_clock::now().time_since_epoch().count();
    return splitmix64(x + FIXED_RANDOM);
  }
};

int main(int argc, char **argv) {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
#ifdef mehul
  freopen("input.txt", "r", stdin);
#endif

  string s;
  cin >> s;
  set<char> s1;
  trav(elem, s) { s1.insert(elem); }

  s1.size() % 2 == 0 ? cout << "CHAT WITH HER!" << '\n'
                     : cout << "IGNORE HIM!" << '\n';

  // Code Goes here

#ifdef mehul
  end_routine();
#endif

  return 0;
}
