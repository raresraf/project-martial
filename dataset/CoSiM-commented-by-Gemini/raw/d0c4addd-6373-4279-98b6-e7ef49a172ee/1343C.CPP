#include <bits/stdc++.h>
using namespace std;
#define F first
#define S second
#define PB push_back
#define MP make_pair
#define REP(i, a, b) for (int i = a; i < b; i++)
typedef long long ll;
#define int ll
typedef vector<int> vi;
typedef vector<long long> vl;
typedef pair<int, int> pi;
#define trace(x) cout << #x << "=" << x << "\n";
#define sz(x) (int)(x.size())
#define mod 1000000007
#define endl "\n"
template <typename F, typename S>
ostream &operator<<(ostream &os, const pair<F, S> &p) {
  return os << "(" << p.first << ", " << p.second << ")";
}

template <typename T> ostream &operator<<(ostream &os, const vector<T> &v) {
  os << "{";
  typename vector<T>::const_iterator it;
  for (it = v.begin(); it != v.end(); it++) {
    if (it != v.begin())
      os << ", ";
    os << *it;
  }
  return os << "}";
}

template <typename T> ostream &operator<<(ostream &os, const set<T> &v) {
  os << "[";
  typename set<T>::const_iterator it;
  for (it = v.begin(); it != v.end(); it++) {
    if (it != v.begin())
      os << ", ";
    os << *it;
  }
  return os << "]";
}

template <typename F, typename S>
ostream &operator<<(ostream &os, const map<F, S> &v) {
  os << "[";
  typename map<F, S>::const_iterator it;
  for (it = v.begin(); it != v.end(); it++) {
    if (it != v.begin())
      os << ", ";
    os << it->first << " = " << it->second;
  }
  return os << "]";
}

#define deb(x) cerr << #x << " = " << x << endl;

int32_t main() {
  ios::sync_with_stdio(false);
  int t;
  cin >> t;
  vi a;
  // set<int>S;
  while (t--) {
    // chck for n=1
    int n;
    cin >> n;
    a.clear();
    a.resize(n);
    REP(i, 0, n) { cin >> a[i]; }
    for (int i = 1; i < n; i++) {
      if (a[i] < 0 && a[i - 1] < 0) {
        if (a[i - 1] > a[i]) {
          a[i] = a[i - 1];
        }
      }
      if (a[i] > 0 && a[i - 1] > 0) {
        if (a[i - 1] > a[i]) {
          a[i] = a[i - 1];
        }
      }
    }
    for (int i = n - 2; i >= 0; i--) {
      if (a[i] < 0 && a[i + 1] < 0) {
        if (a[i + 1] > a[i]) {
          a[i] = a[i + 1];
        }
      }
      if (a[i] > 0 && a[i + 1] > 0) {
        if (a[i + 1] > a[i]) {
          a[i] = a[i + 1];
        }
      }
    }
    // for(int i=0;i<n;i++){
    //     S.insert(a[i]);
    // }if(i==0)
    int sum = 0;
    for (int i = 0; i < n; i++) {
      if (i == 0) {
        sum += a[i];
      } else {
        if (a[i] != a[i - 1]) {
          sum += a[i];
        }
      }
    }
    cout << sum << "\n";
  }

  return 0;
}