#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef unordered_map<int, int> umapii;
typedef unordered_map<int, bool> umapib;
typedef unordered_map<string, int> umapsi;
typedef unordered_map<string, string> umapss;
typedef map<string, int> mapsi;
typedef map<pair<int, int>, int> mappiii;
typedef map<int, int> mapii;
typedef map<int, bool> mapib;
typedef pair<int, int> pii;
typedef pair<long long, long long> pll;
typedef unordered_set<int> useti;
typedef set<int> seti;
#define uset unordered_set
#define it iterator
#define mp make_pair
#define pb push_back
#define all(x) (x).begin(), (x).end()
#define allp(x) (x)->begin(), (x)->end()
#define f first
#define s second
#define MOD 1000000007

// cin.ignore(numeric_limits<streamsize>::max(), '\n'); -> Clears the input
// buffer

int main() {
  int q;
  cin >> q;

  while (q--) {
    ll l, r, d;
    cin >> l >> r >> d;

    ll mul = 1;

    if (d < l) {
      cout << d << endl;
      continue;
    }

    if (d > r) {
      cout << d << endl;
      continue;
    }

    else {
      ll num = r - l + 1;

      ll leap = num / d;

      if (leap == 0)
        leap++;

      ll my_num = d * leap;

      if (my_num > r) {
        cout << my_num << endl;
        continue;
      }

      else if (my_num + d > r) {
        cout << my_num + d << endl;
        continue;
      }

      else if (my_num + d + d > r) {
        cout << my_num + d + d << endl;
        continue;
      }
    }
  }

  return 0;
}
