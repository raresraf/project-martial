/*
  Yadi Qian
*/

#include <bits/stdc++.h>

using namespace std;

const double pi = acos(-1.0);
typedef pair<int, int> PII;
typedef vector<int> VI;
typedef vector<string> VS;
typedef vector<PII> VII;
typedef vector<VI> VVI;
typedef map<int, int> MPII;
typedef set<int> SETI;
typedef multiset<int> MSETI;
typedef long int int32;
typedef unsigned long int uint32;
typedef long long int int64;
typedef unsigned long long int uint64;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  int n, k;
  cin >> n >> k;

  int num = 0;
  int next = 0;
  int nnext;
  while (n > 0) {
    if (k > 0) {
      cin >> next;
      if (next != 0) {
        num++;
      }
    } else {
      cin >> nnext;
      if (nnext == next && nnext != 0)
        num++;
    }
    k--;
    n--;
  }

  cout << num;

  return 0;
}