#include <bits/stdc++.h>
using namespace std;
#define F first
#define S second
#define PB push_back
#define MP make_pair
#define REP(i, a, b) for (int i = a; i < b; i++)
typedef long long ll;
typedef vector<int> vi;
typedef pair<int, int> pi;
#define llp 1000000007
int n;

int main() {
  ios::sync_with_stdio(false);
  cin >> n;
  int i;
  int k = 0;
  for (i = 1;; i++) {
    k = i * (i + 1) / 2;
    if (k > n)
      break;
  }
  i--;
  cout << i << "\n";
  REP(j, 0, i - 1) { cout << j + 1 << " "; }
  cout << (n - (i * (i - 1) / 2)) << "\n";
  return 0;
}
