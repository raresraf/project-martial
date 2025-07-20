#include <bits/stdc++.h>

#define fastIO                                                                 \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(NULL);                                                               \
  cout.tie(NULL);
#define DPT                                                                    \
  for (int i = 0; i <= n; i++) {                                               \
    cout << "dp[" << i << "] = " << dp[i] << endl;                             \
  }
#define CP(s, x) cout << s << " " << x << endl;

using namespace std;
typedef long long ll;
typedef vector<int> vi;

const ll INF = 1e9;
const ll MOD = 1e9 + 7;
const ll MAX = 2e5 + 7;

int n, i;
vector<int> v;

int main() {
  fastIO;
  cin >> n;
  if (n == 1) {
    cout << 1 << endl << 1;
    return 0;
  }
  i = 1;
  while (true) {
    if (i <= n) {
      v.push_back(i);
      n -= i;
      i++;
    } else {
      v[v.size() - 1] += n;
      break;
    }
  }

  cout << v.size() << endl;

  for (i = 0; i < v.size(); i++) {
    cout << v[i] << " ";
  }

  return 0;
}