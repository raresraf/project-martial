#include <bits/stdc++.h>

using namespace std;

int main() {
  // turn off buffer sync between c and c++ streams. Decouple cin and cout
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

#ifndef ONLINE_JUDGE
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
  freopen("error.txt", "w", stderr);
#endif

  int n;
  cin >> n;

  if (n == 1) {
    cout << 1;
    return 0;
  }

  vector<int> coins(n);
  for (int i = 0; i < n; ++i)
    cin >> coins[i];

  sort(coins.begin(), coins.end(), greater<int>());
  for (int i = 0; i < n; ++i) {
    int my_sum = 0;
    for (int j = 0; j <= i; ++j)
      my_sum += coins[j];

    int their_sum = 0;
    for (int j = i + 1; j < n; ++j)
      their_sum += coins[j];

    if (my_sum > their_sum) {
      cout << i + 1;
      break;
    }
  }

  cerr << "time taken: " << (float)clock() / CLOCKS_PER_SEC << " secs" << endl;
  return 0;
}
