#include <bits/stdc++.h>
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);

  int n, k, temp, cnt = 0;
  cin >> n >> k;

  vector<int> vec(n);
  for (int i = 0; i < n; i++)
    cin >> vec[i];

  for (int i = 0; i < n; i++)
    if (vec[k - 1] <= vec[i] && vec[i])
      cnt++;
    else
      break;

  cout << cnt << "\n";
}