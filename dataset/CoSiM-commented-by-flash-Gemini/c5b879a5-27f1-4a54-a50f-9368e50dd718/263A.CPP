#include <bits/stdc++.h>
using namespace std;

int main() {
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  int n = 5, v;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      cin >> v;
      if (v)
        cout << abs(2 - i) + abs(2 - j) << "\n";
    }
}
