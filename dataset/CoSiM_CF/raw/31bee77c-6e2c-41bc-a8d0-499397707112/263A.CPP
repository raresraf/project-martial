#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef pair<int, int> ii;

int main() {
  cin.tie(0);
  cout.tie(0);
  ios_base::sync_with_stdio(0);
  bool x;
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 5; j++) {
      cin >> x;
      if (x) {
        cout << abs(i - 2) + abs(j - 2) << endl;
        return 0;
      }
    }
  return 0;
}
