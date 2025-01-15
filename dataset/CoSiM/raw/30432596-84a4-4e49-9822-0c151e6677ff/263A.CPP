#include <bits/stdc++.h>
using namespace std;

int main() {
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);

  int temp, x, y;
  for (int i = 1; i <= 5; i++)
    for (int j = 1; j <= 5; j++) {
      cin >> temp;
      if (temp) {
        x = i;
        y = j;
      }
    }

  cout << abs(x - 3) + abs(y - 3) << "\n";
}