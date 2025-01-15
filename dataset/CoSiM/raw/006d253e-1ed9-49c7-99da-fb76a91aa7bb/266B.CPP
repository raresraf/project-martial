#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, t, temp;
  cin >> n >> t;
  char bg[n + 1] = {};
  for (int i = 0; i < n; i++)
    cin >> bg[i];
  for (int j = 0; j < t; j++) {
    for (int i = 0; i < n - 1; i++) {
      if (bg[i] == 'B' && bg[i + 1] == 'G') {
        temp = bg[i];
        bg[i] = bg[i + 1];
        bg[i + 1] = temp;
        i++;
      }
    }
  }
  for (int i = 0; i < n; i++)
    cout << bg[i];

  return 0;
}
