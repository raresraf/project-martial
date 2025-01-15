// Roma and Lucky Numbers - Codeforces 160 A
// Clara Moraes Dantas

#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, k;
  cin >> n >> k;

  int lucky = 0;
  for (int i = 0; i < n; i++) {
    string number;
    cin >> number;

    int l = 0;

    for (int j = 0; j < number.size(); j++) {
      if (number[j] == '7' || number[j] == '4')
        l++;
    }

    if (l <= k)
      lucky++;
  }

  cout << lucky << endl;

  return 0;
}
