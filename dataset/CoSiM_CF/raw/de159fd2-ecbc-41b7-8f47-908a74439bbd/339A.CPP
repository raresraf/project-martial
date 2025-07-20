#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  int a[101];
  while (cin >> s) {
    int k = 0;
    for (int i = 0; i < s.size(); i += 2) {
      a[k] = s[i] - '0';
      k++;
    }

    sort(a, a + k);

    for (int i = 0; i < k; i++) {
      if (i == k - 1)
        cout << a[i] << endl;
      else
        cout << a[i] << "+";
    }
  }

  return 0;
}
