#include <bits/stdc++.h>
using namespace std;
int main() {
  string s;
  cin >> s;
  char a[s.length() / 2];
  int k = 0;
  for (int i = 0; i < s.length(); i = i + 2) {
    a[k++] = s[i];
  }
  sort(a, a + k);
  for (int i = 0; i < k; i++) {
    cout << a[i];
    if (i != k - 1)
      cout << "+";
  }
}
