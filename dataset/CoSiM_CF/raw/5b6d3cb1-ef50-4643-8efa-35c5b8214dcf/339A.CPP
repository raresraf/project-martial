#include <bits/stdc++.h>
using namespace std;

int main() {
  int a[1001];
  string s;
  cin >> s;

  int counter = 0;

  for (int i = 0; i < s.size(); ++i) {
    if (s[i] == '+') {
      continue;
    } else {
      a[counter++] = s[i] - '0';
    }
  }

  sort(a, a + counter);

  for (int i = 0; i < counter; ++i) {
    cout << a[i];
    if (i == counter - 1) {
      break;
    }

    cout << "+";
  }

  return 0;
}
