#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, count = 0;
  cin >> n;
  if (n >= 1 && n <= 50) {
    string s;
    char c;
    c = getchar();
    getline(cin, s);

    for (int i = 1; i <= n; i++) {
      if (s[i] == s[i - 1])
        count++;
    }
    cout << count << endl;
  }
  return 0;
}
