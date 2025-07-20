#include <bits/stdc++.h>

using namespace std;

int main() {
  string str, h = "hello";

  cin >> str;

  int len = str.size(), a = 0, count = 0;

  for (int i = 0; i < len; i++) {
    if (str[i] == h[a]) {
      a++;
      count++;
    }
  }
  if (count == 5)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
