#include <bits/stdc++.h>
using namespace std;
int main() {
  string a, b = "hello";
  cin >> a;
  int j = 0, count = 0;
  for (int i = 0; i < a.size(); i++) {
    if (a[i] == b[j]) {
      count++;
      j++;
      if (count == 5)
        break;
    }
  }
  if (count == 5)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
