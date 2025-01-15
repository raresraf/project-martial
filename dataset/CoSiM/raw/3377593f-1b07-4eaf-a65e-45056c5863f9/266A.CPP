#include <iostream>
#include <string>

using namespace std;

main() {
  int n, i, count = 0;
  string st;

  cin >> n;
  cin >> st;

  if (n == 1)
    cout << "0";

  else {
    for (i = 1; i < n; i++) {
      if (st[i] == st[i - 1])
        count++;
    }
    cout << count;
  }
}
