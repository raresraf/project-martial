#include <iostream>
#include <string>

using namespace std;

int main() {
  int m = 1;
  string c;
  cin >> c;
  int n = 0;
  int l = c.length();
  for (int i = 0; i < l - 1; i++) {
    if (c[i] == c[i + 1]) {
      m++;
      if (m == 7) {
        cout << "YES";
        n = 1;
        break;
      }
    } else {
      m = 1;
    }
  }
  if (n == 0) {
    cout << "NO";
  }

  return 0;
}
