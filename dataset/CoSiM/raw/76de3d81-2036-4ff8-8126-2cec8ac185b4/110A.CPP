#include <iostream>
using namespace std;

int main() {
  long long int n, i, c = 0, r = 0, m = 1;
  cin >> n;
  i = n;
  while (i != 0) {
    r = i % 10;
    if (r == 4 || r == 7) {
      c++;
    }
    i = i / 10;
  }
  r = 0;
  if (c == 0) {
    m = 0;
  }
  long long int k = 0, l = 0;
  while (c != 0) {
    r = c % 10;
    k++;
    if (r == 4 || r == 7) {
      l++;
    }
    c /= 10;
  }
  if (l != k || m == 0) {
    cout << "NO";
  } else {
    cout << "YES";
  }
}
