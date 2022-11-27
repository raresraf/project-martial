#include <iostream>
using namespace std;

int main() {
  int n, m, k;
  cin >> n;
  if (n % 2 == 0) {
    k = n;
    m = n - 2;
    n = k - m;
    if (n % 2 == 0 && m % 2 == 0 && n != 0 && m != 0)
      cout << "YES";
    else
      cout << "NO";

  } else
    cout << "NO";

  return 0;
}
