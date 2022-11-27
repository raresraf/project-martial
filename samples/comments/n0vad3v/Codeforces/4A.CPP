#include <iostream>

using namespace std;

bool check(int n) { return (n % 2 == 0); }

int main() {
  int n;
  while (cin >> n) {
    if (n == 2) {
      cout << "NO" << endl;
      continue;
    }
    if (check(n))
      cout << "YES" << endl;
    else
      cout << "NO" << endl;
  }
  return 0;
}
