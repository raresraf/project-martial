#include <iostream>

using namespace std;

int main() {
  int n;

  cin >> n;

  if (!(1 <= n && n <= 100)) {
    return 0;
  } else {
    ;
  }

  if (n % 2 == 0 && n > 2) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  return 0;
}