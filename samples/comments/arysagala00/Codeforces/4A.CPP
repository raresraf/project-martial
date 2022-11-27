#include <iostream>

using namespace std;

int main() {
  int n = 0;
  cin >> n;
  if (n % 2 == 1 || n == 2) {
    cout << "NO" << endl;
  } else if (n % 2 == 0) {
    cout << "YES" << endl;
  }
  return 0;
}
