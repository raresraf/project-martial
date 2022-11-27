#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  if ((n <= 100) && (n >= 1)) {
    if ((n % 2 == 0) && (n != 2))
      cout << "YES";
    else
      cout << "NO";
  }
  return 0;
}
