#include <iostream>
using namespace std;

int main() {
  long long int x;
  int ct = 0;
  cin >> x;
  while (x) {
    if (x % 10 == 4 || x % 10 == 7)
      ct++;
    x /= 10;
  }
  if (ct == 4 || ct == 7)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
