#include <iostream>
using namespace std;
int main() {

  int n, f = 0;
  cin >> n;
  for (int i = 2; i <= n / 2; i += 2) {
    if (i % 2 == 0 && (n - i) % 2 == 0) {
      cout << "\nYES";
      f++;
      break;
    }
  }
  if (f == 0)
    cout << "\nNO";
  return 0;
}
